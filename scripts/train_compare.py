"""
Comparativa RNN vs LSTM vs GRU para predecir el Brent a 5, 63 y 252 días hábiles.

Cada "modelo" es un CONJUNTO de 5 redes idénticas entrenadas con semillas distintas; su
predicción es la media. La dispersión entre miembros mide la estabilidad de la arquitectura.
Las redes se entrenan con error de Huber (los días extremos no dominan el aprendizaje).

  1. Validación cruzada walk-forward (3 folds, ventana creciente) con la MISMA rejilla de
     hiperparámetros para las tres redes. El número de épocas es un hiperparámetro más:
     cada red entrena CV_EPOCHS épocas, se registra la curva de validación del conjunto en
     cada época y se elige la época que minimiza la curva MEDIA de los 3 folds (2017-2022,
     varios regímenes de mercado). Así ninguna decisión depende de un único bloque extremo.
  2. Test (2023 -> hoy): cada red se entrena con 2000-2022 durante sus épocas elegidas y se
     compara con el modelo ingenuo ("precio futuro = precio de hoy").
  3. Selección: la red cuyo conjunto tiene menor error en la validación cruzada. Modelo de
     producción con todos los datos y las mismas épocas.
  4. Comparativa informativa a igual tamaño (las tres celdas con cada configuración).

Durante la ejecución se muestran las curvas de aprendizaje (error / error_ingenuo) por época
en consola y en TensorBoard:  poetry run tensorboard --logdir runs

Uso:
    poetry run python scripts/train_compare.py                    # experimento completo
    poetry run python scripts/train_compare.py --quick            # prueba rápida
    poetry run python scripts/train_compare.py --quiet            # sin una línea por época
    poetry run python scripts/train_compare.py --skip-equal-size  # sin la fase 4
    poetry run python scripts/train_compare.py --equal-size-only  # solo la fase 4
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from oilcast import config as C
from oilcast.pipeline import (Dataset, interval_coverage, make_split, residual_quantiles,
                              train_ensemble, train_production)
from oilcast.tracking import Tracker, cfg_label

COLORS = {"RNN": "#8c8c8c", "LSTM": "#1f77b4", "GRU": "#d62728"}


def cv_score(metrics: list[dict]) -> float:
    """Ratio medio RMSE_modelo / RMSE_ingenuo (menor = mejor; < 1 bate al ingenuo)."""
    return float(np.mean([1 - m["mejora_RMSE_%"] / 100 for m in metrics]))


def make_cv_folds(ds):
    return [make_split(ds, train_end=s, eval_start=s, eval_end=e, eval_cutoff=C.TEST_START)
            for s, e in C.CV_FOLDS]


def tracked(tracker: Tracker, phase: str, arch: str, params: dict, prefix: str = ""):
    return lambda name: tracker.run(phase, arch, params, f"{prefix}{name}")


def evaluate_config_cv(ds, arch, params, folds, tracker):
    """Entrena el conjunto en cada fold durante CV_EPOCHS épocas y construye las curvas.
    Devuelve dict con la época óptima (mínimo de la curva media) y sus métricas."""
    ens_curves, member_curves, runs = [], [], []
    for k, sp in enumerate(folds):
        r = train_ensemble(ds, arch, params, sp, C.ENSEMBLE_SEEDS, C.CV_EPOCHS,
                           track=tracked(tracker, "CV", arch, params, f"fold{k + 1}-"))
        ens_c, mem_c = [], []
        for e in range(1, C.CV_EPOCHS + 1):
            _, mets, mem = r.at_epoch(ds, e)
            ens_c.append(cv_score(mets))
            mem_c.append([cv_score(m) for m in mem])
        ens_curves.append(ens_c)
        member_curves.append(np.array(mem_c).T)          # (semillas, épocas)
        runs.append(r)
    ens_curves = np.array(ens_curves)                    # (folds, épocas)
    mean_curve = ens_curves.mean(axis=0)
    e_star = int(np.argmin(mean_curve)) + 1
    fold_scores = ens_curves[:, e_star - 1]
    seed_std = float(np.mean([mc[:, e_star - 1].std() for mc in member_curves]))
    oof_res = np.vstack([sp.y_ev - r.at_epoch(ds, e_star)[0] for sp, r in zip(folds, runs)])
    return {"params": params, "epoch": e_star, "score_cv": float(mean_curve[e_star - 1]),
            "fold_scores": fold_scores, "std_folds": float(fold_scores.std()), "std_semillas": seed_std,
            "mean_curve": mean_curve, "ens_curves": ens_curves, "member_curves": member_curves,
            "oof_res": oof_res, "seconds": sum(r.seconds for r in runs)}


def test_config(ds, arch, params, n_epochs, test_split, tracker, phase="TEST"):
    r = train_ensemble(ds, arch, params, test_split, C.ENSEMBLE_SEEDS, n_epochs, eval_name="test",
                       track=tracked(tracker, phase, arch, params))
    pred, mets, mem = r.at_epoch(ds, n_epochs)
    return r, pred, mets, mem


def run(quick: bool, verbose: bool, equal_size: bool):
    if quick:
        C.CV_EPOCHS = 3
        C.PARAM_GRID[:] = C.PARAM_GRID[:1]
        C.ENSEMBLE_SEEDS[:] = C.ENSEMBLE_SEEDS[:2]

    C.REPORTS_DIR.mkdir(exist_ok=True)
    C.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    tracker = Tracker(C.RUNS_DIR, verbose=verbose)
    ds = Dataset.load()
    print(f"Datos: {ds.index[0].date()} -> {ds.index[-1].date()}  ({len(ds.index)} días, "
          f"{ds.feats.shape[1]} variables, ventana {C.WINDOW}, conjuntos de {len(C.ENSEMBLE_SEEDS)} semillas, "
          f"{C.CV_EPOCHS} épocas en CV, error Huber)")
    print(f"Curvas en vivo: poetry run tensorboard --logdir {C.RUNS_DIR.name}  "
          f"(esta ejecución: {tracker.root.name})")
    print("Curvas = error / error_ingenuo  (<1 bate al ingenuo)\n")

    # ------------------------------------------------------------------ 1) CV walk-forward
    folds = make_cv_folds(ds)
    cv_rows, best, all_cv = [], {}, {}
    for arch in C.ARCHITECTURES:
        for params in C.PARAM_GRID:
            res = evaluate_config_cv(ds, arch, params, folds, tracker)
            all_cv[(arch, cfg_label(params))] = res
            row = {"red": arch, **params, "epoca_optima": res["epoch"], "score_cv": res["score_cv"],
                   "std_folds": res["std_folds"], "std_semillas": res["std_semillas"],
                   **{f"score_fold{k + 1}": s for k, s in enumerate(res["fold_scores"])},
                   "curva_media": [round(v, 4) for v in res["mean_curve"]],
                   "segundos": round(res["seconds"], 1)}
            cv_rows.append(row)
            curve = " ".join(f"{v:.3f}" for v in res["mean_curve"])
            print(f"[CV] {arch:4s} {cfg_label(params):17s} época*={res['epoch']:2d}  score={res['score_cv']:.3f}  "
                  f"folds={np.round(res['fold_scores'], 3).tolist()}  desv.semillas={res['std_semillas']:.3f}\n"
                  f"     curva media por época: {curve}", flush=True)
            if arch not in best or res["score_cv"] < best[arch]["score_cv"]:
                best[arch] = res
    pd.DataFrame(cv_rows).to_csv(C.REPORTS_DIR / "cv_results.csv", index=False)

    # ------------------------------------------------------------------ 2) Test 2023 -> hoy
    test_split = make_split(ds, train_end=C.TEST_START, eval_start=C.TEST_START)
    comp_rows, test_preds = [], {}
    for arch in C.ARCHITECTURES:
        b = best[arch]
        b["quantiles"] = residual_quantiles(b["oof_res"])
        r, pred, mets, mem = test_config(ds, arch, b["params"], b["epoch"], test_split, tracker)
        cov = interval_coverage(pred, test_split.y_ev, b["quantiles"])
        b["test_res"] = test_split.y_ev - pred
        b["test_metrics"] = mets
        test_preds[arch] = pred
        for j, m in enumerate(mets):
            member_imps = [mm[j]["mejora_RMSE_%"] for mm in mem]
            comp_rows.append({
                "red": arch, "horizonte": m["horizonte"], "dias": m["h"],
                "MAE": m["MAE"], "RMSE": m["RMSE"], "MSE": m["MSE"],
                "MAE_ingenuo": m["MAE_ingenuo"], "RMSE_ingenuo": m["RMSE_ingenuo"],
                "mejora_RMSE_%": m["mejora_RMSE_%"],
                "mejora_media_miembros_%": float(np.mean(member_imps)),
                "std_mejora_RMSE_%": float(np.std(member_imps)),
                "mejora_MAE_%": m["mejora_MAE_%"],
                "acierto_direccion_%": m["acierto_direccion_%"],
                "cobertura_IC80_%": cov[m["horizonte"]],
                "parametros": r.n_params, "miembros": len(r.models), "epocas": b["epoch"],
                "tiempo_entrenamiento_s": round(r.seconds, 1),
                "hiperparametros": cfg_label(b["params"]),
                "score_cv": b["score_cv"], "std_cv_folds": b["std_folds"],
                "std_cv_semillas": b["std_semillas"],
            })
        print(f"[TEST] {arch:4s} épocas={b['epoch']}  " + "  ".join(
            f"{r_['horizonte']}: {r_['mejora_RMSE_%']:+.1f}% (miembros {r_['mejora_media_miembros_%']:+.1f}"
            f" ±{r_['std_mejora_RMSE_%']:.1f})" for r_ in comp_rows[-len(mets):]), flush=True)
    comp = pd.DataFrame(comp_rows)
    comp.to_csv(C.REPORTS_DIR / "comparison.csv", index=False)

    # ------------------------------------------------------------------ 3) Selección
    winner = min(C.ARCHITECTURES, key=lambda a: best[a]["score_cv"])
    print(f"\n>>> Modelo elegido: {winner} {cfg_label(best[winner]['params'])} "
          f"(score CV del conjunto={best[winner]['score_cv']:.3f}, {best[winner]['epoch']} épocas)")
    write_markdown(comp, winner, ds)
    make_figures(ds, test_split, test_preds, comp, winner)
    make_learning_curves(tracker.history(), best)
    make_epoch_selection(all_cv, best)
    tracker.save(C.REPORTS_DIR / "training_history.csv")

    # Intervalo final: residuos fuera de muestra de CV + test (incluye el régimen más reciente)
    w = best[winner]
    quantiles = residual_quantiles(np.vstack([w["oof_res"], w["test_res"]]))
    print("Entrenando modelo de producción con todos los datos...")
    meta = train_production(winner, w["params"], w["epoch"], quantiles, w["test_metrics"], ds=ds,
                            track=tracked(tracker, "PROD", winner, w["params"]),
                            extra_meta={"score_cv": w["score_cv"],
                                        "selected_by": "menor error de validación cruzada del conjunto"})
    print(f"Modelo guardado en {C.MODELS_DIR} (datos hasta {meta['data_until']}, {meta['n_epochs']} épocas)")

    # ------------------------------------------------------------------ 4) Igual tamaño
    if equal_size:
        equal_size_comparison(ds, test_split, pd.DataFrame(cv_rows), tracker)
    tracker.save(C.REPORTS_DIR / "training_history.csv")


# ----------------------------------------------------------------------------
# Comparativa a igual tamaño (solo informativa, NO interviene en la selección)
# ----------------------------------------------------------------------------
EQUAL_SIZE_MARK = "## Comparativa a igual tamaño"


def _same_params(df: pd.DataFrame, params: dict) -> pd.Series:
    m = pd.Series(True, index=df.index)
    for k, v in params.items():
        m &= np.isclose(df[k].astype(float), float(v))
    return m


def equal_size_comparison(ds: Dataset, test_split, cv_df: pd.DataFrame, tracker: Tracker):
    """Evalúa las tres celdas con CADA configuración de la rejilla (cada una con sus épocas
    elegidas en CV) para separar el efecto de la arquitectura del efecto del tamaño."""
    rows = []
    for params in C.PARAM_GRID:
        for arch in C.ARCHITECTURES:
            cv = cv_df[(cv_df["red"] == arch) & _same_params(cv_df, params)].iloc[0]
            n_epochs = int(cv["epoca_optima"])
            r, _, mets, mem = test_config(ds, arch, params, n_epochs, test_split, tracker, phase="IGUAL")
            row = {"configuracion": cfg_label(params), "red": arch, "parametros": r.n_params,
                   "epocas": n_epochs, "score_cv": float(cv["score_cv"]),
                   "tiempo_entrenamiento_s": round(r.seconds, 1)}
            for j, name in enumerate(C.HORIZONS):
                row[f"mejora_{name}_%"] = mets[j]["mejora_RMSE_%"]
                row[f"std_{name}"] = float(np.std([mm[j]["mejora_RMSE_%"] for mm in mem]))
            rows.append(row)
            print(f"[IGUAL] {row['configuracion']:17s} {arch:4s} params={row['parametros']:6d} ép={n_epochs:2d}  "
                  + "  ".join(f"{n}: {row[f'mejora_{n}_%']:+.1f}% (±{row[f'std_{n}']:.1f})" for n in C.HORIZONS),
                  flush=True)
    eq = pd.DataFrame(rows)
    eq.to_csv(C.REPORTS_DIR / "comparison_equal_size.csv", index=False)
    write_equal_size_markdown(eq)
    return eq


def write_equal_size_markdown(eq: pd.DataFrame):
    t = pd.DataFrame({
        "Configuración": eq["configuracion"], "Red": eq["red"],
        "Parámetros (por red)": eq["parametros"].map(lambda v: f"{v:,}".replace(",", ".")),
        "Épocas": eq["epocas"],
        "Score CV": eq["score_cv"].map(lambda v: f"{v:.3f}"),
        **{f"Mejora {n} (%)": [f"{m:+.1f} (±{s:.1f})" for m, s in zip(eq[f"mejora_{n}_%"], eq[f"std_{n}"])]
           for n in C.HORIZONS},
        "Tiempo (s)": eq["tiempo_entrenamiento_s"],
    })
    section = [
        EQUAL_SIZE_MARK, "",
        "Las tres redes con exactamente la misma configuración (neuronas, capas y learning rate), "
        "cada una con sus épocas elegidas en la validación cruzada. "
        f"Mejora del RMSE del conjunto de {len(C.ENSEMBLE_SEEDS)} semillas frente al modelo ingenuo en test; "
        "entre paréntesis, la desviación entre miembros (estabilidad). "
        "Tabla informativa: no interviene en la selección del modelo.", "",
        t.to_markdown(index=False), "",
    ]
    path = C.REPORTS_DIR / "comparison.md"
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    text = text.split(EQUAL_SIZE_MARK)[0].rstrip() + "\n\n"
    path.write_text(text + "\n".join(section), encoding="utf-8")


def run_equal_size_only(verbose: bool):
    """Regenera solo la comparativa a igual tamaño reutilizando reports/cv_results.csv."""
    cv_path = C.REPORTS_DIR / "cv_results.csv"
    if not cv_path.exists():
        raise FileNotFoundError("Falta reports/cv_results.csv: ejecuta antes el experimento completo")
    ds = Dataset.load()
    test_split = make_split(ds, train_end=C.TEST_START, eval_start=C.TEST_START)
    equal_size_comparison(ds, test_split, pd.read_csv(cv_path), Tracker(C.RUNS_DIR, verbose=verbose))


# ----------------------------------------------------------------------------
# Informes y figuras
# ----------------------------------------------------------------------------
def write_markdown(comp: pd.DataFrame, winner: str, ds: Dataset):
    cols = ["red", "hiperparametros", "epocas", "horizonte", "MAE", "RMSE", "MAE_ingenuo", "RMSE_ingenuo",
            "mejora_RMSE_%", "mejora_media_miembros_%", "std_mejora_RMSE_%", "acierto_direccion_%",
            "cobertura_IC80_%", "parametros", "tiempo_entrenamiento_s"]
    t = comp[cols].copy()
    for c in cols[4:13]:
        t[c] = t[c].map(lambda v: f"{v:.2f}")
    lines = [
        "# Comparativa RNN vs LSTM vs GRU",
        "",
        f"Generado: {time.strftime('%Y-%m-%d %H:%M')} · Datos hasta {ds.index[-1].date()} · "
        f"Test: {C.TEST_START} → hoy · Errores en $/barril · Cada modelo = conjunto de "
        f"{len(C.ENSEMBLE_SEEDS)} semillas (`mejora_RMSE_%` del conjunto; `mejora_media_miembros_%` y "
        "`std_mejora_RMSE_%` de las redes individuales) · Épocas elegidas con la curva media de la CV",
        "",
        t.to_markdown(index=False),
        "",
        f"**Modelo elegido: {winner}** (menor error de validación cruzada walk-forward del conjunto).",
    ]
    (C.REPORTS_DIR / "comparison.md").write_text("\n".join(lines), encoding="utf-8")


def make_epoch_selection(all_cv: dict, best: dict):
    """Curva media de validación (conjunto, 3 folds) por época para cada configuración."""
    fig, axes = plt.subplots(1, len(C.ARCHITECTURES), figsize=(5 * len(C.ARCHITECTURES), 3.6), sharey=True)
    epochs = np.arange(1, C.CV_EPOCHS + 1)
    for ax, arch in zip(np.atleast_1d(axes), C.ARCHITECTURES):
        for (a, label), res in all_cv.items():
            if a != arch:
                continue
            chosen = label == cfg_label(best[arch]["params"])
            ax.plot(epochs, res["mean_curve"], lw=2.2 if chosen else 1, alpha=1 if chosen else .5,
                    color=COLORS[arch] if chosen else None, label=label + (" ✓" if chosen else ""))
            ax.scatter([res["epoch"]], [res["score_cv"]], s=30 if chosen else 12,
                       color=COLORS[arch] if chosen else "grey", zorder=3)
        ax.axhline(1.0, color="orange", lw=1)
        ax.set_title(arch)
        ax.set_xlabel("época")
        ax.grid(alpha=.3)
        ax.legend(fontsize=7)
    np.atleast_1d(axes)[0].set_ylabel("RMSE / RMSE ingenuo (media 3 folds)")
    fig.suptitle("Elección del número de épocas: mínimo de la curva media de validación del conjunto")
    fig.tight_layout()
    fig.savefig(C.FIGURES_DIR / "epoch_selection.png", dpi=120)
    plt.close(fig)


def make_learning_curves(hist: pd.DataFrame, best: dict):
    """Curvas de aprendizaje de la configuración elegida de cada red, por fold."""
    if hist.empty:
        return
    n_folds = len(C.CV_FOLDS)
    seed = C.ENSEMBLE_SEEDS[0]
    epochs = np.arange(1, C.CV_EPOCHS + 1)
    fig, axes = plt.subplots(len(C.ARCHITECTURES), n_folds, figsize=(4.2 * n_folds, 3.2 * len(C.ARCHITECTURES)),
                             sharey=True, squeeze=False)
    for i, arch in enumerate(C.ARCHITECTURES):
        b = best[arch]
        label = cfg_label(b["params"])
        for k in range(n_folds):
            ax = axes[i, k]
            h = hist[(hist["fase"] == "CV") & (hist["red"] == arch) & (hist["config"] == label)
                     & (hist["run"] == f"fold{k + 1}-s{seed}")]
            if not h.empty:
                ax.plot(h["epoca"], np.sqrt(h["entrenamiento"]), color=COLORS[arch], ls="--", lw=1,
                        label=f"entrenamiento (semilla {seed})")
            ax.plot(epochs, b["member_curves"][k][0], color=COLORS[arch], lw=1, alpha=.6,
                    label=f"validación (semilla {seed})")
            ax.plot(epochs, b["ens_curves"][k], color="black", lw=2, label="validación (conjunto)")
            ax.axvline(b["epoch"], color="green", ls=":", lw=1)
            ax.axhline(1.0, color="orange", lw=1, label="ingenuo = 1")
            ax.set_title(f"{arch} {label} · {C.CV_FOLDS[k][0][:4]}-{int(C.CV_FOLDS[k][1][:4]) - 1}", fontsize=9)
            ax.grid(alpha=.3)
            if k == 0:
                ax.set_ylabel("RMSE / RMSE ingenuo")
            if i == len(C.ARCHITECTURES) - 1:
                ax.set_xlabel("época")
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("Curvas de aprendizaje (línea verde = época elegida con la curva media de la CV)")
    fig.tight_layout()
    fig.savefig(C.FIGURES_DIR / "learning_curves.png", dpi=120)
    plt.close(fig)


def make_figures(ds, split, preds, comp, winner):
    dates = ds.index[split.ev_origins]
    price0 = ds.df["brent"].values[split.ev_origins]

    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
    for ax, (j, (name, h)) in zip(axes, enumerate(C.HORIZONS.items())):
        m = ~np.isnan(split.y_ev[:, j])
        tgt_dates = dates[m] + pd.offsets.BDay(h)
        ax.plot(tgt_dates, price0[m] * np.exp(split.y_ev[m, j]), color="black", lw=1.4, label="Real")
        ax.plot(tgt_dates, price0[m], color="orange", lw=1, ls="--", label="Ingenuo")
        for arch, p in preds.items():
            ax.plot(tgt_dates, price0[m] * np.exp(p[m, j]), color=COLORS[arch], lw=1, alpha=.85, label=arch)
        ax.set_title(f"{C.HORIZON_LABELS[name]} — predicción para la fecha objetivo")
        ax.set_ylabel("$/barril")
        ax.grid(alpha=.3)
    axes[0].legend(ncol=5, fontsize=8)
    fig.tight_layout()
    fig.savefig(C.FIGURES_DIR / "test_predictions.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    pv = comp.pivot(index="horizonte", columns="red", values="mejora_RMSE_%").loc[list(C.HORIZONS)]
    pv.plot.bar(ax=ax, color=[COLORS[c] for c in pv.columns])
    ax.axhline(0, color="black", lw=.8)
    ax.set_ylabel("Mejora de RMSE vs ingenuo (%)")
    ax.set_title(f"Comparativa en test (ganador: {winner})")
    ax.grid(alpha=.3, axis="y")
    plt.xticks(rotation=0)
    fig.tight_layout()
    fig.savefig(C.FIGURES_DIR / "improvement_vs_naive.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="prueba rápida con pocas épocas")
    ap.add_argument("--quiet", action="store_true", help="no mostrar una línea por época")
    ap.add_argument("--skip-equal-size", action="store_true", help="omitir la comparativa a igual tamaño")
    ap.add_argument("--equal-size-only", action="store_true",
                    help="regenerar solo la comparativa a igual tamaño (usa reports/cv_results.csv)")
    args = ap.parse_args()
    torch.set_num_threads(max(1, torch.get_num_threads()))
    if args.equal_size_only:
        run_equal_size_only(not args.quiet)
    else:
        run(args.quick, not args.quiet, not args.skip_equal_size)
