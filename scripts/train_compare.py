"""
Comparativa RNN vs LSTM vs GRU para predecir el Brent a 5, 63 y 252 días hábiles.

  1. Validación cruzada walk-forward (3 folds, ventana creciente) con la MISMA rejilla de
     hiperparámetros para las tres redes -> se elige la mejor configuración de cada una.
  2. Cada red se reentrena con todo el periodo 2000-2022 y se evalúa en test (2023 -> hoy)
     frente al modelo ingenuo ("precio futuro = precio de hoy").
  3. Se elige la red ganadora (mejor puntuación en validación cruzada), se calcula su
     intervalo de incertidumbre del 80 % y se entrena el modelo de producción con todos
     los datos.
  4. Comparativa informativa a igual tamaño: las tres celdas con cada configuración de la
     rejilla, para separar el efecto de la arquitectura del efecto del tamaño.

Uso:
    poetry run python scripts/train_compare.py                    # experimento completo
    poetry run python scripts/train_compare.py --quick            # prueba rápida (pocas épocas)
    poetry run python scripts/train_compare.py --equal-size-only  # solo la tabla a igual tamaño
"""
import argparse
import ast
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
from oilcast.models import count_params
from oilcast.pipeline import (Dataset, fit_and_eval, interval_coverage, make_split,
                              residual_quantiles, train_production)


def cv_score(metrics: list[dict]) -> float:
    """Ratio medio RMSE_modelo / RMSE_ingenuo (menor = mejor; < 1 bate al ingenuo)."""
    return float(np.mean([1 - m["mejora_RMSE_%"] / 100 for m in metrics]))


def run(quick: bool):
    torch.set_num_threads(max(1, torch.get_num_threads()))
    if quick:
        C.TRAIN_DEFAULTS.update(max_epochs=3, patience=2)
        C.PARAM_GRID[:] = C.PARAM_GRID[:1]

    C.REPORTS_DIR.mkdir(exist_ok=True)
    C.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    ds = Dataset.load()
    print(f"Datos: {ds.index[0].date()} -> {ds.index[-1].date()}  ({len(ds.index)} días, "
          f"{ds.feats.shape[1]} variables, ventana {C.WINDOW})")

    # ------------------------------------------------------------------ 1) CV walk-forward
    folds = [make_split(ds, train_end=s, eval_start=s, eval_end=e, eval_cutoff=C.TEST_START)
             for s, e in C.CV_FOLDS]
    cv_rows, best = [], {}
    for arch in C.ARCHITECTURES:
        for params in C.PARAM_GRID:
            fold_scores, epochs, oof_res, fold_secs = [], [], [], 0.0
            for k, sp in enumerate(folds):
                _, ep, secs, pred, mets = fit_and_eval(ds, arch, params, sp)
                fold_scores.append(cv_score(mets))
                epochs.append(ep)
                fold_secs += secs
                oof_res.append(sp.y_ev - pred)
            score = float(np.mean(fold_scores))
            row = {"red": arch, **params, "score_cv": score, "std_folds": float(np.std(fold_scores)),
                   **{f"score_fold{k + 1}": s for k, s in enumerate(fold_scores)},
                   "epocas": epochs, "segundos": round(fold_secs, 1)}
            cv_rows.append(row)
            print(f"[CV] {arch:4s} {params}  score={score:.3f} (±{row['std_folds']:.3f})  "
                  f"épocas={epochs}  {fold_secs:.0f}s")
            if arch not in best or score < best[arch]["score_cv"]:
                best[arch] = {"params": params, "score_cv": score, "std_folds": row["std_folds"],
                              "n_epochs": int(max(3, np.median(epochs))),
                              "oof_res": np.vstack(oof_res)}
    pd.DataFrame(cv_rows).to_csv(C.REPORTS_DIR / "cv_results.csv", index=False)

    # ------------------------------------------------------------------ 1b) Estabilidad (CV)
    # La mejor configuración de cada red se repite en la CV con otras semillas. Criterio de
    # selección = media + desviación del score entre semillas (premia precisión Y estabilidad)
    # sin mirar el periodo de test.
    for arch in C.ARCHITECTURES:
        b = best[arch]
        seed_scores = [b["score_cv"]]
        for seed in C.STABILITY_SEEDS[1:]:
            seed_scores.append(float(np.mean(
                [cv_score(fit_and_eval(ds, arch, b["params"], sp, seed=seed)[4]) for sp in folds])))
        b["score_cv_seeds"] = float(np.mean(seed_scores))
        b["std_cv_seeds"] = float(np.std(seed_scores))
        b["score_sel"] = b["score_cv_seeds"] + b["std_cv_seeds"]
        print(f"[CV-semillas] {arch:4s} scores={np.round(seed_scores, 3).tolist()}  "
              f"media={b['score_cv_seeds']:.3f} ±{b['std_cv_seeds']:.3f}  -> criterio={b['score_sel']:.3f}")

    # ------------------------------------------------------------------ 2) Test 2023 -> hoy
    test_split = make_split(ds, train_end=C.TEST_START, eval_start=C.TEST_START)
    price0 = ds.df["brent"].values[test_split.ev_origins]
    comp_rows, test_preds = [], {}
    for arch in C.ARCHITECTURES:
        b = best[arch]
        b["quantiles"] = residual_quantiles(b["oof_res"])
        # Se repite el entrenamiento con varias semillas: la media mide la precisión y la
        # desviación típica la ESTABILIDAD de cada red. La semilla principal (la primera)
        # es la que se usa para gráficos, residuos e intervalo.
        runs = []
        for seed in C.STABILITY_SEEDS:
            model, _, secs, pred, mets = fit_and_eval(ds, arch, b["params"], test_split,
                                                      n_epochs=b["n_epochs"], early_stop=False, seed=seed)
            runs.append({"pred": pred, "mets": mets, "secs": secs})
        main = runs[0]
        cov = interval_coverage(main["pred"], test_split.y_ev, b["quantiles"])
        b["test_res"] = test_split.y_ev - main["pred"]
        test_preds[arch] = main["pred"]
        mean_mets = []
        for j, m0 in enumerate(main["mets"]):
            per_seed = [r["mets"][j] for r in runs]
            avg = {k: float(np.mean([p[k] for p in per_seed])) for k in m0 if isinstance(m0[k], float)}
            mean_mets.append({**m0, **avg})
            comp_rows.append({
                "red": arch, "horizonte": m0["horizonte"], "dias": m0["h"],
                "MAE": avg["MAE"], "RMSE": avg["RMSE"], "MSE": avg["MSE"],
                "MAE_ingenuo": m0["MAE_ingenuo"], "RMSE_ingenuo": m0["RMSE_ingenuo"],
                "mejora_RMSE_%": avg["mejora_RMSE_%"],
                "std_mejora_RMSE_%": float(np.std([p["mejora_RMSE_%"] for p in per_seed])),
                "mejora_MAE_%": avg["mejora_MAE_%"],
                "acierto_direccion_%": avg["acierto_direccion_%"],
                "cobertura_IC80_%": cov[m0["horizonte"]],
                "parametros": count_params(model),
                "tiempo_entrenamiento_s": round(float(np.mean([r["secs"] for r in runs])), 1),
                "hiperparametros": f"hidden={b['params']['hidden']}, capas={b['params']['layers']}",
                "score_cv": b["score_cv_seeds"], "std_cv_semillas": b["std_cv_seeds"],
                "criterio_seleccion": b["score_sel"],
            })
        b["test_metrics"] = mean_mets
        print(f"[TEST] {arch:4s} " + "  ".join(
            f"{r['horizonte']}: RMSE={r['RMSE']:.2f} ({r['mejora_RMSE_%']:+.1f}% ±{r['std_mejora_RMSE_%']:.1f})"
            for r in comp_rows[-len(main['mets']):]))
    comp = pd.DataFrame(comp_rows)
    comp.to_csv(C.REPORTS_DIR / "comparison.csv", index=False)

    # ------------------------------------------------------------------ 3) Selección
    winner = min(C.ARCHITECTURES, key=lambda a: best[a]["score_sel"])
    print(f"\n>>> Modelo elegido: {winner} {best[winner]['params']} "
          f"(criterio CV media+desv={best[winner]['score_sel']:.3f})")
    write_markdown(comp, winner, ds)
    make_figures(ds, test_split, test_preds, comp, winner)

    # Intervalo final: residuos fuera de muestra de CV + test (incluye el régimen más reciente)
    w = best[winner]
    quantiles = residual_quantiles(np.vstack([w["oof_res"], w["test_res"]]))
    print("Entrenando modelo de producción con todos los datos...")
    meta = train_production(winner, w["params"], w["n_epochs"], quantiles, w["test_metrics"],
                            extra_meta={"score_cv": w["score_cv_seeds"], "std_cv_seeds": w["std_cv_seeds"], "selected_by": "media + desviación del score CV entre semillas"}, ds=ds)
    print(f"Modelo guardado en {C.MODELS_DIR} (datos hasta {meta['data_until']})")

    # ------------------------------------------------------------------ 4) Igual tamaño
    equal_size_comparison(ds, test_split, pd.DataFrame(cv_rows))


# ----------------------------------------------------------------------------
# Comparativa a igual tamaño (solo informativa, NO interviene en la selección)
# ----------------------------------------------------------------------------
EQUAL_SIZE_MARK = "## Comparativa a igual tamaño"


def equal_size_comparison(ds: Dataset, test_split, cv_df: pd.DataFrame):
    """Evalúa las tres celdas con CADA configuración de la rejilla (mismas neuronas y capas)
    para separar el efecto de la arquitectura del efecto del tamaño de la red."""
    rows = []
    for params in C.PARAM_GRID:
        for arch in C.ARCHITECTURES:
            cv = cv_df[(cv_df["red"] == arch) & (cv_df["hidden"] == params["hidden"])
                       & (cv_df["layers"] == params["layers"])].iloc[0]
            epochs = cv["epocas"] if isinstance(cv["epocas"], list) else ast.literal_eval(cv["epocas"])
            n_epochs = int(max(3, np.median(epochs)))       # mismo criterio que en la comparativa
            runs = [fit_and_eval(ds, arch, params, test_split, n_epochs=n_epochs,
                                 early_stop=False, seed=s) for s in C.STABILITY_SEEDS]
            row = {"configuracion": f"hidden={params['hidden']}, capas={params['layers']}",
                   "red": arch, "parametros": count_params(runs[0][0]),
                   "score_cv": float(cv["score_cv"]),
                   "tiempo_entrenamiento_s": round(float(np.mean([r[2] for r in runs])), 1)}
            for j, name in enumerate(C.HORIZONS):
                imps = [r[4][j]["mejora_RMSE_%"] for r in runs]
                row[f"mejora_{name}_%"] = float(np.mean(imps))
                row[f"std_{name}"] = float(np.std(imps))
            rows.append(row)
            print(f"[IGUAL] {row['configuracion']:18s} {arch:4s} params={row['parametros']:6d}  " + "  ".join(
                f"{n}: {row[f'mejora_{n}_%']:+.1f}% ±{row[f'std_{n}']:.1f}" for n in C.HORIZONS))
    eq = pd.DataFrame(rows)
    eq.to_csv(C.REPORTS_DIR / "comparison_equal_size.csv", index=False)
    write_equal_size_markdown(eq)
    return eq


def write_equal_size_markdown(eq: pd.DataFrame):
    t = pd.DataFrame({
        "Configuración": eq["configuracion"], "Red": eq["red"],
        "Parámetros": eq["parametros"].map(lambda v: f"{v:,}".replace(",", ".")),
        "Score CV": eq["score_cv"].map(lambda v: f"{v:.3f}"),
        **{f"Mejora {n} (%)": [f"{m:+.1f} ± {s:.1f}" for m, s in zip(eq[f"mejora_{n}_%"], eq[f"std_{n}"])]
           for n in C.HORIZONS},
        "Tiempo (s)": eq["tiempo_entrenamiento_s"],
    })
    section = [
        EQUAL_SIZE_MARK, "",
        "Las tres redes con exactamente la misma configuración (neuronas y capas). Mejora del RMSE "
        "frente al modelo ingenuo en test (media ± desviación de 3 semillas). Tabla informativa: "
        "no interviene en la selección del modelo.", "",
        t.to_markdown(index=False) if _has_tabulate() else t.to_string(index=False), "",
    ]
    path = C.REPORTS_DIR / "comparison.md"
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    text = text.split(EQUAL_SIZE_MARK)[0].rstrip() + "\n\n"
    path.write_text(text + "\n".join(section), encoding="utf-8")


def run_equal_size_only():
    """Regenera solo la comparativa a igual tamaño reutilizando reports/cv_results.csv."""
    cv_path = C.REPORTS_DIR / "cv_results.csv"
    if not cv_path.exists():
        raise FileNotFoundError("Falta reports/cv_results.csv: ejecuta antes el experimento completo")
    ds = Dataset.load()
    test_split = make_split(ds, train_end=C.TEST_START, eval_start=C.TEST_START)
    equal_size_comparison(ds, test_split, pd.read_csv(cv_path))


def write_markdown(comp: pd.DataFrame, winner: str, ds: Dataset):
    cols = ["red", "horizonte", "MAE", "RMSE", "MAE_ingenuo", "RMSE_ingenuo", "mejora_RMSE_%",
            "std_mejora_RMSE_%", "acierto_direccion_%", "cobertura_IC80_%", "parametros",
            "tiempo_entrenamiento_s"]
    t = comp[cols].copy()
    for c in cols[2:10]:
        t[c] = t[c].map(lambda v: f"{v:.2f}")
    lines = [
        "# Comparativa RNN vs LSTM vs GRU",
        "",
        f"Generado: {time.strftime('%Y-%m-%d %H:%M')} · Datos hasta {ds.index[-1].date()} · "
        f"Test: {C.TEST_START} → hoy · Errores en $/barril · Métricas de test = media de 3 semillas",
        "",
        t.to_markdown(index=False) if _has_tabulate() else t.to_string(index=False),
        "",
        f"**Modelo elegido: {winner}** (menor «error relativo al ingenuo + inestabilidad entre semillas» en la "
        "validación cruzada walk-forward).",
    ]
    (C.REPORTS_DIR / "comparison.md").write_text("\n".join(lines), encoding="utf-8")


def _has_tabulate():
    try:
        import tabulate  # noqa: F401
        return True
    except ImportError:
        return False


def make_figures(ds, split, preds, comp, winner):
    dates = ds.index[split.ev_origins]
    price0 = ds.df["brent"].values[split.ev_origins]
    colors = {"RNN": "#8c8c8c", "LSTM": "#1f77b4", "GRU": "#d62728"}

    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
    for ax, (j, (name, h)) in zip(axes, enumerate(C.HORIZONS.items())):
        m = ~np.isnan(split.y_ev[:, j])
        tgt_dates = dates[m] + pd.offsets.BDay(h)
        ax.plot(tgt_dates, price0[m] * np.exp(split.y_ev[m, j]), color="black", lw=1.4, label="Real")
        ax.plot(tgt_dates, price0[m], color="orange", lw=1, ls="--", label="Ingenuo")
        for arch, p in preds.items():
            ax.plot(tgt_dates, price0[m] * np.exp(p[m, j]), color=colors[arch], lw=1, alpha=.85, label=arch)
        ax.set_title(f"{C.HORIZON_LABELS[name]} — predicción para la fecha objetivo")
        ax.set_ylabel("$/barril")
        ax.grid(alpha=.3)
    axes[0].legend(ncol=5, fontsize=8)
    fig.tight_layout()
    fig.savefig(C.FIGURES_DIR / "test_predictions.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    pv = comp.pivot(index="horizonte", columns="red", values="mejora_RMSE_%").loc[list(C.HORIZONS)]
    pv.plot.bar(ax=ax, color=[colors[c] for c in pv.columns])
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
    ap.add_argument("--equal-size-only", action="store_true",
                    help="regenerar solo la comparativa a igual tamaño (usa reports/cv_results.csv)")
    args = ap.parse_args()
    if args.equal_size_only:
        run_equal_size_only()
    else:
        run(args.quick)
