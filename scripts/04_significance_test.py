"""
Significación estadística de la comparativa y combinación de referencia.

Reentrena SOLO la fase de test de las tres redes ganadoras (configuración y épocas ya
elegidas en la validación cruzada, leídas de reports/tables/comparison.csv; mismas semillas) y:

  1. Comprueba que reproduce las cifras de la comparativa.
  2. Test de Diebold-Mariano (pérdida cuadrática en $/barril, varianza HAC con h-1 retardos
     y corrección de Harvey-Leybourne-Newbold para previsiones a h días que se solapan):
     cada red frente al ingenuo y las redes entre sí, por horizonte.
  3. Fila de referencia "combinación": media de las predicciones de las tres redes.
     Es solo informativa: el modelo integrado en la aplicación sigue siendo una única red.

Uso:  poetry run python scripts/04_significance_test.py
Salidas: reports/tables/significance.csv, reports/significance.md, reports/tables/test_predictions.csv
"""
import re
import sys
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from scipy import stats

from src import config as C
from src.evaluation import horizon_metrics
from src.training import Dataset, make_split, train_ensemble

NAIVE = "Ingenuo"
COMBO = "Combinación"


def parse_cfg(label: str) -> dict:
    h, l, lr = re.match(r"h(\d+)·l(\d+)·lr([\d.e-]+)", label).groups()
    return {"hidden": int(h), "layers": int(l), "lr": float(lr)}


def diebold_mariano(e1: np.ndarray, e2: np.ndarray, h: int):
    """DM con pérdida cuadrática. d = e1² - e2²; estadístico < 0 -> el modelo 1 es mejor.
    Devuelve (estadístico corregido HLN, p-valor bilateral con t de Student n-1)."""
    d = e1 ** 2 - e2 ** 2
    n = len(d)
    dc = d - d.mean()
    lags = min(h - 1, n - 1)
    gamma = [np.dot(dc[k:], dc[:n - k]) / n for k in range(lags + 1)]
    var = gamma[0] + 2 * sum((1 - k / (lags + 1)) * gamma[k] for k in range(1, lags + 1))
    if var <= 0:                       # puede ocurrir con muchos retardos: varianza sin HAC
        var = gamma[0]
    dm = d.mean() / np.sqrt(var / n)
    hln = np.sqrt(max((n + 1 - 2 * h + h * (h - 1) / n) / n, 1e-12))
    stat = dm * hln
    return float(stat), float(2 * stats.t.sf(abs(stat), df=n - 1))


def stars(p: float) -> str:
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""


def main():
    comp = pd.read_csv(C.TABLES_DIR / "comparison.csv")
    ds = Dataset.load()
    split = make_split(ds, train_end=C.TEST_START, eval_start=C.TEST_START)
    price0 = ds.df["brent"].values[split.ev_origins]
    print(f"Test: {ds.index[split.ev_origins[0]].date()} -> {ds.index[-1].date()} "
          f"({len(split.ev_origins)} orígenes)")

    # ---------------------------------------------------------------- 1) Reentrenar y verificar
    preds, info = {}, {}
    for arch in C.ARCHITECTURES:
        row = comp[comp["red"] == arch].iloc[0]
        params, n_ep = parse_cfg(row["hiperparametros"]), int(row["epocas"])
        r = train_ensemble(ds, arch, params, split, C.ENSEMBLE_SEEDS, n_ep, eval_name="test")
        pred, mets, _ = r.at_epoch(ds, n_ep)
        preds[arch] = pred
        info[arch] = {"cfg": row["hiperparametros"], "epocas": n_ep, "params": r.n_params,
                      "segundos": r.seconds}
        for m in mets:
            ref = comp[(comp["red"] == arch) & (comp["horizonte"] == m["horizonte"])]["mejora_RMSE_%"].iloc[0]
            flag = "OK" if abs(m["mejora_RMSE_%"] - ref) < 0.05 else f"DIFERENTE (comparativa {ref:+.2f}%)"
            print(f"  {arch:4s} {m['horizonte']:5s} mejora {m['mejora_RMSE_%']:+.2f}%  {flag}")
    preds[COMBO] = np.mean([preds[a] for a in C.ARCHITECTURES], axis=0)

    # ---------------------------------------------------------------- 2) Errores en $/barril
    errors = {}                                   # nombre -> lista por horizonte (n_h,)
    masks = [~np.isnan(split.y_ev[:, j]) for j in range(len(C.HORIZONS))]
    real = [price0[m] * np.exp(split.y_ev[m, j]) for j, m in enumerate(masks)]
    errors[NAIVE] = [price0[m] - real[j] for j, m in enumerate(masks)]
    for name, p in preds.items():
        errors[name] = [price0[m] * np.exp(p[m, j]) - real[j] for j, m in enumerate(masks)]

    # ---------------------------------------------------------------- 3) Diebold-Mariano
    models = C.ARCHITECTURES + [COMBO]
    pairs = [(a, NAIVE) for a in models] + list(combinations(C.ARCHITECTURES, 2))
    rows = []
    for j, (hname, h) in enumerate(C.HORIZONS.items()):
        for a, b in pairs:
            stat, p = diebold_mariano(errors[a][j], errors[b][j], h)
            rmse_a, rmse_b = np.sqrt(np.mean(errors[a][j] ** 2)), np.sqrt(np.mean(errors[b][j] ** 2))
            rows.append({"horizonte": hname, "dias": h, "modelo": a, "frente_a": b,
                         "RMSE_modelo": rmse_a, "RMSE_rival": rmse_b,
                         "mejora_RMSE_%": 100 * (1 - rmse_a / rmse_b),
                         "DM": stat, "p_valor": p, "significativo_5%": p < 0.05,
                         "n": len(errors[a][j])})
    sig = pd.DataFrame(rows)
    sig.to_csv(C.TABLES_DIR / "significance.csv", index=False)

    # ---------------------------------------------------------------- 4) Fila de la combinación
    combo_mets = horizon_metrics(preds[COMBO], split.y_ev, price0)

    # ---------------------------------------------------------------- 5) Predicciones de test
    out = pd.DataFrame({"fecha_origen": ds.index[split.ev_origins].date, "brent_origen": price0})
    for j, (hname, h) in enumerate(C.HORIZONS.items()):
        out[f"real_{hname}"] = price0 * np.exp(split.y_ev[:, j])
        for name, p in preds.items():
            out[f"{name}_{hname}"] = price0 * np.exp(p[:, j])
    out.to_csv(C.TABLES_DIR / "test_predictions.csv", index=False, float_format="%.3f")

    write_markdown(sig, combo_mets, comp, info)
    print("\n" + (C.REPORTS_DIR / "significance.md").read_text(encoding="utf-8"))


def write_markdown(sig, combo_mets, comp, info):
    H = list(C.HORIZONS)
    L = ["# Significación estadística (Diebold-Mariano) y combinación de referencia", "",
         "Test de Diebold-Mariano sobre el periodo de test (2023 → hoy), pérdida cuadrática en $/barril, "
         "varianza HAC con h−1 retardos y corrección de Harvey-Leybourne-Newbold (las previsiones a h días "
         "se solapan). Celda = mejora del RMSE (%) del modelo de la fila frente al rival y p-valor. "
         "Significación: \\* 10 %, \\*\\* 5 %, \\*\\*\\* 1 %.", "",
         "**Aviso:** a largo plazo el test solo contiene ~3–4 años no solapados, así que su potencia es muy baja.", "",
         "## Cada modelo frente al ingenuo", "",
         "| Modelo | " + " | ".join(C.HORIZON_LABELS[h].split(" (")[0] for h in H) + " |",
         "|---|" + "---|" * len(H)]

    def cell(r):
        return f"{r['mejora_RMSE_%']:+.1f} % (p={r['p_valor']:.2f}){stars(r['p_valor'])}"

    for m in C.ARCHITECTURES + [COMBO]:
        cells = [cell(sig[(sig.modelo == m) & (sig.frente_a == NAIVE) & (sig.horizonte == h)].iloc[0]) for h in H]
        L.append(f"| {m} | " + " | ".join(cells) + " |")
    L += ["", "## Redes entre sí", "", "| Comparación | " + " | ".join(
        C.HORIZON_LABELS[h].split(" (")[0] for h in H) + " |", "|---|" + "---|" * len(H)]
    for a, b in combinations(C.ARCHITECTURES, 2):
        cells = [cell(sig[(sig.modelo == a) & (sig.frente_a == b) & (sig.horizonte == h)].iloc[0]) for h in H]
        L.append(f"| {a} vs {b} | " + " | ".join(cells) + " |")

    L += ["", "## Combinación de referencia (media de las tres redes)", "",
          "Solo informativa: no interviene en la selección ni se integra en la aplicación.", "",
          "| Modelo | Horizonte | MAE | RMSE | Mejora vs ingenuo (%) | Acierto dirección (%) | Parámetros | Tiempo (s) |",
          "|---|---|---|---|---|---|---|---|"]
    for a in C.ARCHITECTURES:
        for _, r in comp[comp.red == a].iterrows():
            L.append(f"| {a} ({info[a]['cfg']}, {info[a]['epocas']} ép.) | {r['horizonte']} | {r['MAE']:.2f} | "
                     f"{r['RMSE']:.2f} | {r['mejora_RMSE_%']:+.1f} | {r['acierto_direccion_%']:.1f} | "
                     f"{info[a]['params']:,} | {info[a]['segundos']:.1f} |".replace(",", ".").replace("lr0.001. ", "lr0.001, "))
    n_par = sum(i["params"] for i in info.values())
    secs = sum(i["segundos"] for i in info.values())
    for m in combo_mets:
        L.append(f"| **{COMBO}** | {m['horizonte']} | {m['MAE']:.2f} | {m['RMSE']:.2f} | "
                 f"{m['mejora_RMSE_%']:+.1f} | {m['acierto_direccion_%']:.1f} | {n_par:,} | {secs:.1f} |"
                 .replace(",", "."))
    (C.REPORTS_DIR / "significance.md").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
