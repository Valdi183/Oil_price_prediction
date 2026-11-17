"""
Validación fuera de muestra del modelo: ¿funciona, cuándo acierta y cuándo se equivoca?

El modelo se entrenó con datos hasta el 31/12/2022 y aquí se comparan sus predicciones con
los valores reales que se produjeron después (2023 → hoy). No entrena nada: usa las
predicciones de test guardadas por scripts/significance.py.

Calcula:
  1. MAE, MSE, RMSE, R² sobre el precio y R² fuera de muestra frente al ingenuo, por red y horizonte.
  2. Errores de la RNN por año y por régimen de volatilidad del mercado (calma / normal / turbulencia).
  3. Error móvil de la RNN frente al ingenuo (cuándo lo bate y cuándo no).
  4. Los peores fallos de la RNN por horizonte, con su fecha.

Uso:  poetry run python scripts/error_analysis.py
Salidas: reports/test_metrics.csv, reports/error_by_period.csv, reports/worst_errors.csv,
         reports/error_analysis.md, reports/figures/error_analysis.png
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from oilcast import config as C  # noqa: E402
from oilcast.evaluation import (NAIVE, NETS, REGIMES, by_group, full_metrics, load_test_predictions,  # noqa: E402
                                regimes, rolling_ratio, worst_errors)
from oilcast.pipeline import Dataset  # noqa: E402

COLORS = {"RNN": "#d9902f", "LSTM": "#1f77b4", "GRU": "#d62728", "Combinación": "#8c8c8c"}


def fmt(x, nd=2):
    return "—" if pd.isna(x) else f"{x:,.{nd}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def signed(x, nd=1):
    return ("+" if x > 0 else "") + fmt(x, nd)


def main():
    tp = load_test_predictions()
    ds = Dataset.load()
    brent = ds.df["brent"]

    met = full_metrics(tp)
    met.to_csv(C.REPORTS_DIR / "test_metrics.csv", index=False)

    years = tp["fecha_origen"].dt.year.astype(str)
    reg, (q1, q2), _ = regimes(tp, brent)
    periods = pd.concat([by_group(tp, years, "año"), by_group(tp, reg, "régimen")], ignore_index=True)
    periods.to_csv(C.REPORTS_DIR / "error_by_period.csv", index=False)

    worst = worst_errors(tp)
    worst.to_csv(C.REPORTS_DIR / "worst_errors.csv", index=False)

    roll = rolling_ratio(tp)
    make_figure(tp, roll)
    write_markdown(met, periods, worst, roll, tp, (q1, q2))
    print((C.REPORTS_DIR / "error_analysis.md").read_text(encoding="utf-8"))


def make_figure(tp, roll):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    p0 = tp["brent_origen"].to_numpy()
    for j, (hz, h) in enumerate(C.HORIZONS.items()):
        real = tp[f"real_{hz}"].to_numpy()
        pred = tp[f"RNN_{hz}"].to_numpy()
        m = ~np.isnan(real)
        ax = axes[0, j]
        ax.scatter(real[m], pred[m], s=6, alpha=.45, color=COLORS["RNN"], label="RNN")
        ax.scatter(real[m], p0[m], s=4, alpha=.25, color="black", label="Ingenuo")
        lo, hi = np.nanmin(real[m]) * .95, np.nanmax(real[m]) * 1.05
        ax.plot([lo, hi], [lo, hi], color="black", lw=.8, ls="--")
        ax.set_xlim(lo, hi), ax.set_ylim(lo, hi)
        ax.set_title(f"{C.HORIZON_LABELS[hz].split(' (')[0]} ({h} d): real vs predicho")
        ax.set_xlabel("Precio real ($/barril)"), ax.set_ylabel("Predicción ($/barril)")
        ax.grid(alpha=.3)
        ax.legend(fontsize=8, loc="upper left")

        ax = axes[1, j]
        r = roll[hz]
        ax.plot(tp["fecha_origen"], r["ratio"], color=COLORS["RNN"], lw=1.3)
        ax.axhline(1, color="black", lw=.8, ls="--")
        ax.fill_between(tp["fecha_origen"], r["ratio"], 1, where=r["ratio"] < 1, color="#2e9e5b", alpha=.25,
                        label="bate al ingenuo")
        ax.fill_between(tp["fecha_origen"], r["ratio"], 1, where=r["ratio"] >= 1, color="#c0392b", alpha=.2,
                        label="peor que el ingenuo")
        ax.set_title(f"RMSE RNN / RMSE ingenuo (ventana {r['window']} d)")
        ax.set_ylabel("cociente (< 1 = mejor)")
        ax.grid(alpha=.3)
        ax.legend(fontsize=8, loc="upper left")
        ax.tick_params(axis="x", labelrotation=30)
    fig.suptitle("Validación fuera de muestra de la RNN (entrenada hasta 2022, evaluada 2023 → hoy)", fontsize=13)
    fig.tight_layout()
    fig.savefig(C.FIGURES_DIR / "error_analysis.png", dpi=120)
    plt.close(fig)


def write_markdown(met, periods, worst, roll, tp, q):
    H = list(C.HORIZONS)
    hl = {h: C.HORIZON_LABELS[h].split(" (")[0] for h in H}
    L = ["# Validación fuera de muestra: métricas y análisis de errores", "",
         f"Modelo entrenado con datos hasta el 31/12/2022 y evaluado con lo que ocurrió después "
         f"({tp['fecha_origen'].min():%d/%m/%Y} → {tp['fecha_origen'].max():%d/%m/%Y}, "
         f"{len(tp)} días de origen). Errores en $/barril. *Ingenuo* = el precio futuro es el de hoy.", "",
         "## 1. Métricas por red y horizonte", "",
         "- **R² sobre el precio**: 1 − SSE/SST. Con precios tan autocorrelacionados, el precio de hoy ya "
         "\"explica\" casi todo a corto plazo (por eso el ingenuo también saca un R² alto), y a medio y largo plazo "
         "puede ser negativo para todos los modelos si el precio se movió poco en el test y luego dio un salto.",
         "- **R² fuera de muestra (R²ₒₒₛ)** frente al ingenuo (Campbell y Thompson, 2008): 1 − MSE/MSE_ingenuo. "
         "Es la métrica habitual en predicción financiera: > 0 significa que el modelo elimina parte del error "
         "cuadrático de suponer que el precio no cambia.", ""]
    for hz in H:
        L += [f"### {hl[hz]} ({C.HORIZONS[hz]} días hábiles)", "",
              "| Modelo | MAE | MSE | RMSE | R² precio | R²ₒₒₛ vs ingenuo | Acierto dirección |",
              "|---|---|---|---|---|---|---|"]
        for _, r in met[met.horizonte == hz].iterrows():
            name = f"**{r.modelo}**" if r.modelo == "RNN" else r.modelo + (" (referencia)" if r.modelo == "Combinación" else "")
            oos = "—" if r.modelo == NAIVE else f"{signed(r['R2_OOS_%'])} %"
            acc = "—" if r.modelo == NAIVE else f"{fmt(r['acierto_direccion_%'], 1)} %"
            L.append(f"| {name} | {fmt(r.MAE)} | {fmt(r.MSE)} | {fmt(r.RMSE)} | {fmt(r.R2_precio, 3)} | {oos} | {acc} |")
        L.append("")

    L += ["## 2. ¿Cuándo acierta y cuándo se equivoca? (RNN frente al ingenuo)", "",
          "Mejora del RMSE (%) de la RNN frente al ingenuo dentro de cada grupo de orígenes; entre paréntesis, "
          "RMSE de la RNN en $/barril. Positivo = la RNN se equivoca menos que el ingenuo.", "",
          "### Por año de origen", "", "| Año | " + " | ".join(hl[h] for h in H) + " |", "|---|" + "---|" * len(H)]

    def cell(df, g, hz):
        r = df[(df.grupo == g) & (df.horizonte == hz)]
        if r.empty:
            return "— (sin valor real aún)"
        r = r.iloc[0]
        return f"{signed(r['mejora_RMSE_%'])} % ({fmt(r.RMSE)})"

    yrs = periods[periods.tipo == "año"]
    for g in sorted(yrs.grupo.unique()):
        L.append(f"| {g} | " + " | ".join(cell(yrs, g, hz) for hz in H) + " |")
    regs = periods[periods.tipo == "régimen"]
    L += ["", "### Por régimen de volatilidad del mercado", "",
          f"Volatilidad realizada del Brent a 20 días (anualizada) en el día de origen, por terciles del test: "
          f"**calma** ≤ {fmt(q[0], 0)} %, **normal** {fmt(q[0], 0)}–{fmt(q[1], 0)} %, **turbulencia** > {fmt(q[1], 0)} %.", "",
          "| Régimen | " + " | ".join(hl[h] for h in H) + " |", "|---|" + "---|" * len(H)]
    for g in REGIMES:
        L.append(f"| {g} | " + " | ".join(cell(regs, g, hz) for hz in H) + " |")

    L += ["", "### Error móvil frente al ingenuo", "",
          "Porcentaje del periodo de test en el que la RNN bate al ingenuo (cociente de RMSE móvil < 1):", ""]
    for hz in H:
        r = roll[hz]["ratio"]
        ok = r[~np.isnan(r)]
        L.append(f"- {hl[hz]} (ventana {roll[hz]['window']} d): {fmt(100 * np.mean(ok < 1), 0)} % del tiempo")

    L += ["", "## 3. Peores fallos de la RNN", "",
          "Los 5 mayores errores absolutos por horizonte (episodios distintos). *Error* = predicción − real.", "",
          "| Horizonte | Origen | Objetivo | Precio origen | Real | Predicción | Error RNN | Error ingenuo |",
          "|---|---|---|---|---|---|---|---|"]
    for _, r in worst.iterrows():
        L.append(f"| {hl[r.horizonte]} | {r.fecha_origen:%d/%m/%Y} | {r.fecha_objetivo:%d/%m/%Y} | {fmt(r.precio_origen)} | "
                 f"{fmt(r.real)} | {fmt(r.prediccion)} | {signed(r.error, 2)} | {signed(r.error_ingenuo, 2)} |")
    L += ["", "![Análisis de errores](figures/error_analysis.png)", ""]
    (C.REPORTS_DIR / "error_analysis.md").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
