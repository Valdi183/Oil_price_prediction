"""Evaluación fuera de muestra a partir de las predicciones de test guardadas
(reports/test_predictions.csv, generado por scripts/significance.py): métricas completas
(MAE, MSE, RMSE, R²), análisis de errores por año y por régimen de volatilidad, error móvil
frente al ingenuo y peores fallos. No entrena nada: solo compara predicción con realidad.

Convenciones:
- Todas las cifras en $/barril. "Ingenuo" = el precio futuro es el precio de hoy.
- Cada fila del test es un ORIGEN (día en que se hace la predicción); su valor real es el
  precio h días hábiles después. Los grupos (año, régimen) se forman por fecha de origen.
- R² sobre el precio: 1 - SSE / SST (SST respecto a la media del precio real en el test).
  Con precios muy autocorrelacionados casi todo lo "explica" el precio de hoy, así que se
  acompaña siempre del R² fuera de muestra frente al ingenuo (Campbell y Thompson, 2008):
  R²_OOS = 1 - MSE_modelo / MSE_ingenuo  (> 0: el modelo elimina error cuadrático del ingenuo).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import HORIZONS, REPORTS_DIR

NAIVE = "Ingenuo"
NETS = ["RNN", "LSTM", "GRU", "Combinación"]
REGIMES = ["Calma", "Normal", "Turbulencia"]


def load_test_predictions() -> pd.DataFrame:
    return pd.read_csv(REPORTS_DIR / "test_predictions.csv", parse_dates=["fecha_origen"])


def _arrays(tp: pd.DataFrame, horizon: str, model: str):
    """(máscara de filas con valor real, real, predicción) para un modelo y horizonte."""
    real = tp[f"real_{horizon}"].to_numpy()
    pred = tp["brent_origen"].to_numpy() if model == NAIVE else tp[f"{model}_{horizon}"].to_numpy()
    m = ~np.isnan(real)
    return m, real, pred


def metrics(real: np.ndarray, pred: np.ndarray, naive: np.ndarray) -> dict:
    err, err_n = pred - real, naive - real
    mse, mse_n = float(np.mean(err ** 2)), float(np.mean(err_n ** 2))
    sst = float(np.sum((real - real.mean()) ** 2))
    moved = real != naive
    return {
        "n": int(len(real)),
        "MAE": float(np.mean(np.abs(err))), "MSE": mse, "RMSE": float(np.sqrt(mse)),
        "R2_precio": 1 - float(np.sum(err ** 2)) / sst if sst > 0 else float("nan"),
        "R2_OOS_%": 100 * (1 - mse / mse_n),
        "mejora_RMSE_%": 100 * (1 - np.sqrt(mse) / np.sqrt(mse_n)),
        "acierto_direccion_%": 100 * float(np.mean(np.sign(pred[moved] - naive[moved])
                                                   == np.sign(real[moved] - naive[moved]))),
    }


def full_metrics(tp: pd.DataFrame) -> pd.DataFrame:
    """MAE, MSE, RMSE, R² sobre el precio y R² fuera de muestra, por modelo y horizonte."""
    rows = []
    p0 = tp["brent_origen"].to_numpy()
    for hz, h in HORIZONS.items():
        for model in NETS + [NAIVE]:
            m, real, pred = _arrays(tp, hz, model)
            rows.append({"modelo": model, "horizonte": hz, "dias": h, **metrics(real[m], pred[m], p0[m])})
    return pd.DataFrame(rows)


def regimes(tp: pd.DataFrame, brent: pd.Series):
    """Régimen de mercado en cada origen según la volatilidad realizada del Brent a 20 días
    (desviación típica de la log-rentabilidad diaria, anualizada, en %), por terciles del test.
    Devuelve (etiqueta por origen, (umbral calma, umbral turbulencia), volatilidad por origen)."""
    vol = np.log(brent).diff().rolling(20).std() * np.sqrt(252) * 100
    v = vol.reindex(tp["fecha_origen"]).to_numpy()
    q1, q2 = np.nanquantile(v, [1 / 3, 2 / 3])
    lab = np.where(v <= q1, REGIMES[0], np.where(v <= q2, REGIMES[1], REGIMES[2]))
    return pd.Series(lab, index=tp.index, name="regimen"), (float(q1), float(q2)), pd.Series(v, index=tp.index)


def by_group(tp: pd.DataFrame, groups: pd.Series, kind: str, model: str = "RNN") -> pd.DataFrame:
    """Métricas del modelo frente al ingenuo dentro de cada grupo de orígenes."""
    rows = []
    p0 = tp["brent_origen"].to_numpy()
    for hz in HORIZONS:
        m_all, real, pred = _arrays(tp, hz, model)
        for g in pd.unique(groups):
            m = m_all & (groups.to_numpy() == g)
            if m.sum() < 20:
                continue
            met = metrics(real[m], pred[m], p0[m])
            rmse_n = float(np.sqrt(np.mean((p0[m] - real[m]) ** 2)))
            rows.append({"tipo": kind, "grupo": str(g), "horizonte": hz, "modelo": model,
                         "n": met["n"], "MAE": met["MAE"], "RMSE": met["RMSE"], "RMSE_ingenuo": rmse_n,
                         "mejora_RMSE_%": met["mejora_RMSE_%"], "acierto_direccion_%": met["acierto_direccion_%"]})
    return pd.DataFrame(rows)


def rolling_ratio(tp: pd.DataFrame, model: str = "RNN") -> dict:
    """Cociente RMSE_modelo / RMSE_ingenuo en una ventana móvil de orígenes (< 1: bate al ingenuo).
    Ventana = max(63, h) días para que cada punto agrupe suficientes previsiones."""
    out = {}
    p0 = tp["brent_origen"].to_numpy()
    for hz, h in HORIZONS.items():
        m, real, pred = _arrays(tp, hz, model)
        e2 = pd.Series(np.where(m, (pred - real) ** 2, np.nan))
        n2 = pd.Series(np.where(m, (p0 - real) ** 2, np.nan))
        w = max(63, h)
        ratio = np.sqrt(e2.rolling(w, min_periods=w // 2).mean() / n2.rolling(w, min_periods=w // 2).mean())
        out[hz] = {"window": w, "ratio": ratio.to_numpy()}
    return out


def worst_errors(tp: pd.DataFrame, k: int = 5, model: str = "RNN") -> pd.DataFrame:
    """Los k mayores errores absolutos del modelo por horizonte, separados al menos h/2 días
    entre sí para no repetir el mismo episodio (las previsiones vecinas se solapan)."""
    rows = []
    p0 = tp["brent_origen"].to_numpy()
    dates = tp["fecha_origen"].to_numpy()
    for hz, h in HORIZONS.items():
        m, real, pred = _arrays(tp, hz, model)
        idx = np.where(m)[0]
        order = idx[np.argsort(-np.abs(pred[idx] - real[idx]))]
        picked: list[int] = []
        gap = max(5, h // 2)
        for i in order:
            if all(abs(i - j) >= gap for j in picked):
                picked.append(i)
            if len(picked) == k:
                break
        for i in picked:
            rows.append({"horizonte": hz, "fecha_origen": pd.Timestamp(dates[i]).date(),
                         "fecha_objetivo": (pd.Timestamp(dates[i]) + pd.offsets.BDay(h)).date(),
                         "precio_origen": p0[i], "real": real[i], "prediccion": pred[i],
                         "error": pred[i] - real[i], "error_ingenuo": p0[i] - real[i]})
    return pd.DataFrame(rows)
