"""Herramientas de decisión para la aplicación: simulador a cualquier número de días,
probabilidad de precio, escenarios históricos y backtest de las señales del modelo.

Nada de esto entrena ni modifica la red: reutiliza las tres predicciones del modelo de
producción (5, 63 y 252 días), los errores reales que cometió fuera de muestra
(reports/tables/forecast_errors.csv), la serie histórica del Brent y las predicciones de test
(reports/tables/test_predictions.csv). La web solo combina estos resultados con lo que escribe
el usuario (días, precio objetivo, cantidad).

Simulador entre horizontes (interpolación, no una predicción directa de la red):
  - Centro: interpolación monótona (PCHIP) de la log-rentabilidad prevista entre los puntos
    (0, 0), (5, m5), (63, m63) y (252, m252).
  - Incertidumbre: cuantiles empíricos del error en cada horizonte evaluado, normalizados por
    la raíz del tiempo (la incertidumbre de un precio crece con √t) e interpolados linealmente
    entre horizontes. En 5, 63 y 252 días reproduce exactamente el rango de las tarjetas.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator

from .config import HORIZONS, TABLES_DIR

MAX_H = max(HORIZONS.values())
# Percentiles de la distribución de error que se exportan (en %)
LEVELS = [1, 2.5, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 97.5, 99]
HIST_LEVELS = [5, 10, 25, 50, 75, 90, 95]

# Crisis del Brent desde 2000: se aplica al precio actual el movimiento real más extremo de cada
# una (en su dirección) para el número de días elegido, empezando dentro del periodo indicado.
SCENARIOS = [
    {"id": "2008", "title": "Crisis financiera de 2008", "direction": "down", "period": ("2008-07-01", "2009-02-28"),
     "text": "Quiebra de Lehman Brothers y recesión mundial: el Brent pasó de 144 $ a menos de 40 $."},
    {"id": "2014", "title": "Exceso de oferta 2014-2016", "direction": "down", "period": ("2014-06-01", "2016-01-31"),
     "text": "Auge del shale en EE. UU. y la OPEP sin recortar producción."},
    {"id": "2020", "title": "Pandemia de 2020", "direction": "down", "period": ("2020-01-01", "2020-04-30"),
     "text": "Confinamientos y guerra de precios entre Arabia Saudí y Rusia."},
    {"id": "2022", "title": "Invasión de Ucrania (2022)", "direction": "up", "period": ("2021-12-01", "2022-06-30"),
     "text": "Sanciones a Rusia sobre una demanda ya recuperada tras la COVID."},
    {"id": "2026", "title": "Irán y Ormuz (2026)", "direction": "up", "period": ("2026-01-15", "2026-04-30"),
     "text": "Interrupción del tráfico por el estrecho de Ormuz: el mayor shock de oferta de la historia."},
]


# ----------------------------------------------------------------------------- simulador
def load_forecast_errors() -> pd.DataFrame:
    return pd.read_csv(TABLES_DIR / "forecast_errors.csv", parse_dates=["fecha_origen"])


def error_quantiles(errors: pd.DataFrame) -> dict[str, np.ndarray]:
    """Percentiles LEVELS del error fuera de muestra (log-rentabilidad real - predicha) por horizonte."""
    return {name: np.quantile(errors[f"error_{name}"].dropna(), np.array(LEVELS) / 100) for name in HORIZONS}


def simulator_grid(forecast: dict, errors: pd.DataFrame) -> dict:
    """Predicción central y percentiles para cada horizonte de 1 a 252 días hábiles."""
    p0 = forecast["last_price"]
    hs = np.array(list(HORIZONS.values()), dtype=float)
    m = np.array([np.log(forecast["horizons"][name]["price"] / p0) for name in HORIZONS])
    center = PchipInterpolator(np.r_[0.0, hs], np.r_[0.0, m])(np.arange(1, MAX_H + 1))

    q = error_quantiles(errors)
    norm = np.vstack([q[name] / np.sqrt(h) for name, h in HORIZONS.items()])   # (3, niveles)
    days = np.arange(1, MAX_H + 1)
    grid = np.empty((MAX_H, len(LEVELS)))
    for j in range(len(LEVELS)):
        grid[:, j] = np.interp(days, hs, norm[:, j]) * np.sqrt(days)        # constante por debajo de 5 d
    last = pd.Timestamp(forecast["last_date"])
    return {
        "levels": LEVELS,
        "days": days.tolist(),
        "target_dates": [(last + pd.offsets.BDay(int(h))).strftime("%Y-%m-%d") for h in days],
        "center_lr": center.tolist(),
        "error_q": grid.tolist(),       # error_q[h-1][j]: percentil LEVELS[j] del error a h días
        "anchors": {name: h for name, h in HORIZONS.items()},
        "n_errors": {name: int(errors[f"error_{name}"].notna().sum()) for name in HORIZONS},
        "error_period": [errors["fecha_origen"].min().strftime("%Y-%m-%d"),
                         errors["fecha_origen"].max().strftime("%Y-%m-%d")],
    }


# ----------------------------------------------------------------------------- escenarios
def historical_moves(brent: pd.Series) -> dict:
    """Percentiles de la variación logarítmica del Brent a N días (N = 1..252) desde 2000."""
    lb = np.log(brent.to_numpy())
    q = [np.quantile(lb[n:] - lb[:-n], np.array(HIST_LEVELS) / 100) for n in range(1, MAX_H + 1)]
    return {"levels": HIST_LEVELS, "q": np.array(q).tolist(), "since": brent.index[0].strftime("%Y-%m-%d")}


def crisis_scenarios(brent: pd.Series) -> list[dict]:
    """Para cada crisis y cada N, el movimiento real más extremo a N días (en su dirección)
    empezando dentro del periodo de la crisis, con su fecha de inicio."""
    lb = np.log(brent.to_numpy())
    idx = brent.index
    out = []
    for sc in SCENARIOS:
        a, b = idx.searchsorted(pd.Timestamp(sc["period"][0])), idx.searchsorted(pd.Timestamp(sc["period"][1]))
        moves, starts = [], []
        for n in range(1, MAX_H + 1):
            s = np.arange(a, min(b, len(lb) - n))
            if len(s) == 0:
                moves.append(None)
                starts.append(None)
                continue
            r = lb[s + n] - lb[s]
            k = int(np.argmin(r) if sc["direction"] == "down" else np.argmax(r))
            moves.append(float(r[k]))
            starts.append(idx[s[k]].strftime("%Y-%m-%d"))
        out.append({**{k: v for k, v in sc.items() if k != "period"}, "period": list(sc["period"]),
                    "moves": moves, "starts": starts})
    return out


# ----------------------------------------------------------------------------- backtest
def backtest_trades(tp: pd.DataFrame, brent: pd.Series, model: str = "RNN") -> dict:
    """Operaciones sin solapar con las predicciones de test (modelo entrenado hasta 2022).

    En cada operación se abre una posición al precio del día de origen y se cierra h días hábiles
    después. Señales: la del modelo (sube / baja) y, como referencia, «seguir la tendencia» (signo
    de la variación de los h días anteriores). La web aplica los costes y compone el capital."""
    out = {}
    for name, h in [("corto", HORIZONS["corto"]), ("medio", HORIZONS["medio"])]:
        d = tp.dropna(subset=[f"real_{name}"]).reset_index(drop=True)
        pos = brent.index.get_indexer(d["fecha_origen"])
        prev = brent.to_numpy()[np.clip(pos - h, 0, None)]
        sig_model = np.sign(np.log(d[f"{model}_{name}"] / d["brent_origen"])).to_numpy()
        sig_trend = np.sign(np.log(d["brent_origen"].to_numpy() / prev))
        ret = np.log(d[f"real_{name}"] / d["brent_origen"]).to_numpy()

        def trades(offset: int):
            # Rentabilidad simple de cada operación (una venta gana -(P1/P0 - 1)), compuesta
            ii = np.arange(offset, len(d), h)
            return ii, float(np.prod(1 + sig_model[ii] * np.expm1(ret[ii])) - 1)

        ii, _ = trades(0)
        by_offset = np.array([trades(o)[1] for o in range(h)]) * 100
        end_dates = (d["fecha_origen"] + pd.offsets.BDay(h)).dt.strftime("%Y-%m-%d")
        out[name] = {
            "h": h,
            "trades": [{"start": d["fecha_origen"].iloc[i].strftime("%Y-%m-%d"), "end": end_dates.iloc[i],
                        "entry": float(d["brent_origen"].iloc[i]), "exit": float(d[f"real_{name}"].iloc[i]),
                        "model": int(sig_model[i]), "trend": int(sig_trend[i]), "ret_lr": float(ret[i])}
                       for i in ii],
            # Robustez: resultado (largo/corto con el modelo, sin costes) según el día de inicio
            "offsets": {"n": int(h), "median": float(np.median(by_offset)),
                        "min": float(by_offset.min()), "max": float(by_offset.max())},
        }
    return out
