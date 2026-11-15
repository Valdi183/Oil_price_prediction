"""Carga, limpieza, ingeniería de variables y preprocesado (escalado + ventanas)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .config import DATA_PATH, H_VALUES, WINDOW

# Variables explicativas que entran en la red (todas diarias)
FEATURES = [
    "log_brent",        # nivel del precio (log)
    "ret_brent",        # rentabilidad diaria del Brent
    "mom_brent_20",     # momentum ~1 mes
    "mom_brent_60",     # momentum ~3 meses
    "vol_brent_20",     # volatilidad realizada ~1 mes
    "gap_ma200",        # distancia a la media de 200 días (tendencia)
    "ret_wti",          # otra referencia de crudo
    "spread_brent_wti", # diferencial Brent-WTI relativo
    "ret_gas",          # gas natural Henry Hub
    "ret_usd_eur",      # tipo de cambio dólar/euro
    "ret_cny_usd",      # tipo de cambio yuan/dólar
    "ret_dxy",          # índice del dólar
    "ret_oro",          # oro (activo refugio)
    "ret_cobre",        # cobre (actividad industrial)
    "log_vix",          # volatilidad de los mercados
    "tipo_bce",         # tipo de interés del BCE
]
# Variables de rentabilidad: se recortan (winsorizan) para limitar outliers
RETURN_FEATURES = [f for f in FEATURES if f.startswith("ret_")]


def load_raw(path=DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return df.sort_index()


def clean(raw: pd.DataFrame) -> pd.DataFrame:
    """Limpieza de valores faltantes y outliers 'físicos'."""
    df = raw[~raw.index.duplicated(keep="last")].copy()

    # 1) La EIA publica el Brent spot con unos días de retraso: el script de descarga
    #    rellena esos últimos días con el último valor (ffill). Se eliminan las filas
    #    finales en las que el Brent está repetido pero el futuro sí se ha movido.
    if "brent_fut_Close" in df:
        while len(df) > 2:
            last, prev = df.iloc[-1], df.iloc[-2]
            if last["brent"] == prev["brent"] and last["brent_fut_Close"] != prev["brent_fut_Close"]:
                df = df.iloc[:-1]
            else:
                break

    # 2) WTI negativo (20/04/2020, -37 $): anomalía técnica del vencimiento del futuro,
    #    no un precio económico real. Se trata como dato faltante.
    for col in ["wti", "wti_fut_Close"]:
        if col in df:
            df.loc[df[col] <= 0, col] = np.nan

    # 3) Festivos distintos en cada mercado -> último valor disponible
    df = df.ffill().bfill()
    return df


def _logret(s: pd.Series, n: int = 1) -> pd.Series:
    return np.log(s).diff(n)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    b = df["brent"]
    f = pd.DataFrame(index=df.index)
    f["log_brent"] = np.log(b)
    f["ret_brent"] = _logret(b)
    f["mom_brent_20"] = _logret(b, 20)
    f["mom_brent_60"] = _logret(b, 60)
    f["vol_brent_20"] = f["ret_brent"].rolling(20).std()
    f["gap_ma200"] = np.log(b / b.rolling(200, min_periods=50).mean())
    f["ret_wti"] = _logret(df["wti"])
    f["spread_brent_wti"] = (b - df["wti"]) / b
    f["ret_gas"] = _logret(df["gas_natural"])
    f["ret_usd_eur"] = _logret(df["usd_por_eur"])
    f["ret_cny_usd"] = _logret(df["cny_por_usd"])
    f["ret_dxy"] = _logret(df["indice_dolar"])
    f["ret_oro"] = _logret(df["oro"])
    f["ret_cobre"] = _logret(df["cobre"])
    f["log_vix"] = np.log(df["vix"])
    f["tipo_bce"] = df["tipo_bce"]
    f = f.replace([np.inf, -np.inf], np.nan)
    return f[FEATURES]


def build_targets(brent: pd.Series) -> pd.DataFrame:
    """Objetivo: rentabilidad logarítmica a h días, log(P[t+h] / P[t]).
    Predecir 0 equivale al modelo ingenuo ('el precio futuro = el precio de hoy')."""
    lb = np.log(brent)
    return pd.DataFrame({f"h{h}": lb.shift(-h) - lb for h in H_VALUES}, index=brent.index)


def prepare_dataset(path=DATA_PATH):
    """Devuelve (df_limpio, features, targets) alineados y sin NaN al inicio."""
    df = clean(load_raw(path))
    feats = build_features(df)
    first_valid = feats.dropna().index[0]
    df, feats = df.loc[first_valid:], feats.loc[first_valid:]
    targets = build_targets(df["brent"])
    return df, feats, targets


class Preprocessor:
    """Winsorización de rentabilidades + StandardScaler. Se ajusta SOLO con entrenamiento."""

    def __init__(self, clip_q: float = 0.005):
        self.clip_q = clip_q
        self.bounds: dict[str, tuple[float, float]] = {}
        self.scaler = StandardScaler()
        self.target_std: np.ndarray | None = None

    def fit(self, feats: pd.DataFrame, targets: pd.DataFrame | None = None):
        for c in RETURN_FEATURES:
            self.bounds[c] = (feats[c].quantile(self.clip_q), feats[c].quantile(1 - self.clip_q))
        self.scaler.fit(self._clip(feats).values)
        if targets is not None:
            # Solo se divide por la desviación típica (sin restar la media) para que
            # la salida 0 de la red siga significando "precio sin cambios".
            self.target_std = np.nanstd(targets.values, axis=0)
        return self

    def _clip(self, feats: pd.DataFrame) -> pd.DataFrame:
        out = feats.copy()
        for c, (lo, hi) in self.bounds.items():
            out[c] = out[c].clip(lo, hi)
        return out

    def transform(self, feats: pd.DataFrame) -> np.ndarray:
        return self.scaler.transform(self._clip(feats).values).astype(np.float32)

    def scale_targets(self, y: np.ndarray) -> np.ndarray:
        return (y / self.target_std).astype(np.float32)

    def unscale_targets(self, y: np.ndarray) -> np.ndarray:
        return y * self.target_std


def make_windows(x: np.ndarray, origins: np.ndarray, window: int = WINDOW) -> np.ndarray:
    """Para cada posición de origen t devuelve x[t-window+1 : t+1] -> (n, window, n_features)."""
    idx = origins[:, None] + np.arange(-window + 1, 1)[None, :]
    return x[idx]


def masked_targets(targets: pd.DataFrame, origins: np.ndarray, cutoff_pos: int | None) -> np.ndarray:
    """Objetivos de los orígenes dados. Si t + h >= cutoff_pos, el objetivo usaría precios
    posteriores al corte (fuga de información) y se marca como NaN."""
    y = targets.values[origins].copy()
    if cutoff_pos is not None:
        for j, h in enumerate(H_VALUES):
            y[origins + h >= cutoff_pos, j] = np.nan
    return y
