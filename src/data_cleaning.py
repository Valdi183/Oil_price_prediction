"""Carga de los datos en bruto y limpieza (valores faltantes y outliers).

data/raw/brent_dataset_raw.csv  ->  clean()  ->  data/processed/brent_dataset_clean.csv
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import RAW_DATA_PATH


def load_raw(path=RAW_DATA_PATH) -> pd.DataFrame:
    """Datos tal cual los deja scripts/01_download_data.py (FRED + Yahoo Finance, calendario del Brent)."""
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


def load_clean(path=RAW_DATA_PATH) -> pd.DataFrame:
    return clean(load_raw(path))
