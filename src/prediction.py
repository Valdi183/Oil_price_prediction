"""Inferencia: carga el modelo de producción y predice los tres horizontes desde el último dato.

Es lo único que se ejecuta en la actualización diaria: los datos nuevos pasan por el modelo ya
entrenado (descargado de Hugging Face Hub), sin reentrenar.
"""
from __future__ import annotations

import json
import pickle

import numpy as np
import pandas as pd
import torch

from .config import HORIZONS, PROD_META, PROD_MODEL, PROD_PRE
from .feature_engineering import make_windows
from .models import RecurrentForecaster, predict
from .training import Dataset

# Los modelos publicados antes de la reestructuración del repositorio (paquete "oilcast")
# guardaron el preprocesador como oilcast.data.Preprocessor: se traduce a su nueva ubicación
# para no tener que reentrenar.
_LEGACY_MODULES = {"oilcast.data": "src.feature_engineering"}


class _CompatUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        return super().find_class(_LEGACY_MODULES.get(module, module), name)


def load_production():
    meta = json.loads(PROD_META.read_text(encoding="utf-8"))
    with open(PROD_PRE, "rb") as fh:
        pre = _CompatUnpickler(fh).load()
    p = meta["params"]
    members = []
    for sd in torch.load(PROD_MODEL, map_location="cpu")["members"]:
        m = RecurrentForecaster(meta["arch"], len(meta["features"]), p["hidden"], p["layers"],
                                p.get("dropout", 0.2))
        m.load_state_dict(sd)
        m.eval()
        members.append(m)
    return members, pre, meta


def forecast(ds: Dataset | None = None, bundle=None) -> dict:
    """Predicción del conjunto para los tres horizontes desde el último dato disponible."""
    ds = ds or Dataset.load()
    members, pre, meta = bundle or load_production()
    x = pre.transform(ds.feats)
    X = make_windows(x, np.array([len(x) - 1]))
    pred_lr = pre.unscale_targets(np.mean([predict(m, X) for m in members], axis=0))[0]
    p0 = float(ds.df["brent"].iloc[-1])
    last_date = ds.index[-1]
    out = {"last_date": last_date, "last_price": p0, "horizons": {}}
    for j, (name, h) in enumerate(HORIZONS.items()):
        lo, hi = meta["interval_quantiles"][name]
        out["horizons"][name] = {
            "h": h,
            "target_date": last_date + pd.offsets.BDay(h),
            "price": p0 * float(np.exp(pred_lr[j])),
            "low": p0 * float(np.exp(pred_lr[j] + lo)),
            "high": p0 * float(np.exp(pred_lr[j] + hi)),
            "change_pct": 100 * float(np.exp(pred_lr[j]) - 1),
        }
    return out
