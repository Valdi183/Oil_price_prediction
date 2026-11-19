"""Particiones temporales (walk-forward), entrenamiento del conjunto de semillas y del modelo de producción."""
from __future__ import annotations

import json
import pickle
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd
import torch

from .config import ENSEMBLE_SEEDS, HORIZONS, MODELS_DIR, PROD_META, PROD_MODEL, PROD_PRE, WINDOW
from .evaluation import horizon_metrics
from .feature_engineering import FEATURES, Preprocessor, make_windows, masked_targets, prepare_dataset
from .models import count_params, train_model


# ----------------------------------------------------------------------------
# Datos y particiones
# ----------------------------------------------------------------------------
@dataclass
class Dataset:
    df: pd.DataFrame
    feats: pd.DataFrame
    targets: pd.DataFrame

    @classmethod
    def load(cls):
        return cls(*prepare_dataset())

    @property
    def index(self):
        return self.feats.index

    def pos(self, date) -> int:
        """Posición del primer día >= date."""
        if date is None:
            return len(self.index)
        return int(self.index.searchsorted(pd.Timestamp(date)))


@dataclass
class Split:
    X_tr: np.ndarray
    y_tr: np.ndarray          # escalado, con NaN donde no hay objetivo válido
    X_ev: np.ndarray
    y_ev: np.ndarray          # log-rentabilidades reales (sin escalar)
    y_ev_scaled: np.ndarray
    ev_origins: np.ndarray
    pre: Preprocessor


def make_split(ds: Dataset, train_end, eval_start=None, eval_end=None, eval_cutoff=None) -> Split:
    """Entrenamiento: orígenes < train_end, sin objetivos que miren más allá de train_end.
    Evaluación: orígenes en [eval_start, eval_end), objetivos hasta eval_cutoff."""
    p_tr = ds.pos(train_end)
    tr_orig = np.arange(WINDOW - 1, p_tr)
    y_tr = masked_targets(ds.targets, tr_orig, cutoff_pos=p_tr)
    keep = ~np.isnan(y_tr).all(axis=1)
    tr_orig, y_tr = tr_orig[keep], y_tr[keep]

    pre = Preprocessor().fit(ds.feats.iloc[:p_tr], pd.DataFrame(y_tr))
    x = pre.transform(ds.feats)

    X_tr = make_windows(x, tr_orig)
    y_tr_s = pre.scale_targets(y_tr)

    if eval_start is None:
        empty = np.empty((0, WINDOW, x.shape[1]), np.float32)
        return Split(X_tr, y_tr_s, empty, np.empty((0, 3)), np.empty((0, 3), np.float32),
                     np.empty(0, int), pre)

    ev_orig = np.arange(max(ds.pos(eval_start), WINDOW - 1), ds.pos(eval_end))
    cutoff = None if eval_cutoff is None else ds.pos(eval_cutoff)
    y_ev = masked_targets(ds.targets, ev_orig, cutoff_pos=cutoff)
    return Split(X_tr, y_tr_s, make_windows(x, ev_orig), y_ev, pre.scale_targets(y_ev), ev_orig, pre)


# ----------------------------------------------------------------------------
# Entrenamiento + evaluación: conjunto (ensemble) de semillas
# ----------------------------------------------------------------------------
@dataclass
class EnsembleRun:
    """Conjunto de redes (una por semilla) entrenadas n_epochs épocas sobre un split.
    preds[s, e] = predicción (log-rentabilidad) del miembro s tras la época e+1."""
    preds: np.ndarray            # (n_semillas, n_épocas, n_eval, 3)
    split: Split
    models: list                 # modelos tras la última época
    seconds: float
    n_params: int                # parámetros de UNA red

    @property
    def n_epochs(self) -> int:
        return self.preds.shape[1]

    def at_epoch(self, ds: Dataset, epoch: int):
        """(predicción del conjunto, métricas del conjunto, métricas de cada miembro) en la época dada."""
        member = self.preds[:, epoch - 1]
        ens = member.mean(axis=0)
        price0 = ds.df["brent"].values[self.split.ev_origins]
        return (ens, horizon_metrics(ens, self.split.y_ev, price0),
                [horizon_metrics(m, self.split.y_ev, price0) for m in member])


def train_ensemble(ds: Dataset, arch: str, params: dict, split: Split, seeds: list[int],
                   n_epochs: int, eval_name: str = "validacion", track=None) -> EnsembleRun:
    """Entrena un conjunto de redes idénticas con semillas distintas durante n_epochs épocas.
    El conjunto de evaluación solo se MIDE en cada época (curvas); nunca decide nada aquí.
    track(run_name) -> context manager que devuelve el callback on_epoch (o None)."""
    track = track or (lambda name: nullcontext(None))
    has_eval = len(split.X_ev) > 0
    preds, models, secs = [], [], 0.0
    for seed in seeds:
        mon = {eval_name: (split.X_ev, split.y_ev_scaled)} if has_eval else None
        with track(f"s{seed}") as cb:
            model, s, _, collected = train_model(arch, params, split.X_tr, split.y_tr, n_epochs,
                                                 seed=seed, monitors=mon, on_epoch=cb,
                                                 collect=eval_name if has_eval else None)
        secs += s
        models.append(model)
        if has_eval:
            preds.append([split.pre.unscale_targets(p) for p in collected])
    arr = np.array(preds) if has_eval else np.empty((len(seeds), n_epochs, 0, 3))
    return EnsembleRun(arr, split, models, secs, count_params(models[0]))


# ----------------------------------------------------------------------------
# Modelo de producción (el que usa la aplicación)
# ----------------------------------------------------------------------------
def train_production(arch: str, params: dict, n_epochs: int, quantiles: dict, test_metrics: list[dict],
                     extra_meta: dict | None = None, ds: Dataset | None = None, track=None):
    """Entrena el conjunto de la arquitectura elegida con TODOS los datos disponibles durante
    las épocas elegidas en la validación cruzada y guarda los artefactos en models/."""
    ds = ds or Dataset.load()
    run = train_ensemble(ds, arch, params, make_split(ds, train_end=None), ENSEMBLE_SEEDS,
                         n_epochs, track=track)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"members": [m.state_dict() for m in run.models]}, PROD_MODEL)
    with open(PROD_PRE, "wb") as fh:
        pickle.dump(run.split.pre, fh)
    meta = {
        "arch": arch, "params": params, "n_epochs": n_epochs,
        "n_members": len(run.models), "seeds": ENSEMBLE_SEEDS,
        "features": FEATURES, "window": WINDOW, "horizons": HORIZONS,
        "interval_quantiles": quantiles, "test_metrics": test_metrics,
        "n_params": run.n_params, "train_seconds": round(run.seconds, 1),
        "data_until": str(ds.index[-1].date()),
        "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        **(extra_meta or {}),
    }
    PROD_META.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return meta
