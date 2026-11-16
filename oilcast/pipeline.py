"""Particiones temporales, métricas, validación walk-forward y modelo de producción."""
from __future__ import annotations

import json
import pickle
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd
import torch

from .config import (H_VALUES, HORIZONS, INTERVAL_HIGH_Q, INTERVAL_LOW_Q, MODELS_DIR, WINDOW)
from .data import FEATURES, Preprocessor, make_windows, masked_targets, prepare_dataset
from .models import RecurrentForecaster, count_params, predict, train_model


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
# Métricas (en dólares por barril)
# ----------------------------------------------------------------------------
def horizon_metrics(pred_lr: np.ndarray, true_lr: np.ndarray, price0: np.ndarray) -> list[dict]:
    """Métricas por horizonte. pred_lr/true_lr: log-rentabilidades (n, 3); price0: precio en t."""
    rows = []
    for j, (name, h) in enumerate(HORIZONS.items()):
        m = ~np.isnan(true_lr[:, j])
        if m.sum() == 0:
            continue
        p0 = price0[m]
        real = p0 * np.exp(true_lr[m, j])
        pred = p0 * np.exp(pred_lr[m, j])
        err, err_naive = pred - real, p0 - real
        mse, mse_n = np.mean(err ** 2), np.mean(err_naive ** 2)
        mae, mae_n = np.mean(np.abs(err)), np.mean(np.abs(err_naive))
        moved = true_lr[m, j] != 0
        dir_acc = np.mean(np.sign(pred_lr[m, j][moved]) == np.sign(true_lr[m, j][moved]))
        rows.append({
            "horizonte": name, "h": h, "n": int(m.sum()),
            "MSE": mse, "MAE": mae, "RMSE": np.sqrt(mse),
            "MAE_ingenuo": mae_n, "RMSE_ingenuo": np.sqrt(mse_n),
            "mejora_RMSE_%": 100 * (1 - np.sqrt(mse) / np.sqrt(mse_n)),
            "mejora_MAE_%": 100 * (1 - mae / mae_n),
            "acierto_direccion_%": 100 * dir_acc,
        })
    return rows


def residual_quantiles(residuals: np.ndarray) -> dict:
    """Cuantiles empíricos de (real - predicho) en log-rentabilidad -> intervalo del 80 %."""
    out = {}
    for j, name in enumerate(HORIZONS):
        r = residuals[:, j][~np.isnan(residuals[:, j])]
        out[name] = (float(np.quantile(r, INTERVAL_LOW_Q)), float(np.quantile(r, INTERVAL_HIGH_Q)))
    return out


def interval_coverage(pred_lr, true_lr, quantiles) -> dict:
    cov = {}
    for j, name in enumerate(HORIZONS):
        m = ~np.isnan(true_lr[:, j])
        lo, hi = quantiles[name]
        inside = (true_lr[m, j] >= pred_lr[m, j] + lo) & (true_lr[m, j] <= pred_lr[m, j] + hi)
        cov[name] = 100 * float(inside.mean())
    return cov


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
PROD_MODEL = MODELS_DIR / "production_model.pt"
PROD_PRE = MODELS_DIR / "production_preprocessor.pkl"
PROD_META = MODELS_DIR / "production_meta.json"


def train_production(arch: str, params: dict, n_epochs: int, quantiles: dict, test_metrics: list[dict],
                     extra_meta: dict | None = None, ds: Dataset | None = None, track=None):
    """Entrena el conjunto de la arquitectura elegida con TODOS los datos disponibles durante
    las épocas elegidas en la validación cruzada y guarda los artefactos."""
    from .config import ENSEMBLE_SEEDS
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


def load_production():
    meta = json.loads(PROD_META.read_text(encoding="utf-8"))
    with open(PROD_PRE, "rb") as fh:
        pre = pickle.load(fh)
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
