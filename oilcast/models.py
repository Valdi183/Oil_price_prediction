"""Las tres redes recurrentes (RNN simple, LSTM, GRU) y su bucle de entrenamiento."""
from __future__ import annotations

import copy
import time

import numpy as np
import torch
from torch import nn

from .config import H_VALUES, SEED, TRAIN_DEFAULTS

_CELLS = {"RNN": nn.RNN, "LSTM": nn.LSTM, "GRU": nn.GRU}


class RecurrentForecaster(nn.Module):
    """Capa(s) recurrente(s) + cabeza lineal con una salida por horizonte (5, 63, 252 días).
    Las tres arquitecturas comparten exactamente la misma estructura; solo cambia la celda."""

    def __init__(self, arch: str, n_features: int, hidden: int, layers: int,
                 dropout: float = 0.2, n_out: int = len(H_VALUES)):
        super().__init__()
        self.rnn = _CELLS[arch](n_features, hidden, num_layers=layers, batch_first=True,
                                dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, n_out))

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.head(out[:, -1])


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def set_seed(seed: int = SEED):
    torch.manual_seed(seed)
    np.random.seed(seed)


def _masked_mse(pred, y):
    mask = ~torch.isnan(y)
    diff = torch.where(mask, pred - torch.nan_to_num(y), torch.zeros_like(pred))
    return (diff ** 2).sum() / mask.sum().clamp(min=1)


def train_model(arch: str, params: dict, X_tr, y_tr, X_val=None, y_val=None,
                n_epochs: int | None = None, verbose: bool = False, seed: int = SEED):
    """Entrena una red. Con datos de validación aplica early stopping;
    sin ellos entrena un número fijo de épocas (n_epochs).
    Devuelve (modelo, mejor_época, segundos)."""
    cfg = {**TRAIN_DEFAULTS, **params}
    set_seed(seed)
    model = RecurrentForecaster(arch, X_tr.shape[-1], cfg["hidden"], cfg["layers"], cfg["dropout"])
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])

    Xt, yt = torch.from_numpy(X_tr), torch.from_numpy(y_tr)
    has_val = X_val is not None
    if has_val:
        Xv, yv = torch.from_numpy(X_val), torch.from_numpy(y_val)
    epochs = n_epochs or cfg["max_epochs"]

    best_loss, best_state, best_epoch, bad = np.inf, None, epochs, 0
    t0 = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), cfg["batch_size"]):
            b = perm[i:i + cfg["batch_size"]]
            opt.zero_grad()
            loss = _masked_mse(model(Xt[b]), yt[b])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        if has_val:
            model.eval()
            with torch.no_grad():
                vloss = _masked_mse(model(Xv), yv).item()
            if verbose:
                print(f"    época {epoch:2d}  val_loss={vloss:.4f}")
            if vloss < best_loss - 1e-4:
                best_loss, best_state, best_epoch, bad = vloss, copy.deepcopy(model.state_dict()), epoch, 0
            else:
                bad += 1
                if bad >= cfg["patience"]:
                    break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, best_epoch, time.perf_counter() - t0


def predict(model: nn.Module, X: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return model(torch.from_numpy(X)).numpy()
