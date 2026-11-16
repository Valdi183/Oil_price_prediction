"""Las tres redes recurrentes (RNN simple, LSTM, GRU) y su bucle de entrenamiento."""
from __future__ import annotations

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


def _naive_mse(y):
    """Error del modelo ingenuo (predecir 0 = precio sin cambios) sobre objetivos escalados."""
    mask = ~torch.isnan(y)
    return float((torch.nan_to_num(y) ** 2).sum() / mask.sum().clamp(min=1))


def _ratio_per_horizon(pred, y):
    """Error / error_ingenuo para cada horizonte (columna)."""
    out = []
    for j in range(y.shape[1]):
        m = ~torch.isnan(y[:, j])
        if m.sum() == 0:
            out.append(float("nan"))
            continue
        out.append(float(((pred[m, j] - y[m, j]) ** 2).mean() / (y[m, j] ** 2).mean()))
    return out


def _masked_huber(pred, y, delta: float):
    """Error de Huber: cuadrático para errores pequeños y lineal para los grandes (> delta),
    de modo que los días extremos (2008, 2020, 2022, 2026) no dominan el aprendizaje."""
    mask = ~torch.isnan(y)
    diff = torch.where(mask, pred - torch.nan_to_num(y), torch.zeros_like(pred))
    a = diff.abs()
    loss = torch.where(a <= delta, 0.5 * diff ** 2, delta * (a - 0.5 * delta))
    return loss.sum() / mask.sum().clamp(min=1)


def train_model(arch: str, params: dict, X_tr, y_tr, n_epochs: int, seed: int = SEED,
                monitors: dict | None = None, on_epoch=None, collect: str | None = None):
    """Entrena una red durante n_epochs épocas (el número de épocas es un hiperparámetro
    elegido con la validación cruzada, no se decide durante el entrenamiento).

    - Se optimiza el error de Huber (robusto a días extremos); las curvas se miden en MSE.
    - monitors: {nombre: (X, y)} conjuntos que solo se MIDEN por época para las curvas.
    - on_epoch(epoch, info): callback por época con las curvas (error / error_ingenuo).
    - collect: nombre de un monitor cuyas predicciones se guardan en cada época
      (para construir la curva del conjunto de semillas).

    Devuelve (modelo, segundos, historial, predicciones_por_época)."""
    cfg = {**TRAIN_DEFAULTS, **params}
    set_seed(seed)
    model = RecurrentForecaster(arch, X_tr.shape[-1], cfg["hidden"], cfg["layers"], cfg["dropout"])
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])

    Xt, yt = torch.from_numpy(X_tr), torch.from_numpy(y_tr)
    naive_tr = _naive_mse(yt)
    evals = {name: (torch.from_numpy(X), torch.from_numpy(y))
             for name, (X, y) in (monitors or {}).items() if len(X)}
    naive_ev = {k: _naive_mse(y) for k, (_, y) in evals.items()}

    history, collected = [], []
    t0 = time.perf_counter()
    for epoch in range(1, n_epochs + 1):
        model.train()
        perm = torch.randperm(len(Xt))
        tot, nb = 0.0, 0
        for i in range(0, len(Xt), cfg["batch_size"]):
            b = perm[i:i + cfg["batch_size"]]
            opt.zero_grad()
            pred = model(Xt[b])
            loss = _masked_huber(pred, yt[b], cfg["huber_delta"])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot, nb = tot + _masked_mse(pred.detach(), yt[b]).item(), nb + 1

        info = {"epoch": epoch, "train": tot / nb / naive_tr}
        model.eval()
        with torch.no_grad():
            for name, (X, y) in evals.items():
                pred = model(X)
                info[name] = _masked_mse(pred, y).item() / naive_ev[name]
                info[f"{name}_h"] = _ratio_per_horizon(pred, y)
                if name == collect:
                    collected.append(pred.numpy())
        history.append(info)
        if on_epoch:
            on_epoch(epoch, info)
    model.eval()
    return model, time.perf_counter() - t0, history, collected


def predict(model: nn.Module, X: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return model(torch.from_numpy(X)).numpy()
