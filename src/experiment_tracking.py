"""Seguimiento en vivo del entrenamiento: consola + TensorBoard + historial CSV.

Todas las curvas se expresan como  error / error_del_modelo_ingenuo :
  < 1  -> la red bate a "el precio futuro = el precio de hoy"
  = 1  -> igual que el ingenuo
  > 1  -> peor que el ingenuo
La separación entre la curva de entrenamiento (que sigue bajando) y la de validación
(que deja de bajar y sube) es el sobreajuste.

Ver en vivo:  poetry run tensorboard --logdir runs
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from pathlib import Path

import pandas as pd

from .config import HORIZONS

_HNAMES = list(HORIZONS)
_HSHORT = {"corto": "5d", "medio": "63d", "largo": "252d"}
_SKIP = {"epoch", "train"}


def cfg_label(params: dict) -> str:
    return f"h{params['hidden']}·l{params['layers']}·lr{params.get('lr', 1e-3):g}"


class Tracker:
    def __init__(self, runs_dir: Path, verbose: bool = True, tensorboard: bool = True):
        self.root = Path(runs_dir) / time.strftime("%Y%m%d-%H%M%S")
        self.verbose = verbose
        self.tensorboard = tensorboard
        self.rows: list[dict] = []

    @contextmanager
    def run(self, phase: str, arch: str, params: dict, run_name: str):
        """Contexto para un entrenamiento. Devuelve el callback on_epoch."""
        label = cfg_label(params)
        writer = None
        if self.tensorboard:
            from torch.utils.tensorboard import SummaryWriter
            writer = SummaryWriter(str(self.root / phase / arch / label.replace("·", "_") / run_name))
        head = f"  {phase:<6s} {arch:<4s} {label:<17s} {run_name:<7s}"

        def on_epoch(epoch: int, info: dict):
            row = {"fase": phase, "red": arch, "config": label, "run": run_name,
                   "epoca": epoch, "entrenamiento": info["train"]}
            evals = [k for k in info if k not in _SKIP and not k.endswith("_h")]
            for k in evals:
                row[k] = info[k]
                for name, v in zip(_HNAMES, info[f"{k}_h"]):
                    row[f"{k}_{name}"] = v
            self.rows.append(row)

            if writer is not None:
                writer.add_scalar("1_curvas/entrenamiento", info["train"], epoch)
                for k in evals:
                    writer.add_scalar(f"1_curvas/{k}", info[k], epoch)
                    for name, v in zip(_HNAMES, info[f"{k}_h"]):
                        writer.add_scalar(f"2_{k}_por_horizonte/{name}", v, epoch)
                writer.flush()

            if self.verbose:
                parts = [f"entren {info['train']:.3f}"]
                for k in evals:
                    hs = " · ".join(f"{_HSHORT[n]} {v:.2f}" for n, v in zip(_HNAMES, info[f"{k}_h"]))
                    parts.append(f"{k} {info[k]:.3f} [{hs}]")
                print(f"{head} ép {epoch:2d} │ " + " │ ".join(parts), flush=True)

        try:
            yield on_epoch
        finally:
            if writer is not None:
                writer.close()

    def history(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)

    def save(self, path: Path):
        if self.rows:
            self.history().to_csv(path, index=False)
