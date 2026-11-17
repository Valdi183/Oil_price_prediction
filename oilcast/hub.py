"""Publicación y descarga del modelo de producción en Hugging Face Hub.

Separa los dos procesos que el día a día no debe mezclar:
  - ACTUALIZAR DATOS (diario): se descarga el modelo ya entrenado y el último dato pasa por él
    (solo inferencia, sin entrenar).
  - REENTRENAR (manual, cuando haga falta): se entrena con todos los datos y se publica la
    nueva versión en el Hub, que pasa a ser la que usan las actualizaciones diarias.

Repositorio del modelo: variable de entorno HF_MODEL_REPO ("usuario/nombre"). Para publicar
hace falta un token con permiso de escritura en HF_TOKEN; para descargar un repo público, no.
"""
from __future__ import annotations

import json
import os
from datetime import datetime

from .config import MODELS_DIR
from .pipeline import PROD_META, PROD_MODEL, PROD_PRE

FILES = [PROD_MODEL.name, PROD_PRE.name, PROD_META.name]


def repo_id(explicit: str | None = None) -> str:
    rid = explicit or os.environ.get("HF_MODEL_REPO", "").strip()
    if not rid or "/" not in rid:
        raise ValueError("Falta el repositorio del modelo: define HF_MODEL_REPO=\"usuario/nombre\" "
                         "(en GitHub: Settings → Secrets and variables → Actions → Variables).")
    return rid


def download_production(repo: str | None = None, token: str | None = None) -> dict:
    """Descarga el modelo de producción del Hub a models/ y devuelve su meta."""
    from huggingface_hub import hf_hub_download

    rid = repo_id(repo)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        hf_hub_download(rid, name, local_dir=MODELS_DIR, token=token or os.environ.get("HF_TOKEN") or None)
    return json.loads(PROD_META.read_text(encoding="utf-8"))


def _model_card(meta: dict, rid: str) -> str:
    rows = "\n".join(
        f"| {m['horizonte']} ({m['h']} d) | {m['MAE']:.2f} | {m['RMSE']:.2f} | {m['RMSE_ingenuo']:.2f} | "
        f"{m['mejora_RMSE_%']:+.1f} % |" for m in meta["test_metrics"])
    p = meta["params"]
    return f"""---
language: es
tags: [time-series, forecasting, rnn, pytorch, oil, brent]
library_name: pytorch
---

# {rid.split('/')[-1]} · predicción del precio del Brent

Conjunto de {meta['n_members']} redes **{meta['arch']}** ({p['hidden']} neuronas × {p['layers']} capas,
{meta['n_epochs']} épocas, error Huber) que predice el Brent a 5, 63 y 252 días hábiles a partir de una
ventana de {meta['window']} días con {len(meta['features'])} variables (FRED + Yahoo Finance).

- Entrenado con datos hasta **{meta['data_until']}** (reentrenamiento: {meta['trained_at']}).
- Elegido frente a LSTM y GRU con validación cruzada walk-forward; intervalos del 80 % por cuantiles empíricos.
- Proyecto: caso de uso III de Aprendizaje automático (UAX). La web y el código del pipeline están en GitHub.

## Evaluación en test (entrenado hasta 2022, evaluado 2023 → 2026)

| Horizonte | MAE ($) | RMSE ($) | RMSE ingenuo ($) | Mejora vs ingenuo |
|---|---:|---:|---:|---:|
{rows}

## Ficheros

- `production_model.pt`: pesos de los {meta['n_members']} miembros (`state_dict`).
- `production_preprocessor.pkl`: escalador ajustado solo con los datos de entrenamiento.
- `production_meta.json`: arquitectura, variables, cuantiles del intervalo y métricas.

*Herramienta de apoyo a la decisión, no es una recomendación de inversión.*
"""


def upload_production(repo: str | None = None, token: str | None = None, private: bool = False) -> str:
    """Publica models/ en el Hub (crea el repo si no existe) con una ficha del modelo."""
    from huggingface_hub import HfApi

    rid = repo_id(repo)
    token = token or os.environ.get("HF_TOKEN")
    if not token:
        raise PermissionError("Falta HF_TOKEN (token de Hugging Face con permiso de escritura).")
    missing = [n for n in FILES if not (MODELS_DIR / n).exists()]
    if missing:
        raise FileNotFoundError(f"No existen en models/: {missing}. Ejecuta antes scripts/retrain.py")

    meta = json.loads(PROD_META.read_text(encoding="utf-8"))
    (MODELS_DIR / "README.md").write_text(_model_card(meta, rid), encoding="utf-8")
    api = HfApi(token=token)
    api.create_repo(rid, repo_type="model", private=private, exist_ok=True)
    api.upload_folder(
        repo_id=rid, folder_path=str(MODELS_DIR), allow_patterns=FILES + ["README.md"],
        commit_message=f"Modelo {meta['arch']} reentrenado con datos hasta {meta['data_until']} "
                       f"({datetime.now():%Y-%m-%d %H:%M})")
    return f"https://huggingface.co/{rid}"
