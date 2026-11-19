"""
Descarga o publica el modelo de producción en Hugging Face Hub.

Uso:
    poetry run python scripts/hf_model_registry.py download   # models/ <- Hub (actualización diaria, sin entrenar)
    poetry run python scripts/hf_model_registry.py upload     # models/ -> Hub (tras reentrenar; requiere HF_TOKEN)

Variables de entorno: HF_MODEL_REPO="usuario/nombre" (obligatoria) y HF_TOKEN (para publicar).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.model_registry import download_production, upload_production  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["download", "upload"])
    ap.add_argument("--repo", help="usuario/nombre (por defecto, HF_MODEL_REPO)")
    ap.add_argument("--private", action="store_true", help="crear el repo como privado al publicar")
    args = ap.parse_args()
    if args.action == "download":
        meta = download_production(args.repo)
        print(f"Modelo {meta['arch']} descargado: entrenado el {meta['trained_at']} "
              f"con datos hasta {meta['data_until']}")
    else:
        print("Publicado en", upload_production(args.repo, private=args.private))
