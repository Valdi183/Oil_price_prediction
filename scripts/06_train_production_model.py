"""
Reentrena el modelo de producción (arquitectura e hiperparámetros ya elegidos en la
comparativa) con los datos más recientes. Pensado para ejecutarse periódicamente
(en local o en GitHub Actions).

Uso:
    poetry run python scripts/06_train_production_model.py              # reentrena con data/raw/brent_dataset_raw.csv
    poetry run python scripts/06_train_production_model.py --download   # descarga datos nuevos y reentrena
    poetry run python scripts/hf_model_registry.py upload     # publica el nuevo modelo en Hugging Face

Reentrenar es un proceso distinto de incorporar datos: la actualización diaria solo pasa el
último dato por el modelo ya entrenado (scripts/hf_model_registry.py download + 07_export_app_data.py).

La receta (red, hiperparámetros, épocas, cuantiles del intervalo y métricas de test) se lee
de models/production_meta.json si existe y, si no (p. ej. en un runner limpio de GitHub
Actions, porque models/ no se versiona), de config/production_model_config.json.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import PROD_META, PRODUCTION_CONFIG_PATH as RECIPE  # noqa: E402
from src.training import train_production  # noqa: E402

RECIPE_KEYS = ("arch", "params", "n_epochs", "interval_quantiles", "test_metrics")


def download_data():
    subprocess.run([sys.executable, str(ROOT / "scripts" / "01_download_data.py")], check=True, cwd=ROOT)


def load_recipe() -> dict:
    """Receta del modelo elegido: el meta del último modelo entrenado o, en su defecto, la
    receta versionada que escribió la comparativa."""
    if PROD_META.exists():
        return json.loads(PROD_META.read_text(encoding="utf-8"))
    # En el día a día no se reentrena: la Action descarga el modelo publicado en Hugging Face
    # (scripts/hf_model_registry.py download). Este script solo se usa al reentrenar a mano.
    if RECIPE.exists():
        return json.loads(RECIPE.read_text(encoding="utf-8"))
    raise FileNotFoundError(
        "No hay modelo elegido todavía: ejecuta antes scripts/03_train_compare_models.py "
        f"(o añade la receta en {RECIPE.relative_to(ROOT)})")


def retrain() -> dict:
    recipe = load_recipe()
    missing = [k for k in RECIPE_KEYS if k not in recipe]
    if missing:
        raise KeyError(f"La receta no tiene las claves {missing}")
    # Misma receta que en la comparativa: conjunto de semillas entrenado con todos los datos
    # durante el número de épocas elegido en la validación cruzada.
    keep = {k: recipe[k] for k in ("score_cv", "selected_by") if k in recipe}
    return train_production(recipe["arch"], recipe["params"], recipe["n_epochs"],
                            recipe["interval_quantiles"], recipe["test_metrics"], extra_meta=keep)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--download", action="store_true", help="descargar datos recientes antes")
    if ap.parse_args().download:
        download_data()
    m = retrain()
    print(f"Modelo {m['arch']} reentrenado con datos hasta {m['data_until']} ({m['train_seconds']} s)")
