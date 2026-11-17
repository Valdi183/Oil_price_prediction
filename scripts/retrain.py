"""
Reentrena el modelo de producción (arquitectura e hiperparámetros ya elegidos en la
comparativa) con los datos más recientes. Pensado para ejecutarse periódicamente
(en local o en GitHub Actions).

Uso:
    poetry run python scripts/retrain.py              # reentrena con data/petroleo_dataset.csv
    poetry run python scripts/retrain.py --download   # descarga datos nuevos y reentrena

La receta (red, hiperparámetros, épocas, cuantiles del intervalo y métricas de test) se lee
de models/production_meta.json si existe y, si no (p. ej. en un runner limpio de GitHub
Actions, porque models/ no se versiona), de config/production_recipe.json.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from oilcast.pipeline import PROD_META, train_production  # noqa: E402

RECIPE = ROOT / "config" / "production_recipe.json"
RECIPE_KEYS = ("arch", "params", "n_epochs", "interval_quantiles", "test_metrics")


def download_data():
    subprocess.run([sys.executable, str(ROOT / "scripts" / "download_data.py")], check=True, cwd=ROOT)


def load_recipe() -> dict:
    """Receta del modelo elegido: el meta del último modelo entrenado o, en su defecto, la
    receta versionada que escribió la comparativa."""
    if PROD_META.exists():
        return json.loads(PROD_META.read_text(encoding="utf-8"))
    # TODO Hugging Face: cuando los modelos estén publicados en el Hub, descargar aquí
    # production_model.pt, production_preprocessor.pkl y production_meta.json en lugar
    # de reentrenar desde cero en cada ejecución.
    if RECIPE.exists():
        return json.loads(RECIPE.read_text(encoding="utf-8"))
    raise FileNotFoundError(
        "No hay modelo elegido todavía: ejecuta antes scripts/train_compare.py "
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
