"""
Reentrena el modelo de producción (arquitectura e hiperparámetros ya elegidos en la
comparativa) con los datos más recientes. Pensado para ejecutarse periódicamente.

Uso:
    poetry run python scripts/retrain.py              # reentrena con data/petroleo_dataset.csv
    poetry run python scripts/retrain.py --download   # descarga datos nuevos y reentrena
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from oilcast.pipeline import PROD_META, train_production  # noqa: E402


def download_data():
    subprocess.run([sys.executable, str(ROOT / "scripts" / "download_data.py")], check=True, cwd=ROOT)


def retrain() -> dict:
    if not PROD_META.exists():
        raise FileNotFoundError("No hay modelo elegido todavía: ejecuta antes scripts/train_compare.py")
    meta = json.loads(PROD_META.read_text(encoding="utf-8"))
    # Misma receta que en la comparativa: conjunto de semillas entrenado con todos los datos
    # durante el número de épocas elegido en la validación cruzada.
    keep = {k: meta[k] for k in ("score_cv", "selected_by") if k in meta}
    return train_production(meta["arch"], meta["params"], meta["n_epochs"], meta["interval_quantiles"],
                            meta["test_metrics"], extra_meta=keep)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--download", action="store_true", help="descargar datos recientes antes")
    if ap.parse_args().download:
        download_data()
    m = retrain()
    print(f"Modelo {m['arch']} reentrenado con datos hasta {m['data_until']} ({m['train_seconds']} s)")
