"""Configuración central del proyecto (rutas, horizontes, particiones e hiperparámetros)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "petroleo_dataset.csv"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Horizontes de predicción en días hábiles
HORIZONS = {"corto": 5, "medio": 63, "largo": 252}
HORIZON_LABELS = {
    "corto": "Corto plazo (5 días hábiles, ~1 semana)",
    "medio": "Medio plazo (63 días hábiles, ~3 meses)",
    "largo": "Largo plazo (252 días hábiles, ~12 meses)",
}
H_VALUES = list(HORIZONS.values())

WINDOW = 60          # días de historia que ve la red en cada predicción
SEED = 42
STABILITY_SEEDS = [42, 7, 2024]   # semillas extra en test para medir la estabilidad

# Partición temporal (sin mezclar fechas).
# Validación cruzada walk-forward con ventana de entrenamiento creciente:
CV_FOLDS = [
    ("2017-01-01", "2019-01-01"),   # fold 1: valida 2017-2018
    ("2019-01-01", "2021-01-01"),   # fold 2: valida 2019-2020 (incluye COVID)
    ("2021-01-01", "2023-01-01"),   # fold 3: valida 2021-2022 (incluye Ucrania)
]
TEST_START = "2023-01-01"           # test final: 2023 -> hoy (incluye picos 2026)

# Presupuesto de ajuste de hiperparámetros IDÉNTICO para las tres redes
ARCHITECTURES = ["RNN", "LSTM", "GRU"]
PARAM_GRID = [
    {"hidden": 32, "layers": 1},
    {"hidden": 64, "layers": 1},
    {"hidden": 32, "layers": 2},
    {"hidden": 64, "layers": 2},
]
TRAIN_DEFAULTS = {
    "dropout": 0.2,
    "lr": 1e-3,
    "weight_decay": 1e-5,
    "batch_size": 128,
    "max_epochs": 40,
    "patience": 6,
}

# Intervalo de incertidumbre (80 %)
INTERVAL_LOW_Q = 0.10
INTERVAL_HIGH_Q = 0.90
