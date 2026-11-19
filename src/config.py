"""Configuración central del proyecto (rutas, horizontes, particiones e hiperparámetros)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Datos: raw = tal cual se descargan; processed = limpios y con las variables del modelo
DATA_DIR = ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "brent_dataset_raw.csv"
PROCESSED_DIR = DATA_DIR / "processed"
CLEAN_DATA_PATH = PROCESSED_DIR / "brent_dataset_clean.csv"
FEATURES_PATH = PROCESSED_DIR / "brent_features.csv"

# Resultados: informes (.md), tablas (.csv) y figuras (.png)
REPORTS_DIR = ROOT / "reports"
TABLES_DIR = REPORTS_DIR / "tables"
FIGURES_DIR = REPORTS_DIR / "figures"

# Aplicación web (GitHub Pages): los JSON que muestra se generan en app/data/
APP_DIR = ROOT / "app"
APP_DATA_DIR = APP_DIR / "data"

# Modelo de producción (no se versiona: se publica en Hugging Face Hub)
MODELS_DIR = ROOT / "models"
PROD_MODEL = MODELS_DIR / "production_model.pt"
PROD_PRE = MODELS_DIR / "production_preprocessor.pkl"
PROD_META = MODELS_DIR / "production_meta.json"
PRODUCTION_CONFIG_PATH = ROOT / "config" / "production_model_config.json"

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
# Cada "modelo" es un conjunto de 5 redes idénticas entrenadas con semillas distintas; su
# predicción es la media. Reduce la dependencia de la suerte (inicialización, orden de lotes)
# y la dispersión entre miembros mide la estabilidad de cada arquitectura.
ENSEMBLE_SEEDS = [42, 7, 2024, 11, 99]

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
# lr = 3e-4 se probó (ver historial del proyecto) y no mejoró a ninguna red: se descartó.
PARAM_GRID = [
    {"hidden": h, "layers": l, "lr": lr}
    for lr in (1e-3,)
    for l in (1, 2)
    for h in (32, 64)
]
TRAIN_DEFAULTS = {
    "dropout": 0.2,
    "lr": 1e-3,
    "weight_decay": 1e-5,
    "batch_size": 128,
    "huber_delta": 1.0,   # error de Huber: lineal a partir de 1 desviación típica del objetivo
}
# El número de épocas es un hiperparámetro más: en la CV cada red entrena CV_EPOCHS épocas
# sin parar, se registra la curva de validación del conjunto en cada época y se elige la época
# que minimiza la curva MEDIA de los 3 folds (varios regímenes de mercado, no un único bloque).
CV_EPOCHS = 15

RUNS_DIR = ROOT / "runs"     # registros de TensorBoard (curvas en vivo)

# Intervalo de incertidumbre (80 %)
INTERVAL_LOW_Q = 0.10
INTERVAL_HIGH_Q = 0.90
