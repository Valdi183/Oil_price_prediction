"""
Paso 2 del pipeline: limpieza de los datos y construcción de las variables del modelo.

Lee los datos en bruto (data/raw/brent_dataset_raw.csv) y guarda:
  - data/processed/brent_dataset_clean.csv  datos limpios (huecos rellenados, outliers tratados)
  - data/processed/brent_features.csv       las 16 variables de entrada de la red y los 3 objetivos
                                            (rentabilidad logarítmica a 5, 63 y 252 días hábiles)
  - reports/data_cleaning.md                resumen de la limpieza (faltantes, outliers, rango)

Es la misma limpieza que aplica el pipeline en memoria (src/data_cleaning.py y
src/feature_engineering.py): estos ficheros sirven para revisarla. El escalado y la
winsorización NO se aplican aquí porque se ajustan solo con los datos de entrenamiento de
cada partición (src/feature_engineering.py::Preprocessor).

Uso:  poetry run python scripts/02_clean_data.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from src import config as C  # noqa: E402
from src.data_cleaning import clean, load_raw  # noqa: E402
from src.feature_engineering import FEATURES, prepare_dataset  # noqa: E402


def main():
    raw = load_raw()
    clean_df = clean(raw)
    df, feats, targets = prepare_dataset()

    C.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    clean_df.to_csv(C.CLEAN_DATA_PATH, float_format="%.6g")
    targets = targets.rename(columns={f"h{h}": f"objetivo_{name}_{h}d" for name, h in C.HORIZONS.items()})
    out = pd.concat([df[["brent"]], feats, targets], axis=1)
    out.index.name = "Date"
    out.to_csv(C.FEATURES_PATH, float_format="%.6g")

    write_report(raw, clean_df, df, out)
    print(f"Datos limpios: {C.CLEAN_DATA_PATH.relative_to(C.ROOT)} {clean_df.shape}")
    print(f"Variables:     {C.FEATURES_PATH.relative_to(C.ROOT)} {out.shape}")
    print(f"Informe:       reports/data_cleaning.md")


def write_report(raw, clean_df, df, out):
    dropped_tail = raw.index.difference(clean_df.index)
    wti = raw.get("wti", pd.Series(dtype=float))
    neg_wti = int((wti <= 0).sum())
    wti_neg_txt = ", ".join(f"{v:.2f} $".replace(".", ",").replace("-", "−") for v in wti[wti <= 0]) or "—"
    missing = raw.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    first_obs = raw.apply(lambda s: s.first_valid_index())

    L = ["# Limpieza de los datos", "",
         f"Generado por `scripts/02_clean_data.py`. Datos en bruto: `data/raw/brent_dataset_raw.csv` "
         f"({len(raw)} días hábiles, {raw.index[0]:%d/%m/%Y} → {raw.index[-1]:%d/%m/%Y}, {raw.shape[1]} columnas).", "",
         "## Pasos", "",
         "1. **Calendario del Brent**: todas las series se alinean con los días hábiles en que cotiza el Brent "
         "(se hace ya en la descarga).",
         f"2. **Retraso de la EIA**: se eliminan las últimas filas en las que el Brent spot está repetido pero "
         f"el futuro sí se ha movido (el dato aún no está publicado): **{len(dropped_tail)} fila(s)** eliminada(s).",
         f"3. **Outlier del WTI**: el precio negativo del 20/04/2020 ({wti_neg_txt} en la serie spot de la EIA; "
         f"−37,63 $ el futuro) es una anomalía técnica del vencimiento del contrato, no un precio económico; "
         f"se trata como dato faltante: **{neg_wti} valor(es)**.",
         "4. **Festivos distintos en cada mercado**: los huecos se rellenan con el último valor disponible "
         "(y, al inicio de las series que empiezan más tarde, con el primero).",
         "5. **Variables del modelo** (`data/processed/brent_features.csv`): rentabilidades logarítmicas, momentum, "
         "volatilidad, distancia a la media de 200 días, diferencial Brent-WTI, VIX en logaritmos y tipo del BCE. "
         f"Se descartan los primeros días sin historia suficiente: la serie útil empieza el {df.index[0]:%d/%m/%Y}.",
         "6. **Outliers de las rentabilidades y escalado**: winsorización al 0,5 % / 99,5 % y estandarización, "
         "ajustadas **solo con los datos de entrenamiento** de cada partición (por eso no se aplican a este fichero).",
         "", "## Valores faltantes en los datos en bruto", "",
         "| Columna | Faltantes | % | Primer dato |", "|---|---:|---:|---|"]
    for col, n in missing.items():
        L.append(f"| `{col}` | {n} | {100 * n / len(raw):.1f} % | {first_obs[col]:%d/%m/%Y} |")
    L += ["", f"Tras la limpieza: **{int(clean_df.isna().sum().sum())} valores faltantes**.", "",
          "## Variables del modelo", "",
          f"{len(FEATURES)} variables de entrada + 3 objetivos, {len(out)} días ({out.index[0]:%d/%m/%Y} → "
          f"{out.index[-1]:%d/%m/%Y}). Los objetivos de los últimos días quedan vacíos porque su valor real "
          "aún no se conoce.", "",
          "| Variable | Media | Desv. típica | Mín. | Máx. |", "|---|---:|---:|---:|---:|"]
    desc = out.describe().T
    for col, r in desc.iterrows():
        L.append(f"| `{col}` | {r['mean']:.4g} | {r['std']:.4g} | {r['min']:.4g} | {r['max']:.4g} |")
    (C.REPORTS_DIR / "data_cleaning.md").write_text("\n".join(L) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
