# Limpieza de los datos

Generado por `scripts/02_clean_data.py`. Datos en bruto: `data/raw/brent_dataset_raw.csv` (7009 días hábiles, 04/01/2000 → 13/11/2026, 20 columnas).

## Pasos

1. **Calendario del Brent**: todas las series se alinean con los días hábiles en que cotiza el Brent (se hace ya en la descarga).
2. **Retraso de la EIA**: se eliminan las últimas filas en las que el Brent spot está repetido pero el futuro sí se ha movido (el dato aún no está publicado): **2 fila(s)** eliminada(s).
3. **Outlier del WTI**: el precio negativo del 20/04/2020 (−36,98 $ en la serie spot de la EIA; −37,63 $ el futuro) es una anomalía técnica del vencimiento del contrato, no un precio económico; se trata como dato faltante: **1 valor(es)**.
4. **Festivos distintos en cada mercado**: los huecos se rellenan con el último valor disponible (y, al inicio de las series que empiezan más tarde, con el primero).
5. **Variables del modelo** (`data/processed/brent_features.csv`): rentabilidades logarítmicas, momentum, volatilidad, distancia a la media de 200 días, diferencial Brent-WTI, VIX en logaritmos y tipo del BCE. Se descartan los primeros días sin historia suficiente: la serie útil empieza el 28/03/2000.
6. **Outliers de las rentabilidades y escalado**: winsorización al 0,5 % / 99,5 % y estandarización, ajustadas **solo con los datos de entrenamiento** de cada partición (por eso no se aplican a este fichero).

## Valores faltantes en los datos en bruto

| Columna | Faltantes | % | Primer dato |
|---|---:|---:|---|
| `brent_fut_Open` | 1974 | 28.2 % | 30/07/2007 |
| `brent_fut_Low` | 1974 | 28.2 % | 30/07/2007 |
| `brent_fut_High` | 1974 | 28.2 % | 30/07/2007 |
| `brent_fut_Close` | 1974 | 28.2 % | 30/07/2007 |
| `brent_fut_Volume` | 1974 | 28.2 % | 30/07/2007 |
| `oro` | 171 | 2.4 % | 30/08/2000 |
| `cobre` | 171 | 2.4 % | 30/08/2000 |
| `wti_fut_Low` | 166 | 2.4 % | 23/08/2000 |
| `wti_fut_Open` | 166 | 2.4 % | 23/08/2000 |
| `wti_fut_Volume` | 166 | 2.4 % | 23/08/2000 |
| `wti_fut_High` | 166 | 2.4 % | 23/08/2000 |
| `wti_fut_Close` | 166 | 2.4 % | 23/08/2000 |
| `gas_natural` | 4 | 0.1 % | 10/01/2000 |

Tras la limpieza: **0 valores faltantes**.

## Variables del modelo

16 variables de entrada + 3 objetivos, 6947 días (28/03/2000 → 11/11/2026). Los objetivos de los últimos días quedan vacíos porque su valor real aún no se conoce.

| Variable | Media | Desv. típica | Mín. | Máx. |
|---|---:|---:|---:|---:|
| `brent` | 68.1 | 28.46 | 9.12 | 143.9 |
| `log_brent` | 4.119 | 0.4775 | 2.21 | 4.969 |
| `ret_brent` | 0.0001869 | 0.02652 | -0.6437 | 0.412 |
| `mom_brent_20` | 0.003945 | 0.1211 | -1.258 | 1.288 |
| `mom_brent_60` | 0.01223 | 0.212 | -1.873 | 1.55 |
| `vol_brent_20` | 0.02192 | 0.01475 | 0.005142 | 0.2191 |
| `gap_ma200` | 0.008665 | 0.2037 | -1.803 | 0.6468 |
| `ret_wti` | 0.0001696 | 0.02846 | -0.7203 | 0.4258 |
| `spread_brent_wti` | 0.02873 | 0.07676 | -0.3811 | 0.3086 |
| `ret_gas` | 8.397e-06 | 0.06488 | -1.402 | 1.433 |
| `ret_usd_eur` | 2.092e-05 | 0.00561 | -0.03003 | 0.04621 |
| `ret_cny_usd` | -3.098e-05 | 0.00174 | -0.02019 | 0.01816 |
| `ret_dxy` | -3.51e-06 | 0.004745 | -0.02717 | 0.02524 |
| `ret_oro` | 0.000386 | 0.01102 | -0.1141 | 0.08643 |
| `ret_cobre` | 0.0002819 | 0.01673 | -0.2517 | 0.1244 |
| `log_vix` | 2.914 | 0.353 | 2.213 | 4.415 |
| `tipo_bce` | 1.073 | 1.44 | -0.5 | 4 |
| `objetivo_corto_5d` | 0.0009613 | 0.05817 | -0.8687 | 0.5368 |
| `objetivo_medio_63d` | 0.01319 | 0.2173 | -1.905 | 1.563 |
| `objetivo_largo_252d` | 0.04388 | 0.3627 | -2.066 | 1.918 |
