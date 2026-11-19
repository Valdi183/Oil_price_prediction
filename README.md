# Brent Forecast · predicción del precio del petróleo con redes recurrentes

Caso de uso III de la asignatura *Aprendizaje automático* (UAX): comparar una **RNN simple, una LSTM y una GRU** para predecir el precio del **Brent** a corto (5 días hábiles), medio (63) y largo plazo (252), e integrar el mejor modelo en una aplicación que calcule las predicciones, cuantifique su incertidumbre y las explique en lenguaje sencillo.

**Web de la aplicación:** <https://valdi183.github.io/Oil_price_prediction/>

**Modelo en producción (Hugging Face Hub):** <https://huggingface.co/Valdi183/brent-rnn-forecast>

**Presentación (portada y 5 diapositivas):** [docs/presentacion_brent_forecast.pdf](docs/presentacion_brent_forecast.pdf)

> Herramienta de apoyo a la decisión, no es una recomendación de inversión.

## Qué hace la aplicación

- Muestra el último precio del Brent y, para cada horizonte, el **precio previsto, la variación esperada y el rango de incertidumbre del 80 %**, dibujado como un abanico sobre el histórico.
- Genera una **explicación en lenguaje sencillo**: qué significa la predicción, qué fiabilidad tiene en ese horizonte según la comparativa (sello de Diebold-Mariano incluido) y qué factores pueden invalidarla.
- Marca en el gráfico los **picos y caídas del ciclo económico** (2008, 2014-2016, 2020, 2022, 2025-2026…) con una línea de tiempo navegable que hace zoom al periodo.
- **Simulador para inversores** (sin entrenar nada nuevo, a partir de las tres predicciones de la red y de los 2.447 errores reales que cometió fuera de muestra):
  - **predicción a cualquier plazo** de 1 a 252 días hábiles, interpolada entre los tres horizontes con sus rangos del 80 % y del 90 % (en 5, 63 y 252 días coincide exactamente con las tarjetas);
  - **probabilidad de precio**: probabilidad de que el Brent esté por encima o por debajo de un precio, o en qué percentil queda tu propia previsión;
  - **calculadora de posición**: compra, venta o cobertura de una compra futura de crudo (en barriles o contratos de 1.000 barriles), con resultado central, rango del 80 %, probabilidad de ganar y pérdida en el escenario malo (VaR 95 %);
  - **escenarios históricos**: qué pasaría con el precio y con tu posición si se repitiera, en ese plazo, el movimiento de 2008, 2014-2016, 2020, 2022 o 2026, junto a la variación habitual del Brent desde 2000;
  - **backtest de señales** 2023 → hoy (periodo que la red nunca vio): operar según la dirección prevista frente a solo comprar, seguir la tendencia o comprar y mantener, con costes por operación, capital, máxima caída, resultados por año y la sensibilidad al día de inicio.
- Presenta la **comparativa de las tres redes** frente al modelo ingenuo (tablas, barras, real vs predicho en test y test de significación) y justifica la elección.
- Se **actualiza sola**: cada día una GitHub Action descarga los datos recientes y los pasa por el modelo ya entrenado (guardado en Hugging Face Hub), **sin reentrenar**, y vuelve a publicar la web. Reentrenar es un proceso aparte.

## Estructura del repositorio

Organización estándar de un proyecto de datos: datos en bruto y procesados por separado, código reutilizable en `src/`, pasos del pipeline numerados en `scripts/` y resultados en `reports/`.

```
Oil_price_prediction/
├─ data/
│  ├─ raw/brent_dataset_raw.csv           datos tal cual se descargan (FRED + Yahoo Finance)
│  └─ processed/
│     ├─ brent_dataset_clean.csv          datos limpios: huecos rellenados, outliers tratados
│     └─ brent_features.csv               las 16 variables de entrada de la red + los 3 objetivos
├─ notebooks/
│  └─ 01_exploratory_analysis.ipynb       análisis exploratorio (series, outliers, volatilidad, correlaciones)
├─ src/                                   código fuente (lo importan los scripts)
├─ scripts/                               pasos del pipeline, en orden (01 → 07)
├─ reports/
│  ├─ *.md                                informes: limpieza, comparativa, significación, análisis de errores
│  ├─ tables/*.csv                        resultados en tabla (validación cruzada, comparativa, test…)
│  └─ figures/*.png                       figuras
├─ config/production_model_config.json    receta del modelo de producción (red, hiperparámetros, épocas)
├─ models/                                modelo entrenado (no se versiona: está en Hugging Face Hub)
├─ app/                                   aplicación web (index.html, css/, js/, assets/, data/*.json)
└─ .github/workflows/daily_forecast.yml   actualización diaria y despliegue en GitHub Pages
```

### Pipeline (`scripts/`)

| Paso | Script | Qué hace | Salida |
|---|---|---|---|
| 1 | `01_download_data.py` | Descarga los datos diarios desde 2000 y los alinea con el calendario del Brent | `data/raw/` |
| 2 | `02_clean_data.py` | Limpieza (festivos, WTI negativo, retraso de la EIA) y construcción de las 16 variables | `data/processed/`, `reports/data_cleaning.md` |
| 3 | `03_train_compare_models.py` | Comparativa RNN vs LSTM vs GRU: validación cruzada walk-forward, test y modelo de producción (60-70 min en CPU) | `reports/comparison.md`, `reports/tables/`, `reports/figures/` |
| 4 | `04_significance_test.py` | Test de Diebold-Mariano, combinación de referencia y errores fuera de muestra de la red elegida (reentrena las redes en test y la elegida en los folds de CV, ~1 min) | `reports/significance.md`, `reports/tables/test_predictions.csv`, `reports/tables/forecast_errors.csv` |
| 5 | `05_error_analysis.py` | MAE, MSE, RMSE, R² y cuándo acierta y cuándo falla el modelo (sin entrenar) | `reports/error_analysis.md`, `reports/figures/error_analysis.png` |
| 6 | `06_train_production_model.py` | Reentrena el modelo elegido con todos los datos (~13 s) | `models/` |
| 7 | `07_export_app_data.py` | Predicción con el modelo de `models/` (solo inferencia), simulador, escenarios, backtest y JSON de la web | `app/data/*.json` |
| — | `hf_model_registry.py` | Descarga (`download`) o publica (`upload`) el modelo en Hugging Face Hub | `models/` ↔ Hub |

### Código fuente (`src/`)

| Módulo | Qué hace |
|---|---|
| `config.py` | Rutas, horizontes, folds de la validación cruzada, rejilla de hiperparámetros, intervalo del 80 % |
| `data_cleaning.py` | Carga de los datos en bruto y limpieza (festivos, WTI negativo, retraso de la EIA) |
| `feature_engineering.py` | 16 variables, objetivos (rentabilidad logarítmica a 5/63/252 días), escalado sin fuga de información y ventanas de 60 días |
| `models.py` | RNN / LSTM / GRU en PyTorch con el mismo bucle de entrenamiento (error de Huber) |
| `training.py` | Particiones temporales, conjunto de 5 semillas y entrenamiento del modelo de producción |
| `prediction.py` | Carga del modelo de producción y predicción de los tres horizontes (lo que se ejecuta cada día) |
| `evaluation.py` | Métricas (MAE, MSE, RMSE, R², acierto de dirección), intervalo del 80 % y análisis de errores |
| `explanations.py` | Texto en lenguaje sencillo y nivel de fiabilidad por horizonte |
| `decision_tools.py` | Simulador a 1-252 días (interpolación PCHIP del centro y cuantiles empíricos del error escalados con √t), movimientos históricos, escenarios de crisis y operaciones del backtest |
| `economic_events.py` | Picos y caídas del ciclo económico localizados sobre la serie real |
| `model_registry.py` | Publicación y descarga del modelo en Hugging Face Hub, con su ficha |
| `experiment_tracking.py` | Registro de las curvas de entrenamiento (TensorBoard y CSV) |

## Arquitectura de la aplicación

La aplicación es una web estática en GitHub Pages, pero **todo el cálculo lo hace Python** (descarga, limpieza, predicción, intervalos, explicaciones, comparativa) dentro de GitHub Actions. La web solo lee los JSON que genera `scripts/07_export_app_data.py` y los dibuja: ninguna cifra está escrita a mano. En el simulador, Python precalcula la trayectoria, los percentiles del error para cada día, los escenarios y las operaciones del backtest (`app/data/tools.json`); el navegador solo los combina con lo que escribe el usuario (días, precio, cantidad, costes).

```
GitHub Actions  (.github/workflows/daily_forecast.yml · cada día laborable)
│
├─ 1. scripts/01_download_data.py           FRED + Yahoo Finance        →  data/raw/
├─ 2. scripts/hf_model_registry.py download  modelo ya entrenado (Hub)   →  models/
├─ 3. scripts/07_export_app_data.py         predicción, histórico, eventos, comparativa → app/data/*.json
└─ 4. actions/deploy-pages                  app/  →  GitHub Pages
                                                  │
                    app/index.html + css + js (Apache ECharts)  ←  solo lectura de los JSON
```

## Ejecutar en local

Requisitos: Python ≥ 3.12 y [Poetry](https://python-poetry.org). `torch` se instala desde el índice CPU de PyTorch (sin CUDA).

```bash
poetry install                                          # crea .venv/ en el proyecto
poetry run python scripts/01_download_data.py           # datos recientes (FRED + Yahoo Finance)
poetry run python scripts/02_clean_data.py              # datos limpios y variables → data/processed/
poetry run python scripts/03_train_compare_models.py    # opcional: comparativa completa (60-70 min)
poetry run python scripts/04_significance_test.py       # opcional: Diebold-Mariano (reentrena las 3 redes en test)
poetry run python scripts/05_error_analysis.py          # opcional: MAE/MSE/RMSE/R² y análisis de errores
poetry run python scripts/06_train_production_model.py  # modelo de producción con la receta (~13 s)
# o, sin entrenar: HF_MODEL_REPO=Valdi183/brent-rnn-forecast poetry run python scripts/hf_model_registry.py download
poetry run python scripts/07_export_app_data.py         # JSON de la web → app/data/
python -m http.server -d app                            # abre http://localhost:8000
```

Notebook de análisis exploratorio: `poetry install --with notebooks` y abrir `notebooks/01_exploratory_analysis.ipynb` (se puede leer directamente en GitHub, con los gráficos ya generados).

`models/` no se versiona (el modelo se publica en Hugging Face Hub con `scripts/hf_model_registry.py`); por eso `06_train_production_model.py` puede partir de la receta versionada `config/production_model_config.json` (arquitectura, hiperparámetros, épocas, cuantiles del intervalo y métricas de test de la comparativa).

Las curvas de entrenamiento se pueden seguir con `poetry run tensorboard --logdir runs`.

## Actualización automática

Cada día laborable (07:30 UTC) el flujo `.github/workflows/daily_forecast.yml` descarga los datos nuevos (si una fuente falla, sigue con la instantánea de `data/raw/`), descarga el modelo ya entrenado desde **Hugging Face Hub**, pasa el último dato por el modelo, exporta los JSON y vuelve a publicar `app/` en GitHub Pages. **No se entrena nada**: incorporar datos nuevos y reentrenar son procesos distintos, y el reentrenamiento queda fuera de la actualización diaria. La publicación se hace como artefacto de Pages, así que la actualización no genera commits en el repositorio.

## Resultados clave

Validación cruzada walk-forward (2017-18, 2019-20 con COVID, 2021-22 con Ucrania), test final 2023 → hoy, cada modelo = conjunto de 5 semillas, épocas elegidas con la curva media de la CV. Detalle en [`reports/comparison.md`](reports/comparison.md) y [`reports/significance.md`](reports/significance.md).

| Red (config.) | Parámetros | Tiempo de entrenamiento | Mejora RMSE vs ingenuo: corto | medio | largo | Error CV |
|---|---:|---:|---:|---:|---:|---:|
| **RNN** (64×2, 2 ép.) — *en producción* | 13.763 | 11,7 s | **+1,2 %** (p = 0,01) | **+7,3 %** (p = 0,07) | +5,4 % (p = 0,69) | **0,909** |
| LSTM (32×1, 2 ép.) | 6.499 | 3,3 s | +0,5 % (p = 0,13) | +3,4 % (p = 0,05) | +4,3 % (p = 0,49) | 0,928 |
| GRU (32×1, 3 ép.) | 4.899 | 14,8 s | +0,8 % (p = 0,02) | +4,4 % (p = 0,08) | −8,0 % (p = 0,44) | 0,937 |

Tiempo de entrenamiento del conjunto de 5 semillas en CPU. MAE y RMSE de cada red y horizonte, en [`reports/comparison.md`](reports/comparison.md).

Por qué la RNN:

1. **Menor error de validación cruzada** en los tres regímenes de mercado, y el test lo confirma (mejor a corto y medio plazo).
2. **La única red que bate al ingenuo con p = 0,01 a corto plazo** (Diebold-Mariano): mejora pequeña pero demostrada → orientación de tendencia.
3. **La mejor a medio plazo** (+7,3 %, 62 % de acierto en la dirección): es donde el modelo aporta valor.
4. **A largo plazo ninguna red es fiable** (p ≥ 0,44 y una dispersión entre semillas enorme): la aplicación atenúa la cifra y destaca el rango del 80 %.
5. Coste contenido: 13,8k parámetros y ~13 s de reentrenamiento en CPU. En el día a día no se reentrena: los datos nuevos pasan por el modelo ya entrenado.

La combinación de las tres redes (media) se calcula solo como referencia: no se integra en la aplicación porque el enunciado pide integrar un único modelo.

## Validación fuera de muestra: ¿funciona el modelo?

La RNN se entrenó solo con datos hasta el 31/12/2022 y se comparan sus predicciones con los precios reales que hubo después (2023 → hoy, ~1.000 días). Detalle en [`reports/error_analysis.md`](reports/error_analysis.md) y en la sección *Validación* de la web.

| Horizonte | MAE ($) | MSE | RMSE ($) | R² precio (ingenuo) | R² fuera de muestra vs ingenuo |
|---|---:|---:|---:|---:|---:|
| Corto (5 d) | 3,63 | 28,92 | 5,38 | 0,862 (0,858) | **+2,3 %** |
| Medio (63 d) | 10,71 | 277,11 | 16,65 | −0,256 (−0,461) | **+14,1 %** |
| Largo (252 d) | 14,70 | 423,37 | 20,58 | −0,602 (−0,788) | **+10,4 %** |

- **Dos R²**: el R² sobre el precio engaña con series tan autocorrelacionadas (copiar el precio de hoy ya da 0,86 a corto plazo, y a medio/largo plazo sale negativo para todos porque 2023–25 fue un rango estrecho seguido del salto de 2026). Por eso se reporta también el **R² fuera de muestra frente al ingenuo** (1 − MSE/MSE ingenuo; Campbell y Thompson, 2008), positivo en los tres horizontes.
- **Cuándo acierta**: a medio plazo bate al ingenuo en todos los años (2023–2026) y en todos los regímenes de volatilidad; a corto plazo, con una mejora pequeña pero en todos los grupos.
- **Cuándo falla**: a corto y medio plazo, los mayores errores coinciden con la crisis de Irán y Ormuz de 2026, un shock que el ingenuo tampoco anticipa (falla casi lo mismo). A largo plazo sí hay errores propios del modelo: las predicciones hechas en 2023 y parte de 2024 anticipaban caídas que no llegaron (peor que el ingenuo en 2023, −69 %), mientras que las de 2024 en conjunto fueron mucho mejores (+44 %). La cifra a 12 meses no es fiable y la aplicación muestra el rango.
- Gráficos de real frente a predicho y del error móvil frente al ingenuo: [`reports/figures/error_analysis.png`](reports/figures/error_analysis.png).

## Fuentes de datos

- **FRED** (<https://fred.stlouisfed.org>): Brent y WTI spot (EIA), gas natural Henry Hub, tipo de la facilidad de depósito del BCE, dólar/euro, yuan/dólar y VIX.
- **Yahoo Finance** (<https://finance.yahoo.com>): futuros del Brent y del WTI, cobre, oro e índice del dólar.

Todos los datos son diarios (días hábiles) y gratuitos, sin clave. La EIA publica el Brent spot con unos días de retraso: hasta que llega el dato nuevo, se repite el último valor disponible (como indica el enunciado para los huecos).

La **cesta de la OPEP (ORB)** no se incluye: no está en FRED ni en Yahoo Finance y la OPEP solo la publica en su web, sin una descarga estable que una actualización diaria automática pueda usar. Además sigue muy de cerca al Brent, que ya es la variable objetivo, así que aportaría poca información nueva.
