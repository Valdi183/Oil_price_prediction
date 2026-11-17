# Brent Forecast · predicción del precio del petróleo con redes recurrentes

Caso de uso III de la asignatura *Aprendizaje automático* (UAX): comparar una **RNN simple, una LSTM y una GRU** para predecir el precio del **Brent** a corto (5 días hábiles), medio (63) y largo plazo (252), e integrar el mejor modelo en una aplicación que calcule las predicciones, cuantifique su incertidumbre y las explique en lenguaje sencillo.

**Web de la aplicación:** <https://valdi183.github.io/Oil_price_prediction/>

**Modelo en producción (Hugging Face Hub):** <https://huggingface.co/Valdi183/brent-rnn-forecast>

> Herramienta de apoyo a la decisión, no es una recomendación de inversión.

## Qué hace la aplicación

- Muestra el último precio del Brent y, para cada horizonte, el **precio previsto, la variación esperada y el rango de incertidumbre del 80 %**, dibujado como un abanico sobre el histórico.
- Genera una **explicación en lenguaje sencillo**: qué significa la predicción, qué fiabilidad tiene en ese horizonte según la comparativa (sello de Diebold-Mariano incluido) y qué factores pueden invalidarla.
- Marca en el gráfico los **picos y caídas del ciclo económico** (2008, 2014-2016, 2020, 2022, 2025-2026…) con una línea de tiempo navegable que hace zoom al periodo.
- Presenta la **comparativa de las tres redes** frente al modelo ingenuo (tablas, barras, real vs predicho en test y test de significación) y justifica la elección.
- Se **actualiza sola**: cada día una GitHub Action descarga los datos recientes y los pasa por el modelo ya entrenado (guardado en Hugging Face Hub), **sin reentrenar**, y vuelve a publicar la web. Reentrenar es una acción manual aparte; el botón «Actualizar ahora» lanza la Action.

## Arquitectura

La aplicación es una web estática en GitHub Pages, pero **todo el cálculo lo hace Python** (descarga, limpieza, predicción, intervalos, explicaciones, comparativa) dentro de GitHub Actions. La web solo lee los JSON que genera `scripts/export_site.py` y los dibuja: ninguna cifra está escrita a mano.

```
GitHub Actions  (.github/workflows/update-site.yml · cron días laborables + ejecución manual)
│
├─ 1. scripts/download_data.py   FRED (EIA, BCE, Fed, Cboe) + Yahoo Finance  →  data/petroleo_dataset.csv
├─ 2. scripts/retrain.py         receta config/production_recipe.json        →  models/  (RNN 64×2, 5 semillas, ~13 s)
├─ 3. scripts/export_site.py     forecast · history · events · models · meta →  site/data/*.json
└─ 4. actions/deploy-pages       site/  →  GitHub Pages
                                        │
                   site/index.html + css + js (Apache ECharts)  ←  solo lectura de los JSON
```

Módulos de Python (`oilcast/`):

| Módulo | Qué hace |
|---|---|
| `data.py` | Carga, limpieza (festivos, WTI negativo, retraso de la EIA), 16 variables, escalado sin fuga y ventanas de 60 días |
| `models.py` | RNN / LSTM / GRU en PyTorch con el mismo bucle de entrenamiento (error Huber) |
| `pipeline.py` | Particiones temporales, métricas, validación walk-forward, conjunto de semillas, modelo de producción y `forecast()` |
| `explain.py` | Texto en lenguaje sencillo y nivel de fiabilidad por horizonte (métricas de test + p-valor de Diebold-Mariano) |
| `evaluation.py` | Métricas fuera de muestra (MAE, MSE, RMSE, R²), errores por año y régimen, error móvil y peores fallos |
| `hub.py` | Publicación y descarga del modelo en Hugging Face Hub, con su ficha |
| `events.py` | Picos y caídas del ciclo económico localizados sobre la serie real y resumen de movimientos recientes |
| `config.py` | Horizontes, folds de la validación cruzada, rejilla de hiperparámetros, intervalo del 80 % |

Scripts:

| Script | Qué hace |
|---|---|
| `scripts/download_data.py` | Descarga los datos diarios desde 2000 |
| `scripts/train_compare.py` | Comparativa completa RNN vs LSTM vs GRU (60-70 min en CPU) → `reports/` y modelo de producción |
| `scripts/significance.py` | Test de Diebold-Mariano y combinación de referencia → `reports/significance.*`, `reports/test_predictions.csv` |
| `scripts/retrain.py` | Reentrena el modelo elegido con todos los datos (lee `models/production_meta.json` o, si no existe, `config/production_recipe.json`) |
| `scripts/error_analysis.py` | MAE, MSE, RMSE, R² y análisis de cuándo acierta y cuándo falla (sin entrenar) → `reports/error_analysis.md`, `reports/figures/error_analysis.png` |
| `scripts/model_hub.py` | Descarga (`download`) o publica (`upload`) el modelo de producción en Hugging Face Hub |
| `scripts/export_site.py` | Exporta los JSON de la web a `site/data/` (solo inferencia con el modelo de `models/`) |

## Ejecutar en local

Requisitos: Python ≥ 3.12 y [Poetry](https://python-poetry.org). `torch` se instala desde el índice CPU de PyTorch (sin CUDA).

```bash
poetry install                                   # crea .venv/ en el proyecto
poetry run python scripts/download_data.py       # datos recientes (FRED + Yahoo Finance)
poetry run python scripts/train_compare.py       # opcional: comparativa completa (60-70 min)
poetry run python scripts/significance.py        # opcional: Diebold-Mariano (reentrena las 3 redes en test)
poetry run python scripts/error_analysis.py      # opcional: MAE/MSE/RMSE/R² y análisis de errores (sin entrenar)
poetry run python scripts/retrain.py             # modelo de producción con la receta (~13 s)
# o, sin entrenar: HF_MODEL_REPO=usuario/nombre poetry run python scripts/model_hub.py download
poetry run python scripts/export_site.py         # JSON de la web → site/data/
python -m http.server -d site                    # abre http://localhost:8000
```

`models/` no se versiona (el modelo se publica en Hugging Face Hub con `scripts/model_hub.py`); por eso `retrain.py` puede partir de la receta versionada `config/production_recipe.json` (arquitectura, hiperparámetros, épocas, cuantiles del intervalo y métricas de test de la comparativa).

Las curvas de entrenamiento se pueden seguir con `poetry run tensorboard --logdir runs`.

## Actualización automática y GitHub Pages

El flujo `.github/workflows/update-site.yml`:

- se ejecuta **cada día laborable** (07:30 UTC), **a mano** (pestaña *Actions* → *Actualizar predicción y desplegar la web* → *Run workflow*; es el enlace del botón «Actualizar ahora» de la web) y al hacer push en `main` de ficheros que afecten a la web o al pipeline;
- **incorporar datos y reentrenar son procesos distintos**:
  - **actualización (por defecto, diaria)**: descarga los datos nuevos (si una fuente falla, sigue con la instantánea de `data/`) y el modelo ya entrenado desde **Hugging Face Hub**, y el último dato pasa por el modelo. **No se entrena nada**;
  - **reentrenamiento (solo a mano)**: *Run workflow* marcando *Reentrenar*. Entrena el conjunto con todos los datos (~13 s en CPU) y publica la nueva versión en el Hub, que pasa a ser la que usan las actualizaciones diarias;
- cachea `.venv` por el hash de `poetry.lock`, exporta los JSON y despliega `site/` con `actions/upload-pages-artifact` + `actions/deploy-pages`.

Configuración del modelo en Hugging Face (una vez):

1. Crea un token de escritura en <https://huggingface.co/settings/tokens>.
2. En GitHub: **Settings → Secrets and variables → Actions** → *Variables*: `HF_MODEL_REPO` = `usuario/nombre-del-modelo`; *Secrets*: `HF_TOKEN` = el token.
3. Publica el modelo actual (desde local, con `models/` entrenado): `HF_MODEL_REPO=usuario/nombre HF_TOKEN=... poetry run python scripts/model_hub.py upload`, o lanza la Action con *Reentrenar*.

Si el Hub aún no tiene modelo, la Action entrena uno temporal como respaldo para que la web no se rompa (y lo avisa).

Para activarlo en un fork: **Settings → Pages → Build and deployment → Source: GitHub Actions**. La URL será `https://<usuario>.github.io/<repo>/` (la web deduce el repositorio de `git remote` o de `GITHUB_REPOSITORY`).

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

## Estructura del repositorio

```
.github/workflows/update-site.yml   Action: datos → modelo del Hub (o reentrenar a mano) → exportar → Pages
config/production_recipe.json       receta del modelo de producción (versionada)
data/petroleo_dataset.csv           instantánea del dataset
docs/assignment/ENUNCIADO.pdf       enunciado del caso de uso
oilcast/                            código del proyecto (datos, modelos, pipeline, explicación, eventos)
reports/                            comparativa, significación, predicciones de test y figuras
scripts/                            descarga, comparativa, significación, reentrenamiento, exportación
site/                               web estática (index.html, css/, js/, assets/, data/)
```
