# Comparativa RNN vs LSTM vs GRU

Generado: 2026-11-15 02:05 · Datos hasta 2026-11-11 · Test: 2023-01-01 → hoy · Errores en $/barril · Métricas de test = media de 3 semillas

| red   | horizonte   |   MAE |   RMSE |   MAE_ingenuo |   RMSE_ingenuo |   mejora_RMSE_% |   std_mejora_RMSE_% |   acierto_direccion_% |   cobertura_IC80_% |   parametros |   tiempo_entrenamiento_s |
|:------|:------------|------:|-------:|--------------:|---------------:|----------------:|--------------------:|----------------------:|-------------------:|-------------:|-------------------------:|
| RNN   | corto       |  3.67 |   5.44 |          3.66 |           5.44 |            0.02 |                1    |                 51.57 |              77.87 |         3811 |                      2.6 |
| RNN   | medio       | 12.76 |  19.14 |         11.59 |          17.96 |           -6.56 |                9.32 |                 57.39 |              86.35 |         3811 |                      2.6 |
| RNN   | largo       | 20.27 |  24.87 |         15.56 |          21.74 |          -14.36 |               27.98 |                 64.37 |              86.51 |         3811 |                      2.6 |
| LSTM  | corto       |  3.64 |   5.41 |          3.66 |           5.44 |            0.61 |                0.63 |                 54.47 |              77.07 |         6499 |                      1   |
| LSTM  | medio       | 11.58 |  17.56 |         11.59 |          17.96 |            2.2  |                2.18 |                 58.13 |              83.17 |         6499 |                      1   |
| LSTM  | largo       | 15.19 |  19.12 |         15.56 |          21.74 |           12.06 |                1.74 |                 54.01 |              73.94 |         6499 |                      1   |
| GRU   | corto       |  3.64 |   5.4  |          3.66 |           5.44 |            0.72 |                0.2  |                 53.07 |              78.07 |        40899 |                      9.5 |
| GRU   | medio       | 11.09 |  16.13 |         11.59 |          17.96 |           10.19 |                0.42 |                 57.53 |              85.4  |        40899 |                      9.5 |
| GRU   | largo       | 11.54 |  15.78 |         15.56 |          21.74 |           27.42 |                1.23 |                 76.01 |              94.97 |        40899 |                      9.5 |

**Modelo elegido: LSTM** (menor «error relativo al ingenuo + inestabilidad entre semillas» en la validación cruzada walk-forward).

## Comparativa a igual tamaño

Las tres redes con exactamente la misma configuración (neuronas y capas). Mejora del RMSE frente al modelo ingenuo en test (media ± desviación de 3 semillas). Tabla informativa: no interviene en la selección del modelo.

| Configuración      | Red   |   Parámetros |   Score CV | Mejora corto (%)   | Mejora medio (%)   | Mejora largo (%)   |   Tiempo (s) |
|:-------------------|:------|-------------:|-----------:|:-------------------|:-------------------|:-------------------|-------------:|
| hidden=32, capas=1 | RNN   |        1.699 |      0.863 | +1.0 ± 0.7         | -5.7 ± 6.1         | -5.0 ± 19.1        |          1.9 |
| hidden=32, capas=1 | LSTM  |        6.499 |      0.887 | +0.6 ± 0.6         | +2.2 ± 2.2         | +12.1 ± 1.7        |          0.8 |
| hidden=32, capas=1 | GRU   |        4.899 |      0.952 | -0.1 ± 0.3         | -6.1 ± 2.7         | -14.9 ± 8.5        |          2.6 |
| hidden=64, capas=1 | RNN   |        5.443 |      0.858 | +1.0 ± 0.2         | -0.8 ± 7.5         | -8.6 ± 23.2        |          1.3 |
| hidden=64, capas=1 | LSTM  |       21.187 |      0.906 | +1.0 ± 0.5         | -2.3 ± 1.0         | +1.0 ± 5.1         |          1.3 |
| hidden=64, capas=1 | GRU   |       15.939 |      0.917 | +0.1 ± 0.2         | +6.8 ± 0.6         | +15.2 ± 4.0        |          3.4 |
| hidden=32, capas=2 | RNN   |        3.811 |      0.851 | +0.0 ± 1.0         | -6.6 ± 9.3         | -14.4 ± 28.0       |          2.1 |
| hidden=32, capas=2 | LSTM  |       14.947 |      0.909 | +0.4 ± 0.6         | -1.2 ± 12.8        | -8.8 ± 17.6        |          1.9 |
| hidden=32, capas=2 | GRU   |       11.235 |      0.918 | -0.1 ± 0.7         | +7.3 ± 7.9         | +13.6 ± 12.5       |          5.4 |
| hidden=64, capas=2 | RNN   |       13.763 |      0.861 | +0.5 ± 0.4         | -0.9 ± 3.1         | +4.0 ± 6.3         |          3.1 |
| hidden=64, capas=2 | LSTM  |       54.467 |      0.903 | -0.9 ± 0.9         | -1.1 ± 3.6         | -6.5 ± 10.1        |          3.6 |
| hidden=64, capas=2 | GRU   |       40.899 |      0.913 | +0.7 ± 0.2         | +10.2 ± 0.4        | +27.4 ± 1.2        |          7.8 |
