# Comparativa RNN vs LSTM vs GRU

Generado: 2026-11-15 17:43 · Datos hasta 2026-11-11 · Test: 2023-01-01 → hoy · Errores en $/barril · Cada modelo = conjunto de 5 semillas (`mejora_RMSE_%` del conjunto; `mejora_media_miembros_%` y `std_mejora_RMSE_%` de las redes individuales) · Épocas elegidas con la curva media de la CV

| red   | hiperparametros   |   epocas | horizonte   |   MAE |   RMSE |   MAE_ingenuo |   RMSE_ingenuo |   mejora_RMSE_% |   mejora_media_miembros_% |   std_mejora_RMSE_% |   acierto_direccion_% |   cobertura_IC80_% |   parametros |   tiempo_entrenamiento_s |
|:------|:------------------|---------:|:------------|------:|-------:|--------------:|---------------:|----------------:|--------------------------:|--------------------:|----------------------:|-------------------:|-------------:|-------------------------:|
| RNN   | h64·l2·lr0.001    |        2 | corto       |  3.63 |   5.38 |          3.66 |           5.44 |            1.16 |                      0.97 |                0.26 |                 53.9  |              76.17 |        13763 |                     11.7 |
| RNN   | h64·l2·lr0.001    |        2 | medio       | 10.71 |  16.65 |         11.59 |          17.96 |            7.3  |                      6.04 |                4.47 |                 62.12 |              84.55 |        13763 |                     11.7 |
| RNN   | h64·l2·lr0.001    |        2 | largo       | 14.7  |  20.58 |         15.56 |          21.74 |            5.37 |                      1.54 |               16.1  |                 79.89 |              87.83 |        13763 |                     11.7 |
| LSTM  | h32·l1·lr0.001    |        2 | corto       |  3.65 |   5.42 |          3.66 |           5.44 |            0.45 |                      0.34 |                0.36 |                 52.6  |              77.37 |         6499 |                      3.3 |
| LSTM  | h32·l1·lr0.001    |        2 | medio       | 11.21 |  17.35 |         11.59 |          17.96 |            3.38 |                      3.04 |                1.22 |                 62.22 |              84.34 |         6499 |                      3.3 |
| LSTM  | h32·l1·lr0.001    |        2 | largo       | 13.5  |  20.8  |         15.56 |          21.74 |            4.31 |                      3.01 |                9.11 |                 73.54 |              88.36 |         6499 |                      3.3 |
| GRU   | h32·l1·lr0.001    |        3 | corto       |  3.64 |   5.4  |          3.66 |           5.44 |            0.77 |                      0.67 |                0.43 |                 53.6  |              76.77 |         4899 |                     14.8 |
| GRU   | h32·l1·lr0.001    |        3 | medio       | 10.92 |  17.16 |         11.59 |          17.96 |            4.42 |                      4.18 |                2.95 |                 63.28 |              82.65 |         4899 |                     14.8 |
| GRU   | h32·l1·lr0.001    |        3 | largo       | 17.02 |  23.48 |         15.56 |          21.74 |           -8    |                     -8.44 |               11.08 |                 65.48 |              83.33 |         4899 |                     14.8 |

**Modelo elegido: RNN** (menor error de validación cruzada walk-forward del conjunto).

## Comparativa a igual tamaño

Las tres redes con exactamente la misma configuración (neuronas, capas y learning rate), cada una con sus épocas elegidas en la validación cruzada. Mejora del RMSE del conjunto de 5 semillas frente al modelo ingenuo en test; entre paréntesis, la desviación entre miembros (estabilidad). Tabla informativa: no interviene en la selección del modelo.

| Configuración   | Red   |   Parámetros (por red) |   Épocas |   Score CV | Mejora corto (%)   | Mejora medio (%)   | Mejora largo (%)   |   Tiempo (s) |
|:----------------|:------|-----------------------:|---------:|-----------:|:-------------------|:-------------------|:-------------------|-------------:|
| h32·l1·lr0.001  | RNN   |                  1.699 |        5 |      0.932 | +1.3 (±0.5)        | +7.2 (±6.1)        | +20.4 (±18.8)      |          9.4 |
| h32·l1·lr0.001  | LSTM  |                  6.499 |        2 |      0.928 | +0.5 (±0.4)        | +3.4 (±1.2)        | +4.3 (±9.1)        |          3.6 |
| h32·l1·lr0.001  | GRU   |                  4.899 |        3 |      0.937 | +0.8 (±0.4)        | +4.4 (±3.0)        | -8.0 (±11.1)       |         14.6 |
| h64·l1·lr0.001  | RNN   |                  5.443 |       13 |      0.922 | +1.2 (±0.5)        | +0.4 (±5.9)        | -39.0 (±10.3)      |         30.7 |
| h64·l1·lr0.001  | LSTM  |                 21.187 |        1 |      0.96  | +0.5 (±0.1)        | +0.2 (±1.0)        | -1.7 (±5.4)        |          2.9 |
| h64·l1·lr0.001  | GRU   |                 15.939 |        2 |      0.939 | +0.6 (±0.2)        | +3.5 (±2.0)        | -22.2 (±8.6)       |         13.3 |
| h32·l2·lr0.001  | RNN   |                  3.811 |        3 |      0.923 | +1.0 (±0.6)        | +8.1 (±5.7)        | +14.1 (±33.9)      |         12   |
| h32·l2·lr0.001  | LSTM  |                 14.947 |        2 |      0.94  | +0.4 (±0.1)        | +4.3 (±2.9)        | +6.9 (±10.1)       |          7   |
| h32·l2·lr0.001  | GRU   |                 11.235 |        2 |      0.946 | +0.1 (±0.3)        | +3.3 (±2.1)        | -21.6 (±8.1)       |         20.6 |
| h64·l2·lr0.001  | RNN   |                 13.763 |        2 |      0.909 | +1.2 (±0.3)        | +7.3 (±4.5)        | +5.4 (±16.1)       |         13.1 |
| h64·l2·lr0.001  | LSTM  |                 54.467 |        1 |      0.933 | +0.0 (±0.2)        | +0.1 (±1.7)        | -10.0 (±6.4)       |          6.5 |
| h64·l2·lr0.001  | GRU   |                 40.899 |        2 |      0.949 | +0.9 (±0.3)        | +12.0 (±1.5)       | +12.2 (±13.1)      |         29.4 |
