# Significación estadística (Diebold-Mariano) y combinación de referencia

Test de Diebold-Mariano sobre el periodo de test (2023 → hoy), pérdida cuadrática en $/barril, varianza HAC con h−1 retardos y corrección de Harvey-Leybourne-Newbold (las previsiones a h días se solapan). Celda = mejora del RMSE (%) del modelo de la fila frente al rival y p-valor. Significación: \* 10 %, \*\* 5 %, \*\*\* 1 %.

**Aviso:** a largo plazo el test solo contiene ~3–4 años no solapados, así que su potencia es muy baja.

## Cada modelo frente al ingenuo

| Modelo | Corto plazo | Medio plazo | Largo plazo |
|---|---|---|---|
| RNN | +1.2 % (p=0.01)*** | +7.3 % (p=0.07)* | +5.4 % (p=0.69) |
| LSTM | +0.5 % (p=0.13) | +3.4 % (p=0.05)* | +4.3 % (p=0.49) |
| GRU | +0.8 % (p=0.02)** | +4.4 % (p=0.08)* | -8.0 % (p=0.44) |
| Combinación | +0.8 % (p=0.01)*** | +5.3 % (p=0.04)** | +1.8 % (p=0.84) |

## Redes entre sí

| Comparación | Corto plazo | Medio plazo | Largo plazo |
|---|---|---|---|
| RNN vs LSTM | +0.7 % (p=0.13) | +4.1 % (p=0.16) | +1.1 % (p=0.92) |
| RNN vs GRU | +0.4 % (p=0.11) | +3.0 % (p=0.32) | +12.4 % (p=0.28) |
| LSTM vs GRU | -0.3 % (p=0.33) | -1.1 % (p=0.37) | +11.4 % (p=0.03)** |

## Combinación de referencia (media de las tres redes)

Solo informativa: no interviene en la selección ni se integra en la aplicación.

| Modelo | Horizonte | MAE | RMSE | Mejora vs ingenuo (%) | Acierto dirección (%) | Parámetros | Tiempo (s) |
|---|---|---|---|---|---|---|---|
| RNN (h64·l2·lr0.001, 2 ép.) | corto | 3.63 | 5.38 | +1.2 | 53.9 | 13.763 | 11.9 |
| RNN (h64·l2·lr0.001, 2 ép.) | medio | 10.71 | 16.65 | +7.3 | 62.1 | 13.763 | 11.9 |
| RNN (h64·l2·lr0.001, 2 ép.) | largo | 14.70 | 20.58 | +5.4 | 79.9 | 13.763 | 11.9 |
| LSTM (h32·l1·lr0.001, 2 ép.) | corto | 3.65 | 5.42 | +0.5 | 52.6 | 6.499 | 3.2 |
| LSTM (h32·l1·lr0.001, 2 ép.) | medio | 11.21 | 17.35 | +3.4 | 62.2 | 6.499 | 3.2 |
| LSTM (h32·l1·lr0.001, 2 ép.) | largo | 13.50 | 20.80 | +4.3 | 73.5 | 6.499 | 3.2 |
| GRU (h32·l1·lr0.001, 3 ép.) | corto | 3.64 | 5.40 | +0.8 | 53.6 | 4.899 | 15.4 |
| GRU (h32·l1·lr0.001, 3 ép.) | medio | 10.92 | 17.16 | +4.4 | 63.3 | 4.899 | 15.4 |
| GRU (h32·l1·lr0.001, 3 ép.) | largo | 17.02 | 23.48 | -8.0 | 65.5 | 4.899 | 15.4 |
| **Combinación** | corto | 3.64 | 5.40 | +0.8 | 54.7 | 25.161 | 30.5 |
| **Combinación** | medio | 10.88 | 17.00 | +5.3 | 64.9 | 25.161 | 30.5 |
| **Combinación** | largo | 14.41 | 21.35 | +1.8 | 73.7 | 25.161 | 30.5 |
