# Análisis exploratorio de `fraud_dataset.csv`

Números calculados sobre el CSV completo. Los gráficos de tasa de fraude por feature salen de [`fraud_analysis.py`](fraud_analysis.py).

## 1. Composición

- Son 7500 filas y 11 columnas. No hay nulos ni filas duplicadas.
- Hay 9 features, más `big_model_fraud_probability` (la salida de BigModel, que es el **target** de la destilación) y `flagged_fraud` (la verdad observada, que **no se usa para entrenar**).

## 2. Balance de clases

- `flagged_fraud = 1` en 869 filas: **11.6 %**. Predecir siempre "no fraude" da 88.4 % de accuracy, así que la accuracy sola no sirve como métrica.
- `big_model_fraud_probability` tiene media 0.42 y mediana 0.36, con un pico cerca de 1: el 7.1 % de las filas tiene p > 0.99.

## 3. Rangos

| Columna | Mín | Mediana | Máx | Observación |
|---|---|---|---|---|
| `timestamp` | 1.70e9 | 1.72e9 | 1.73e9 | ~1 año de compras (nov 2023 a nov 2024) |
| `amount_usd` | 1 | 63.49 | 2000 | Muy sesgada a la derecha. El máximo parece truncado |
| `quantity_purchased` | 1 | 5 | 24 | |
| `session_duration_seconds` | 5 | 287 | 726.8 | |
| `days_since_last_purchase` | 0 | 8.59 | 142.3 | Sesgada |
| `account_age_days` | 1 | 1633 | 3649 | Hasta ~10 años |
| `device_screen_resolution` | 1.0e6 | 2.07e6 | 8.3e6 | Ver §4 |
| `time_since_last_login_s` | 10 | 2478 | 40161 | Sesgada |
| `items_viewed_before_purchase` | 1 | 8 | 29 | |

Las escalas difieren en 9 órdenes de magnitud (`timestamp` contra `quantity_purchased`). **Estandarizar es obligatorio.** Sin eso, la excitación h satura la sigmoide desde el primer paso.

## 4. ¿Los datos están limpios?

- **Montos redondos:** `amount_usd == 100` aparece 156 veces, con 98.7 % de fraude. En cambio, `amount_usd == 1` aparece 82 veces con 0 % de fraude. Si se toman los múltiplos de 50, son 172 filas con 93 % de fraude. No son errores: son un patrón del fraude, y sirven para construir una feature (ver `docs/opcionales_ej1.md`).
- **Resolución de pantalla:** la documentación habla de valores comunes, como 1 049 088 (1366×768) o 2 073 600 (1920×1080). En los datos aparecen valores *cercanos* con ruido (1 045 573, 2 070 176, 3 686 974…), y ninguno se repite más de 3 veces. La columna es ruidosa. Además, un producto ancho×alto no se puede descomponer de forma única.
- No hay valores imposibles: no hay negativos ni duraciones de sesión nulas.

## 5. Correlación con el target (`big_model_fraud_probability`)

| Feature | Pearson |
|---|---|
| `quantity_purchased` | +0.563 |
| `amount_usd` | +0.557 |
| `items_viewed_before_purchase` | +0.334 |
| `device_screen_resolution` | +0.025 |
| `time_since_last_login_s` | +0.002 |
| `timestamp` | +0.001 (la hora del día da −0.019) |
| `days_since_last_purchase` | −0.404 |
| `session_duration_seconds` | −0.514 |
| `account_age_days` | −0.585 |

Se usan las 6 features con señal. Se descartan `timestamp`, `device_screen_resolution` y `time_since_last_login_s`, que tienen correlación ≈ 0. En `generalizacion.py` se comprueba que agregarlas no mejora el MSE de validación más allá de la variación entre folds (ver `resultados/generalizacion/comparacion_features.csv`).

## 6. Relación entre BigModel y `flagged_fraud`

- `flagged_fraud == (big_model_fraud_probability > 0.85)` en **todas** las filas. El máximo de p entre los no fraudes es 0.849891 y el mínimo entre los fraudes es 0.850089.
- Es decir, BigModel con umbral 0.85 separa perfecto: AUC = 1. Ese es el umbral de referencia.
- TinyModel imita a p, no a `flagged_fraud`, y su salida queda "comprimida" respecto de la de BigModel. Por eso su umbral óptimo no tiene por qué ser 0.85: en `generalizacion.py` se elige con predicciones out-of-fold.
