# Ej. 1 · Opcionales teóricos

El opcional práctico, ReLU, está en [`relu.py`](../relu.py) y en `resultados/relu/`.

## 1. Features: qué construir y qué descartar

### Candidatas a descartar

| Feature | Motivo |
|---|---|
| `timestamp` | Correlación 0.001 con p. La hora del día da −0.019. Como número crudo solo ordena en el tiempo y no aporta señal. Además, usarlo en crudo invita a que el modelo aprenda "la época" del dataset y no el patrón de fraude. |
| `time_since_last_login_s` | Correlación 0.002. |
| `device_screen_resolution` | Correlación 0.025. Es ruidosa: no toma los valores estándar de la documentación (ver `edas/analisis_dataset.md` §4). Además, el producto ancho×alto no es monótono con nada que tenga sentido. |

En `generalizacion.py` (`comparacion_features.csv`), usar las 9 features en lugar de 6 mejora el MSE de validación en ~6e-5. Eso está dentro de la variación entre folds (desvío ~2.5e-4), así que descartarlas no cuesta nada y el modelo queda más chico.

### Candidatas a construir

Un perceptrón simple calcula θ(w·x + b): solo modela relaciones **monótonas** en cada feature y **sin interacciones**. Todo lo que no sea monótono o dependa de combinar features hay que dárselo ya construido.

| Feature nueva | De dónde sale | Por qué ayudaría |
|---|---|---|
| `is_round_amount` (monto múltiplo de 50 o 100) | `amount_usd` | El monto 100 exacto tiene 98.7 % de fraude, y los múltiplos de 50, 93 %. Es un pico aislado que ningún peso lineal sobre `amount_usd` puede capturar. |
| `unit_price = amount_usd / quantity_purchased` | Interacción | Distingue "muchas unidades baratas" de "una unidad cara". Hoy ambos casos suben h por caminos separados. |
| `log(amount_usd)`, `log(1 + days_since_last_purchase)`, `log(time_since_last_login_s)` | Variables sesgadas | Comprimen la cola larga: con z-score, unos pocos montos de 2000 dominan la escala. |
| `items_viewed / session_duration` | Interacción | Ritmo de navegación: un bot o una compra apurada ve muchos ítems en poco tiempo. |
| `is_new_account` (edad < 30 días) o `log(account_age_days)` | `account_age_days` | La relación con el fraude probablemente se concentra en cuentas muy nuevas: es un umbral, no una recta. |
| Hora local, día de semana y "horario nocturno" | `timestamp` | Acá no mostraron señal. Se pueden argumentar, pero habría que convertirlos a variables cíclicas (sen/cos) o categóricas, porque la hora 23 está cerca de la 0. |
| "Resolución no estándar" (distancia a la resolución común más cercana) | `device_screen_resolution` | Un emulador o un navegador automatizado suele reportar resoluciones raras. |

## 2. Calibración

Un modelo está **calibrado** cuando sus probabilidades se pueden leer literalmente: entre las transacciones a las que les asigna p ≈ 0.7, alrededor del 70 % resulta fraude.

### ¿Por qué hace falta analizarla en este caso?

1. **El cliente pidió probabilidades.** La consigna define la salida como "0 = 0 %, 1 = 100 % de probabilidad de fraude". Si el número no está calibrado, cualquier decisión que se tome sobre él (costo esperado, priorizar la revisión manual) es incorrecta.
2. **TinyModel hereda la calibración de BigModel, sea cual sea.** Destilar es imitar p, no `flagged_fraud`. Lo medimos: el perceptrón sigmoide se entrenó con 6000 filas, se evaluó sobre 1500 no vistas y se comparó por tramo de salida.

| Salida de TinyModel | n | Media TinyModel | Media BigModel | **% de fraude real** |
|---|---|---|---|---|
| 0.0–0.1 | 142 | 0.059 | 0.049 | 0 % |
| 0.3–0.4 | 189 | 0.345 | 0.345 | 0 % |
| 0.5–0.6 | 144 | 0.546 | 0.544 | **0 %** |
| 0.7–0.8 | 71 | 0.748 | 0.767 | 12.7 % |
| 0.8–0.9 | 65 | 0.849 | 0.829 | 38.5 % |
| 0.9–1.0 | 149 | 0.981 | 0.967 | 89.3 % |

TinyModel es **fiel** a BigModel: las medias por tramo coinciden con un error de ±0.02. Pero ninguno de los dos está calibrado respecto del fraude real: con "probabilidad 0.55" no hubo ni un fraude, y el error de calibración esperado (ECE) contra `flagged_fraud` da **0.31**. La salida de BigModel funciona como un **score** con un corte en 0.85 (§6 del EDA), no como una probabilidad.

3. **El umbral no se traslada.** Como TinyModel comprime la salida cerca del corte, su umbral óptimo (0.80–0.89 según el criterio, ver `generalizacion.py`) no tiene por qué coincidir con el 0.85 de BigModel. Calibrar ambos contra `flagged_fraud` pondría los umbrales en la misma escala.

### ¿Cómo se ajustaría?

Después de entrenar TinyModel, se ajusta un calibrador **sobre un set aparte** (o sobre las predicciones out-of-fold), usando `flagged_fraud` como verdad:

- **Platt scaling:** una regresión logística de un parámetro sobre la salida, p_cal = σ(a·s + b). Sirve si la distorsión tiene forma de sigmoide.
- **Isotonic regression:** una función escalonada monótona. Es más flexible y necesita más datos. Con 869 fraudes alcanza.
- **Verificación:** un reliability diagram (la tabla de arriba, graficada) antes y después, más el ECE y el Brier score.

Calibrar no cambia el ranking: el ROC-AUC queda igual. Cambia la interpretación de la salida, y eso hace que el umbral y los costos se puedan razonar en términos de probabilidad real.
