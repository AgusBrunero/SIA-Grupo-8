# TP3 — Perceptrón simple y multicapa

Perceptrón simple (escalón, lineal y no lineal) y perceptrón multicapa con backpropagation matricial, implementados desde cero con numpy. Consigna completa en [`consigna.md`](consigna.md).

> **Lo distintivo: [`pizarra.py`](pizarra.py).** Es una app interactiva (tkinter) con tres pestañas:
>
> 1. **Pizarra.** Dibujás un dígito con el mouse y la red lo lee en tiempo real. Se ven:
>    - las probabilidades de los modelos del Ej. 2 y del Ej. 3, lado a lado (el del Ej. 2 nunca vio un 8: dibujá uno);
>    - qué neuronas ocultas se activan (con click ves los pesos de cada una como imagen 28×28);
>    - el mapa de atribución (gradiente × entrada);
>    - un slider de ruido gaussiano.
> 2. **Entrenamiento en vivo.** Un MLP aprende delante tuyo. Los pesos de cada neurona pasan de ruido a trazos y la matriz de confusión se va llenando. Con `digits.csv` solo, la fila del 8 queda vacía y el techo de accuracy es ~90 %. El modelo entrenado se puede llevar a la pizarra.
> 3. **Fraude · TinyModel.** Armás una transacción con sliders y ves la probabilidad de fraude y cuánto empuja cada feature. También elegís el umbral según cuánto cuesta un fraude no detectado frente a una falsa alarma.

## Setup

```bash
cd TP3
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Los CSV de dígitos pesan 75–95 MB y **no están en git**. Hay que copiarlos del zip de la cátedra (`data and documentation.zip`) a `datasets/`: `digits.csv`, `digits_test.csv` y `more_digits.csv`. La primera carga los parsea y los cachea en `datasets/cache/`, y tarda ~7 s.

## Cómo correr

| Qué | Comando | Salida |
|---|---|---|
| Tests de validación (AND, XOR, y=x, tanh, gradient check, cuenta a mano) | `python tests_validacion.py` | consola |
| Ej. 1 · aprendizaje | `python aprendizaje.py` | `resultados/aprendizaje/` |
| Ej. 1 · generalización y umbral | `python generalizacion.py` | `resultados/generalizacion/` |
| Ej. 1 · opcional ReLU | `python relu.py` | `resultados/relu/` |
| Ej. 2 · barridos + modelo final | `python ej2.py` (o `barridos` / `final`) | `resultados/digitos/ej2/`, `modelos/ej2_final.npz` |
| Ej. 3 · ablación + modelo final | `python ej3.py todo --extra final_ruido` | `resultados/digitos/ej3/`, `modelos/ej3_*.npz` |
| Gráficos y tablas de Ej. 2, Ej. 3 y opcionales | `python analisis_digitos.py` | `resultados/digitos/` |
| **App interactiva** | `python pizarra.py` | — |

Los modelos finales ya están versionados en `modelos/`, así que la pizarra funciona sin reentrenar.

## Estructura

```
perceptron/
  perceptron.py      perceptrón simple (escalón, lineal, sigmoide, tanh, ReLU); online y batch
  mlp.py             MLP: feed-forward y backprop matricial, bias por capa, β, dropout, save/load, saliency
  optimizadores.py   GD, Momentum, RMSProp, Adam (misma interfaz step), η adaptativo, L2
  entrenamiento.py   loop: online / mini-batch / batch, early stopping, historial por época
datos.py             dataset de fraude (features, estandarización)
digitos.py           dataset de dígitos: carga con caché, one-hot, duplicados, split estratificado, augmentation, ruido
experimentos.py      runner de barridos: config JSON × semillas en paralelo → corridas.jsonl (retomable)
configs/ej2.json     config base + barridos del Ej. 2 (una variable por vez) + config final
configs/ej3.json     ablación acumulativa del Ej. 3 + configs finales
analisis_digitos.py  SOLO lee resultados/ y modelos/ y grafica (separado de los experimentos)
edas/                exploración del dataset de fraude
docs/opcionales_ej1.md  opcionales teóricos del Ej. 1 (features y calibración)
```

Se siguieron las recomendaciones de la consigna:
- operaciones matriciales;
- progreso por época en consola;
- configuración en JSON que queda guardada junto a cada corrida;
- guardado y carga de modelos (`.npz`, con su config);
- experimentos separados del análisis.

## Ejercicio de validación

Todos pasan (`python tests_validacion.py`):

| Test | Resultado |
|---|---|
| AND con escalón | Converge en las 10 semillas probadas |
| XOR con escalón | No converge en ninguna (1000 épocas): no es linealmente separable |
| Lineal, y = x | w → 1 y b → 0 (error < 1e-3) |
| No lineal, y = tanh(x) | Error < 1e-4 |
| XOR con MLP `[2,2,1]` y `[2,3,2,1]` | 100 % en ≥ 8 de 10 semillas. `[2,2,1]` puede caer en un mínimo local |
| Gradient check | Backprop contra diferencias finitas, error relativo < 1e-6 (tanh, logística, ReLU, softmax + CE) |
| Cuenta a mano de `[2,2,1]` | Una iteración escalar por escalar coincide con el código matricial |
| Optimizadores | Los cuatro reducen la pérdida en un problema no lineal |

## Ejercicio 1 — TinyModel de fraude

Se entrena con las 6 features con señal (ver [`edas/analisis_dataset.md`](edas/analisis_dataset.md)). El target es `big_model_fraud_probability`, porque es Knowledge Distillation. `flagged_fraud` solo se usa para evaluar y para elegir el umbral.

### Aprendizaje (todas las muestras)

| | MSE | R² | Salidas fuera de [0,1] |
|---|---|---|---|
| Lineal | 0.0261 | 0.714 | 5.9 % |
| **Sigmoide** | **0.0110** | **0.880** | 0 % |
| ReLU (opcional) | 0.0258 | 0.718 | 4.7 % (> 1) |

**a) Underfitting.** Sí, en el lineal. Su error de entrenamiento se queda en el óptimo lineal cerrado (OLS), 2.4 veces el del sigmoide. Donde más se nota es en los extremos: entre 0.8 y 1 el MAE es 0.19 contra 0.04 del sigmoide.

**b) Saturación de capacidades.** Sí. Los dos llegan a un piso que no baja con más épocas ni cambiando η: el desvío entre semillas es ~0, así que es un límite del modelo y no del optimizador. El lineal, además, produce "probabilidades" fuera de [0,1].

**c) Elección.** El **sigmoide**: tiene más capacidad y su salida está acotada a (0,1), como la del target.

### Generalización

**a) Métricas.** Hay dos niveles:
- **Fidelidad** a BigModel: MSE, MAE y R², que es lo que se optimiza.
- **Detección** contra `flagged_fraud`: ROC-AUC, PR-AUC, precision, recall, F1 y F2. Con 11.6 % de fraude, la accuracy sola engaña.

**b) Estrategia.**
- Se separa un 20 % de test estratificado que no se toca hasta el final.
- Sobre el 80 % restante se hace **k-fold estratificado (k=5)** para elegir η, épocas, features y umbral. El umbral se elige con predicciones out-of-fold.
- Además, 50 particiones distintas (estratificadas contra aleatorias) muestran cuánto varía el resultado según el split. El "mejor conjunto de entrenamiento" no es el fold que dio mejor, que puede ser suerte. Es uno **representativo**, con la misma proporción de fraude y la misma distribución que la población. Lo que se elige son hiperparámetros por desempeño **promedio**, y el modelo final se reentrena con todo dev.

**c) Mejor modelo.**
- 6 pesos + bias, con η = 20 y 41 épocas en batch.
- En test: MSE 0.0110 (igual que en entrenamiento, sin sobreajuste) y ROC-AUC 0.993.
- **Umbral recomendado: 0.89** (máximo F1: precision 0.90, recall 0.86). Si un fraude no detectado cuesta más que una falsa alarma, **0.80** (máximo F2: recall 0.94).
- BigModel usa 0.85 implícitamente. En la pestaña de fraude de la pizarra se ve cómo se mueve el umbral de mínimo costo según la relación FN/FP.

### Opcionales

- **ReLU** ([`relu.py`](relu.py)): se comporta como el lineal truncado en 0. Tiene el mismo underfitting, salidas mayores a 1 y un 1.9 % de neuronas con h ≤ 0 que no aprenden. Rankea bien (AUC 0.993), pero no sirve como probabilidad.
- **Features y calibración** (teóricos): [`docs/opcionales_ej1.md`](docs/opcionales_ej1.md). Hallazgo: TinyModel es fiel a BigModel, pero **BigModel no está calibrado**. Entre las transacciones con p ≈ 0.55 no hubo ni un fraude (ECE 0.31 contra `flagged_fraud`).

## Ejercicio 2 — Dígitos con MLP (`digits.csv`)

- `digits.csv` se divide en 80 % train y 20 % validación, estratificado y con una partición fija para comparar.
- `digits_test.csv` se usa **una sola vez**, con el modelo final.
- Cada variante se corre con 3 semillas (media ± desvío) y se cambia una variable por vez sobre la base de la cátedra: `[784,64,10]`, tanh oculta, logística a la salida, MSE, Adam η=1e-3, mini-batch 64, 40 épocas.
- Tabla completa en `resultados/digitos/ej2/resumen_barridos.csv`.

### (a) ¿Cómo evalúo el desempeño?

- **Accuracy** global, porque el test está balanceado.
- **Matriz de confusión 10×10.**
- **Precision, recall y F1 por clase**, porque la accuracy global esconde qué clase falla.
- **Curvas de pérdida y accuracy de train contra validación por época**, para detectar under y overfitting.
- **Tiempo de entrenamiento**, que también es parte del costo.

### (b) Variantes

| Factor | Resultado (accuracy máxima en validación) |
|---|---|
| **Tasa de aprendizaje** | Cada optimizador tiene su ventana: GD 0.5 → 95.8 %, Momentum 0.05 → 95.8 %, RMSProp 1e-3 → 96.1 %, Adam 1e-3 → 95.9 %. Con η demasiado grande **todos divergen** (10–20 %, azar). Con η chico convergen lento (94 % en 40 épocas). η adaptativo sobre GD da lo mismo que el mejor η fijo |
| **Optimizador** | A su mejor η, todos terminan en ~96 % de validación. RMSProp llega a 94 % en la época 4, contra 7 de GD y Adam y 10 de Momentum. RMSProp y Adam bajan la pérdida de train 2–3 veces más (0.003–0.005 contra ~0.010), pero eso no se traduce en validación: la diferencia pasa a ser sobreajuste |
| **Arquitectura** | Más neuronas ayudan hasta ~256 (16 → 93.9 %, 64 → 95.9 %, 256 → 96.3 %), con rendimiento decreciente y costo creciente (2.7 s → 21 s). Tres capas `[256,128,64]` → 96.7 % |
| Activación oculta | ReLU 96.3 % > tanh 95.9 % ≈ logística 95.8 % |
| Tamaño de batch | Online (1) 96.0 % pero **30 veces más lento** (127 s), mini-batch 16 → 96.3 %, 64 → 95.9 %, 256 → 95.4 %, **batch completo 76 %** (solo 40 actualizaciones en 40 épocas) |
| Pérdida | Softmax + cross-entropy 96.1 % contra logística + MSE 95.9 %. Llega a su mejor época en 17 épocas en lugar de 38 |
| β, inicialización | Efecto chico: β = 2 da 96.0 % y β = 0.5 da 95.7 %. Xavier y uniforme chica quedan iguales |
| **Combinación** | ReLU + softmax/CE + `[784,256,10]` + mini-batch 16 + Adam 1e-3 → **97.2 ± 0.1 %** |

### Modelo final y resultado en "producción"

- Es la combinación ganadora: 203 530 parámetros, con early stopping por pérdida de validación.
- **Accuracy en digits_test: 85.5 %.** Si se sacan los ochos, la accuracy sobre el resto es 94.8 %.
- La matriz de confusión muestra que **el 8 tiene recall 0**: no hay ningún 8 en `digits.csv`. De los 243 ochos del test, 109 se predicen como 3, 28 como 2 y 27 como 9.
- El 5, con apenas 271 imágenes de entrenamiento, tiene recall 0.78.
- **El techo lo pone el dato, no el modelo:** aunque acertara todo lo demás, sin ochos el máximo es 90.3 %. Ningún hiperparámetro lo mueve.
- Nota: el early stopping por pérdida de validación corta en la época 4. Con cross-entropy, la pérdida de validación empieza a subir temprano, cuando la red se vuelve demasiado segura, aunque la accuracy siga mejorando unas épocas más.

## Ejercicio 3 — ≥ 98 % con `more_digits.csv`

- Se combinan `digits` y `more_digits` **sin duplicados**: 3689 imágenes de `more_digits` ya estaban en `digits`, así que quedan 24 501 únicas.
- Se separa un 15 % de validación estratificada.
- **Se elige por balanced accuracy en validación.** El test está balanceado y el train no: el 5 y el 8 siguen siendo minoría. Por eso la accuracy común sobreestima.
- Se hace una ablación **acumulativa**: cada paso suma una técnica al anterior, con 3 semillas por paso. La accuracy en test se muestra solo como referencia de "producción" y no se usa para elegir.

| Paso | Validación (balanced) | digits_test | Recall 5 / 8 en test |
|---|---|---|---|
| 1. Config final del Ej. 2, entrenada solo con las imágenes de `digits` | 86.6 % | 86.8 % | 0.85 / **0.00** |
| 2. + `more_digits` sin duplicados | 96.5 % | 97.0 % | 0.92 / 0.94 |
| 3. + early stopping (paciencia 6) | 95.8 % | 96.4 % | 0.92 / 0.90 |
| 4. + balance de clases (oversampling de 5 y 8) | 96.3 % | 96.8 % | 0.94 / 0.95 |
| 5. + augmentation (traslación ±2 px, rotación ±10°, solo en train) | 97.9 % | **98.1 %** | 0.97 / 0.96 |
| 6. + L2 (λ = 1e-4) | 97.6 % | 98.0 % | 0.97 / 0.97 |
| 7. + capa oculta de 512 | 97.8 % | **98.2 %** | 0.96 / 0.97 |
| 7 con L2 λ = 1e-3 | 97.1 % | 97.5 % | 0.95 / 0.97 |
| 7 + dropout 0.2 | 97.8 % | 98.1 % | 0.97 / 0.98 |
| 7 sin quitar duplicados | 98.1 % | 98.1 % | 0.96 / 0.97 |

### (a) Mejor resultado

- Config del paso 7: `[784, 512, 10]` (407 050 parámetros), ReLU, softmax + cross-entropy, Adam η = 1e-3, mini-batch 16, oversampling, augmentation, L2 1e-4 y early stopping en la época 13.
- **Accuracy en digits_test: 98.04 %** (balanced 98.0 %, macro-F1 0.98). Todas las clases superan 95 % de recall.
- Los peores son el 9 (0.956), el 8 (0.963) y el 5 (0.964).

### (b) Técnicas que mejoraron el rendimiento

1. **Augmentation** es la técnica que más aporta: +1.3 puntos, y es la que cruza el 98 %. Las traslaciones y rotaciones chicas enseñan invariancias que el MLP no tiene por construcción, porque para él cada píxel es una entrada independiente. Con rotaciones grandes, un 6 pasaría a ser un 9.
2. **Balance de clases:** +0.4 puntos. Sube el recall del 5 y del 8, que es justo lo que penaliza un test balanceado.
3. **Early stopping:** solo, sin augmentation, empeora un poco. Corta en ~12 épocas por pérdida de validación, pero es 3 veces más rápido. Con augmentation el sobreajuste llega más tarde, la red entrena ~25 épocas y guarda el mejor modelo, no el último.
4. **Capacidad (512 neuronas):** aporta +0.15 puntos, dentro del ruido entre semillas y al doble de costo.
5. **Lo que no ayudó:**
   - L2 con λ = 1e-4 es neutro, y con λ = 1e-3 empeora (−0.6 puntos): regulariza de más.
   - Dropout 0.2 da lo mismo y tarda el doble.
6. **Quitar duplicados** no cambia el test, pero sin quitarlos la validación sube a 98.1 %. Las imágenes repetidas caen a la vez en train y en validación: es **leakage**, y la validación deja de medir generalización.

### (c) Otros factores además de las técnicas

- **Cambió la distribución de los datos, no solo la cantidad.** Aparece la clase 8 (585 imágenes) y el 5 pasa de 271 a 785. Solo con eso (paso 2) el test salta de 86.8 % a 97.0 %: +10 puntos, más que todas las técnicas juntas.
- **Hay menos datos nuevos de los que parece:** 15 741 filas, de las cuales 3689 ya estaban. El aumento real es de 12 449 a 24 501 imágenes únicas.
- **La validación del Ej. 2 no representaba a producción.** Como no tenía ochos, daba 97.2 % mientras el test daba 85.5 %. En el Ej. 3 la validación sí tiene ochos, y validación y test coinciden dentro de medio punto.
- El test viene de la misma distribución que el train: mismo centrado por centro de masa, misma cantidad de tinta y ningún solapamiento con los otros sets. La mejora no se explica por un cambio de dominio.

## Opcionales de Ej. 2 y 3

### Robustez al ruido (`resultados/digitos/opcionales/robustez_ruido.png`)

Se suma ruido gaussiano N(0, σ²) a digits_test y se recorta a [0,1] (3 semillas de ruido por σ).

| σ | Ej. 2 final | Ej. 3 final | Ej. 3 entrenado con ruido (σ = 0.2 en la augmentation) |
|---|---|---|---|
| 0 | 85.5 % | **98.0 %** | 96.3 % |
| 0.1 | 84.5 % | 97.1 % | 97.5 % |
| 0.2 | 74.7 % | 84.2 % | **97.1 %** |
| 0.3 | 58.6 % | 55.0 % | **92.1 %** |
| 0.5 | 36.7 % | 26.6 % | 67.3 % |

El modelo final **no es robusto**: aguanta σ ≤ 0.1, pero con σ = 0.3 cae a 55 %. Entrenar con ruido lo vuelve robusto hasta σ ≈ 0.3, con un costo de 1.8 puntos sin ruido (el 8 baja a 0.82). Es un trade-off: se elige según cómo sean las planillas reales de CompanyX.

### Interpretabilidad (`resultados/digitos/opcionales/`)

- **Pesos de la primera capa** (`pesos_primera_capa.png`): cada neurona es un detector de **trazos orientados** en una zona de la imagen, con barras horizontales, diagonales y arcos. El rojo marca dónde la tinta la excita y el azul dónde la inhibe. Muchas tienen forma de "borde": un trazo excitador al lado de uno inhibidor.
- **Atribución** (`atribucion.png`): la saliency ∂h/∂x muestra qué píxeles, si se encendieran, cambiarían la salida. Gradiente × entrada muestra qué parte del trazo presente empujó la decisión. Por ejemplo, para el 7 pesa la barra de arriba y para el 0, el contorno del lazo.
- **Errores** (`errores_ej3.png`): son 49 de 2497, dispersos. Las confusiones más frecuentes son 9→4 (4), 5→3, 9→3, 7→9 y 8→6 (3 cada una): pares con trazos parecidos.
- Todo esto se puede explorar en vivo en `pizarra.py`: se dibuja, se ve la atribución, y con click en una neurona se ven sus pesos.
