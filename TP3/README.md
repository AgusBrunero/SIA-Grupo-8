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

Todos los datasets están versionados en `datasets/`. La primera carga de los CSV de dígitos los parsea y los cachea en `datasets/cache/` (ignorada por git), y tarda ~7 s.

`pizarra.py` usa tkinter, que no se instala con pip: en Debian/Ubuntu es `sudo apt install python3-tk`. El resto del TP no lo necesita.

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
aprendizaje.py       Ej. 1: capacidad de aprendizaje, lineal vs sigmoide (todas las muestras)
generalizacion.py    Ej. 1: generalización del sigmoide, k-fold, umbral y modelo final
relu.py              Ej. 1 opcional: ReLU contra lineal y sigmoide
tests_validacion.py  ejercicio de validación (AND, XOR, y=x, tanh, gradient check)
edas/                exploración del dataset de fraude (analisis_dataset.md + fraud_analysis.py)
docs/opcionales_ej1.md  opcionales teóricos del Ej. 1 (features y calibración)
resultados/          salidas de cada script; cada carpeta del Ej. 1 tiene su conclusiones.md
modelos/             modelos finales de Ej. 2 y 3 (.npz con su config)
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
| Cuenta a mano de `[2,2,1]` y `[2,3,2,1]` | Una iteración escalar por escalar (neurona por neurona) coincide con el código matricial |
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
- **Umbral recomendado: 0.80** (máximo F2). En test detecta 164 de 174 fraudes (recall 0.94, precision 0.74) y manda a revisión al 4.3 % de las legítimas.
- Por qué 0.80 y no 0.89 (máximo F1): minimizando `C_FN·FN + C_FP·FP` sobre las predicciones out-of-fold, 0.80 es el umbral de mínimo costo para toda relación C_FN/C_FP entre 2.8 y 7.3. Un fraude cuesta el monto entero (mediana USD 256) y una falsa alarma, una verificación, así que ese es el rango razonable. **0.89** (precision 0.90, recall 0.86) solo conviene si un fraude cuesta menos de ~2.2 falsas alarmas, por ejemplo si las alertas se bloquean automáticamente sin revisión. Entre 2.3 y 2.7 el óptimo es 0.81, prácticamente el mismo que 0.80.
- BigModel usa 0.85 implícitamente. En la pestaña de fraude de la pizarra se ve cómo se mueve el umbral de mínimo costo según la relación FN/FP.
- Respuestas completas de cada punto: [`resultados/aprendizaje/conclusiones.md`](resultados/aprendizaje/conclusiones.md), [`resultados/generalizacion/conclusiones.md`](resultados/generalizacion/conclusiones.md) y [`resultados/relu/conclusiones.md`](resultados/relu/conclusiones.md).

### Opcionales

- **ReLU** ([`relu.py`](relu.py)): se comporta como el lineal truncado en 0. Tiene el mismo underfitting, salidas mayores a 1 y un 1.9 % de muestras con h ≤ 0, que no aportan gradiente. Rankea bien (AUC 0.993), pero no sirve como probabilidad.
- **Features y calibración** (teóricos): [`docs/opcionales_ej1.md`](docs/opcionales_ej1.md). Hallazgo: TinyModel es fiel a BigModel, pero **BigModel no está calibrado**. Entre las transacciones con p ≈ 0.55 no hubo ni un fraude (ECE 0.31 contra `flagged_fraud`).

## Ejercicio 2 — Dígitos con MLP (`digits.csv`)

- `digits.csv` se divide en 80 % train y 20 % validación, estratificado y con una partición fija para comparar.
- `digits_test.csv` es "producción": **no interviene en ninguna decisión**. Todo se elige con validación. Los barridos también registran la accuracy en test (`test_acc_*` en los CSV resumen), pero solo para mostrar después cómo se tradujo cada decisión en producción; ninguna elección la mira.
- Cada variante se corre con 3 semillas (media ± desvío) y se cambia una variable por vez sobre la base de la cátedra: `[784,64,10]`, tanh oculta, logística a la salida, MSE, Adam η=1e-3, mini-batch 64, 40 épocas.
- Tabla completa en `resultados/digitos/ej2/resumen_barridos.csv`.

### (a) ¿Cómo evalúo el desempeño?

- **Accuracy** global, porque el test está balanceado.
- **Matriz de confusión 10×10.**
- **Precision, recall y F1 por clase**, porque la accuracy global esconde qué clase falla.
- **Curvas de pérdida y accuracy de train contra validación por época**, para detectar under y overfitting.
- **Tiempo de entrenamiento**, que también es parte del costo.

### (b) Variantes

| Factor | Resultado (accuracy máxima en validación, media de 3 semillas) |
|---|---|
| **Tasa de aprendizaje** | Cada optimizador tiene su ventana: GD 0.5 → 95.8 %, Momentum 0.05 → 95.8 %, RMSProp 1e-3 → 96.1 %, Adam 1e-3 → 95.8 %. Con η demasiado grande **todos divergen**: entre 5 % y 17 %, el nivel del azar. Con η chico convergen lento: ~94 % en 40 épocas. η adaptativo sobre GD da lo mismo que el mejor η fijo (95.8 %) |
| **Optimizador** | A su mejor η, todos terminan en ~96 % de validación. RMSProp llega a 94 % en la época 4, Adam en la 6, GD en la 8 y Momentum en la 9 (medias de 3 semillas). RMSProp y Adam bajan la pérdida de train 2–3 veces más (0.003–0.005 contra ~0.010), pero eso no se traduce en validación: la diferencia pasa a ser sobreajuste |
| **Arquitectura** | Más neuronas ayudan hasta ~256 (16 → 93.9 %, 64 → 95.8 %, 256 → 96.3 %), con rendimiento decreciente y 8 veces más tiempo de entrenamiento. Tres capas ocultas `[256,128,64]` → 96.9 % |
| Activación oculta | ReLU 96.3 % > tanh 95.8 % ≈ logística 95.8 % |
| Tamaño de batch | Online (1) 96.1 %, pero **19 veces más lento** que mini-batch 64. Mini-batch 16 → 96.2 %, 64 → 95.8 %, 256 → 95.4 %. **Batch completo → 83.5 %**: hace una sola actualización por época, así que en 40 épocas no llega a converger |
| Pérdida | Softmax + cross-entropy 96.1 % contra logística + MSE 95.8 %. La pérdida de validación toca su mínimo en ~17 épocas, contra 38 |
| β, inicialización | Efecto chico: β = 2 da 95.9 % y β = 0.5 da 95.7 %. Xavier y uniforme chica quedan iguales |
| **Combinación** | ReLU + softmax/CE + `[784,256,10]` + mini-batch 16 + Adam 1e-3 → **97.0 ± 0.2 %**. Empata con ReLU + softmax/CE + `[256,128,64]` (97.1 ± 0.1 %): la diferencia está dentro del desvío entre semillas. Se elige la de una capa oculta porque tiene menos parámetros (204 mil contra 243 mil) y es más simple |

Los tiempos se midieron con 11 corridas en paralelo, así que solo sirven para comparar entre sí.

### Modelo final y resultado en "producción"

- Es la combinación ganadora: 203 530 parámetros. El early stopping mira la **accuracy de validación** (paciencia 8), el mismo criterio con el que se compararon las variantes, y se queda con la mejor época (la 10, con 96.8 % en validación).
  - Si se corta por pérdida de validación, corta en la época 4 (96.5 %). Con cross-entropy la pérdida sube temprano, cuando la red se vuelve demasiado segura, aunque la accuracy siga mejorando unas épocas más.
- **Accuracy en digits_test: 86.7 %** (balanced 86.4 %). Si se sacan los ochos del test, la accuracy sobre el resto es 96.1 %, en línea con la validación.
- La matriz de confusión muestra que **el 8 tiene recall 0**: no hay ningún 8 en `digits.csv`. De los 243 ochos del test, 95 se predicen como 3, 50 como 5 y 41 como 2.
- El 5, con apenas 271 imágenes en `digits.csv` (~217 en el train), tiene recall 0.86, el más bajo después del 8.
- **El techo lo pone el dato, no el modelo:** aunque acertara todo lo demás, sin ochos el máximo es 90.3 %. Ningún hiperparámetro lo mueve.
- La validación sale de `digits.csv`, así que tampoco tiene ochos: por eso da ~97 % mientras producción da 86.7 %. No representa al "mundo real", y ese es el hallazgo principal del Ej. 2.

## Ejercicio 3 — ≥ 98 % con `more_digits.csv`

- Se combinan `digits` y `more_digits` **sin duplicados**: 3689 imágenes de `more_digits` ya estaban en `digits`, así que quedan 24 501 únicas.
- Se separa un 15 % de validación estratificada.
- **Se elige por balanced accuracy en validación.** El test está balanceado y el train no: el 5 y el 8 siguen siendo minoría. Por eso la accuracy común sobreestima. El early stopping mira la misma métrica y restaura la mejor época.
- Se hace una ablación **acumulativa**: cada paso suma una técnica al anterior, con 3 semillas por paso. La columna de test se muestra solo como referencia de "producción" y no se usa para elegir.

| Paso | Validación (balanced) | digits_test | Recall 5 / 8 en test |
|---|---|---|---|
| 1. Config final del Ej. 2, entrenada solo con las imágenes de `digits` | 86.4 % | 86.7 % | 0.88 / **0.00** |
| 2. + `more_digits` sin duplicados | 96.6 % | 97.0 % | 0.92 / 0.92 |
| 3. + early stopping (paciencia 6) | 96.5 % | 97.0 % | 0.94 / 0.94 |
| 4. + balance de clases (oversampling de 5 y 8) | 96.6 % | 97.1 % | 0.96 / 0.94 |
| **5. + augmentation (traslación ±2 px, rotación ±10°, solo en train)** | **97.9 %** | **98.3 %** | 0.97 / 0.97 |
| 6. + L2 (λ = 1e-4) | 97.7 % | 98.2 % | 0.97 / 0.97 |
| 7. + capa oculta de 512 | 97.6 % | 97.7 % | 0.96 / 0.97 |
| *Extra:* 7 con L2 λ = 1e-3 (1 semilla) | 96.5 % | 97.1 % | 0.95 / 0.93 |
| *Extra:* 7 sin quitar duplicados (1 semilla) | 98.2 % | 98.0 % | 0.97 / 0.98 |

Validación = máximo por época de la balanced accuracy, media de 3 semillas (los extras, 1 semilla: solo muestran efectos más grandes que el ruido entre semillas, de 0.1–0.3 puntos). Desde el paso 3 coincide con el modelo que se evalúa, porque el early stopping restaura esa época. En los pasos 1 y 2 se evalúa la última época.

### (a) Mejor resultado

- Se elige el **paso 5**, que tiene la mejor balanced accuracy en validación (97.9 %). Los pasos 6 y 7 agregan L2 y más capacidad, y no la mejoran.
- Config: `[784, 256, 10]` (203 530 parámetros), ReLU, softmax + cross-entropy, Adam η = 1e-3, mini-batch 16, oversampling, augmentation y early stopping (mejor época: 19).
- **Accuracy en digits_test: 98.48 %** (balanced 98.5 %, macro-F1 0.98). Todas las clases superan 97 % de recall.
- Los peores son el 6 (0.971), el 9 (0.972) y el 8 (0.979).

### (b) Técnicas que mejoraron el rendimiento

1. **Augmentation** es la técnica que más aporta: +1.3 puntos en validación (96.6 % → 97.9 %), y es la que cruza el 98 % en test. Las traslaciones y rotaciones chicas enseñan invariancias que el MLP no tiene por construcción, porque para él cada píxel es una entrada independiente. Con rotaciones grandes, un 6 pasaría a ser un 9.
2. **Balance de clases:** en validación el efecto está dentro del ruido (+0.2 puntos de balanced accuracy, con un desvío de 0.1–0.3 entre semillas), y el recall de 5 y 8 no cambia. Se mantiene porque no cuesta nada y apunta al problema correcto: la validación y producción pesan igual a todas las clases, y el train no.
3. **Early stopping:** solo, sin augmentation, no cambia el resultado (96.6 % → 96.5 %), pero entrena 2.5 veces más rápido porque corta en ~16 épocas en vez de 40. Con augmentation el sobreajuste llega más tarde, la red entrena ~25 épocas y se queda con la mejor, no con la última.
4. **Lo que no ayudó:**
   - L2 con λ = 1e-4 queda igual o un poco peor (97.7 %), y con λ = 1e-3 empeora: regulariza de más.
   - Más capacidad (512 neuronas) no mejora (97.6 %): duplica los parámetros y tarda 1.5 veces más.
5. **Quitar duplicados** es necesario para que la validación sea honesta. Sin quitarlos, la validación sube a 98.2 %, la más alta de la tabla, mientras que el test (98.0 %) queda igual que el del paso 7. Las imágenes repetidas caen a la vez en train y en validación: es **leakage**, y la validación deja de medir generalización.

### (c) Otros factores además de las técnicas

- **Cambió la distribución de los datos, no solo la cantidad.** Aparece la clase 8 (585 imágenes) y el 5 pasa de 271 a 785. Solo con eso (paso 2) el test salta de 86.7 % a 97.0 %: +10 puntos. Todas las técnicas juntas suman +1.4 (de 97.0 % a 98.3 %, media de 3 semillas).
- **Hay menos datos nuevos de los que parece:** 15 741 filas, de las cuales 3689 ya estaban. El aumento real es de 12 449 a 24 501 imágenes únicas.
- **La validación del Ej. 2 no representaba a producción.** Como no tenía ochos, daba ~97 % mientras el test daba 86.7 %. En el Ej. 3 la validación sí tiene ochos, y en el modelo final coinciden dentro de medio punto (98.0 % de validación contra 98.5 % de test).
- El test viene de la misma distribución que el train: mismo centrado por centro de masa, misma cantidad de tinta y ningún solapamiento con los otros sets. La mejora no se explica por un cambio de dominio.

## Opcionales de Ej. 2 y 3

### Robustez al ruido (`resultados/digitos/opcionales/robustez_ruido.png`)

Se suma ruido gaussiano N(0, σ²) a digits_test y se recorta a [0,1] (3 semillas de ruido por σ).

| σ | Ej. 2 final | Ej. 3 final | Ej. 3 entrenado con ruido (σ = 0.2 en la augmentation) |
|---|---|---|---|
| 0 | 86.7 % | **98.5 %** | 96.5 % |
| 0.1 | 86.1 % | 94.0 % | **96.6 %** |
| 0.2 | 76.4 % | 66.1 % | **96.2 %** |
| 0.3 | 63.3 % | 40.9 % | **92.9 %** |
| 0.5 | 39.2 % | 20.7 % | 70.7 % |

El modelo final **no es robusto**: con σ = 0.1 pierde 4.5 puntos y con σ = 0.2 cae a 66 %. Hasta es menos robusto que el del Ej. 2, que es menos preciso pero se degrada más despacio. Entrenar con ruido lo vuelve robusto hasta σ ≈ 0.3, con un costo de 2 puntos sin ruido (el 5 y el 9 bajan a ~0.92). Es un trade-off: se elige según cómo sean las planillas reales de CompanyX.

### Interpretabilidad (`resultados/digitos/opcionales/`)

- **Pesos de la primera capa** (`pesos_primera_capa.png`): cada neurona es un detector de **trazos orientados** en una zona de la imagen, con barras horizontales, diagonales y arcos. El rojo marca dónde la tinta la excita y el azul dónde la inhibe. La mayoría tiene forma de "borde": un trazo inhibidor con una franja excitadora al costado.
- **Atribución** (`atribucion.png`): la saliency ∂h/∂x muestra qué píxeles, si se encendieran, cambiarían la salida. Gradiente × entrada muestra qué parte del trazo presente empujó la decisión. Por ejemplo, para el 7 pesa la barra de arriba y para el 0, el contorno del lazo.
- **Errores** (`errores_ej3.png`): son 38 de 2497, dispersos. La confusión más frecuente es 9→7 (4 veces); le siguen 9→4, 6→2, 6→0, 3→5 y 2→7 (2 cada una): pares con trazos parecidos.
- Todo esto se puede explorar en vivo en `pizarra.py`: se dibuja, se ve la atribución, y con click en una neurona se ven sus pesos.
