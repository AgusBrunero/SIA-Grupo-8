# TP5 · Decisiones

Cada decisión lleva su referencia a la cátedra (archivo y diapositiva). El detalle de cada fuente está en `docs/tp5/enunciado_y_catedra_tp5.md`.

## Sesión 0 (2026-11-10)

Tomadas por el grupo a partir de las recomendaciones de la sesión 0. D3 y D4 se ajustaron después de leer el Colab del VAE de la cátedra, que se consiguió más tarde en la misma sesión; el grupo confirmó ese ajuste el 2026-11-10.

| # | Tema | Decisión | Sostén en la cátedra |
|---|---|---|---|
| D1 | "Nueva letra" (1a.4) | Una muestra decodificada que el grupo juzgue letra. Condición mínima: distancia de Hamming > 0 a los 32 patrones. Se muestran todas las candidatas con su distancia; elige el grupo | Autoencoders 33–34 |
| D2 | Arquitectura del DAE (1b) | Libre (el latente no tiene que ser 2D). La red de 1a con latente 2D se evalúa con el mismo protocolo como comparación | Autoencoders 6 y 22 |
| D3 | Pérdida de reconstrucción | **AE y DAE:** ½ Σ (ζ − O)². **VAE:** BCE sumada sobre los píxeles, con salida logística y datos en [0, 1] | AE: Autoencoders 3–4, Clase 11 diap. 37. VAE: Colab de la diap. 90; fórmula de BCE en Autoencoders 51 |
| D4 | Agregación sobre los patrones | **AE y DAE:** suma sobre los patrones (sin 1/p), y en batch se suman los Δw. **VAE:** suma sobre píxeles y sobre el latente, promedio sobre el batch | AE: Clase 11 diap. 37 y 54. VAE: Colab de la diap. 90 |
| D5 | Σ en el VAE | El encoder da μ(x) y s(x) = log σ²(x); z = μ + ϵ ⊙ exp(s/2); ϵ ~ N(0, 1). Ver `DESVIOS.md`, V1 | Colab de la diap. 90; Autoencoders 77, 86, 88–89 |
| D6 | Powell | Se prueba en un escalón de e1a2 con una red chica, midiendo tiempo y evaluaciones de la función | Análisis de Datos diap. 9; NoLineal 35–37 |
| D7 | Interpretación de "1 píxel" | Interpretación A: por patrón, píxeles umbralizados ≠ entrada ≤ 1. Se reportan siempre el máximo por patrón y el total | Enunciado 1a.2; §0.2 del doc de cátedra |
| D8 | Línea base PCA 2D | Se compara el AE 2D contra PCA 2D hecho con la SVD de numpy. Es decisión nuestra: la cátedra relaciona AE y PCA pero no propone la comparación | Autoencoders 7–16 |
| D9 | Adam, momentum, RMSProp | Las versiones de la Clase 12.1 (Adam con corrección de sesgo y ϵ fuera de la raíz; RMSProp con ϵ dentro) | Clase 12.1 diap. 5, 8, 11–13 |

## Sesión 1 (2026-11-10): motor de redes

| # | Tema | Decisión | Sostén en la cátedra |
|---|---|---|---|
| D10 | Base del motor | `src/nn/` parte de `TP3/perceptron/mlp.py` y `optimizadores.py`. Se sacaron ReLU, softmax, dropout, Xavier y weight decay | Solo nombrados o no pedidos en la sesión |
| D11 | Bias | x0 = 1 agregado a la entrada de cada capa; la fila 0 de cada matriz de pesos es w0 | Clase 11 diap. 66–68 |
| D12 | Inicialización | Uniforme en ±`amplitud_inicial` (0,1 por defecto), con semilla; cero da error | Clase 11 diap. 58 |
| D13 | Autoencoder como dos MLP | `MLP.backward` recibe ∂E/∂O y devuelve también ∂E/∂X, para encadenar encoder y decoder | Autoencoders diap. 3 y 86–89 |
| D14 | Cota de la logística | 2βh se acota a ±30 antes de la exponencial. Ver `DESVIOS.md` | Resguardo numérico; no está en la cátedra |
| D15 | η adaptativo | Dos ventanas, k y k′. Después de cambiar η la racha se reinicia (interpretación nuestra de "en forma consistente") | Clase 12.1 diap. 6–7 |
| D16 | Hiperparámetros sin valor de la cátedra | γ y ϵ de RMSProp, y a, b, k, k′ del η adaptativo no tienen valor por defecto en el código: van en el config | Clase 12.1 diap. 6–8 |
| D17 | Powell | `scipy.optimize.minimize(method='Powell')` sobre el vector de pesos de una o más redes; devuelve tiempo y evaluaciones | Análisis de Datos diap. 9; NoLineal 35–37 |
| D18 | Gradient check | Diferencias centradas, paso 1e-5, error relativo ‖a − n‖/(‖a‖ + ‖n‖) < 1e-6 | `CLAUDE.md` §4 |

## Sesión 2 (2026-11-11): autoencoder

| # | Tema | Decisión | Sostén en la cátedra |
|---|---|---|---|
| D19 | Autoencoder como dos MLP | Encoder y decoder son dos `MLP`; el decoder es el espejo: capas invertidas y las mismas activaciones ocultas en orden inverso | Autoencoders diap. 3 |
| D20 | Umbral de la métrica | 0,5 con datos en {0, 1} y 0 con datos en {−1, 1}; salida logística solo con {0, 1} y tanh solo con {−1, 1} (da error si se mezclan) | Clase 10.2 diap. 23 (imágenes); doc de cátedra §0.2 |
| D21 | Error por época | La pérdida con reducción suma sobre los 32 patrones, medida al terminar la época (después de las actualizaciones) | Clase 11 diap. 37 y 52 |
| D22 | "Logró el objetivo" | Se registran la primera época con todos los patrones a ≤ 1 píxel y el estado al final del entrenamiento; una corrida cuenta como lograda según su estado final | Enunciado 1a.2 |

| D23 | BCE en el autoencoder (modifica D3) | Aprobada por el grupo el 2026-11-11 como escalón E9 de e1a2. Era [EXTRA]: problema = píxeles saturados en el valor opuesto que no aprenden con error cuadrático; evidencia = `results/e1a2_piloto/` y `results/e1a2_diagnostico/` | Autoencoders diap. 51; Colab de la diap. 90 |
| D24 | Escalera e1a2 | 10 escalones, 10 semillas (0–9), 10.000 épocas sin cortar al llegar. Se sacó el escalón de minibatch para hacer lugar a BCE | Ver `config/e1a2.json`, campo `sosten` de cada escalón |
| D25 | Hiperparámetros sin valor de la cátedra | Elegidos por menor error final promedio en un piloto con las semillas 100 y 101: GD η = 0,01; momentum η = 0,001; RMSProp η = 0,01 (γ = 0,9 y ϵ = 1e-8 fijados por el grupo); η adaptativo a = 0,002, b = 0,5, k = 5, k′ = 3. Adam con los valores de la diap. 13 | Clase 12.1 diap. 5–13; `config/e1a2.json`, campo `origen_de_los_hiperparametros` |
| D26 | Píxel saturado | Píxel incorrecto con \|O − X\| > 0,99 | Definición nuestra, para medir el hallazgo del piloto |

## Sesión 3 (2026-11-11): latente y generación

| # | Tema | Decisión | Sostén en la cátedra |
|---|---|---|---|
| D27 | Modelo para 1a.3 y 1a.4 | Escalón E9 de e1a2, semillas 0, 1 y 2 (reproducen las corridas de E9) | `results/e1a2/` |
| D28 | PCA | SVD de numpy sobre X centrada; se reconstruye con 2 componentes y se umbraliza en 0,5 | Autoencoders diap. 7–16 (media cero: diap. 14) |
| D29 | Qué se repite entre semillas | Se comparan las 496 distancias entre pares de patrones en el latente (correlación de Pearson) y el vecino más cercano de cada patrón; no dependen de rotación, reflejo ni traslación | Medida nuestra; la cátedra no define una |
| D30 | Generación | Grilla regular de 10 × 10 sobre el rectángulo que cubre los códigos, e interpolación lineal de 9 puntos entre dos códigos. Cada muestra se umbraliza en 0,5 y se anota su distancia de Hamming a la letra más cercana | Autoencoders diap. 33–34 |

## Sesión 4 (2026-11-11): Denoising Autoencoder

| # | Tema | Decisión | Sostén en la cátedra |
|---|---|---|---|
| D31 | Ruido | Salt-and-pepper: cada píxel se invierte con probabilidad p, independiente. Niveles p ∈ {0; 0,05; 0,1; 0,2; 0,3}, fijados por el grupo (resuelve P2) | Autoencoders diap. 22 (nombra el tipo de ruido, no da niveles) |
| D32 | Entrenamiento del DAE | La misma clase `Autoencoder`; en cada época la entrada es X̃ con ruido nuevo (una realización por patrón) y la salida esperada es X | Autoencoders diap. 22 |
| D33 | Arquitecturas | A: 35–20–2–20–35 (la de 1a); B: 35–20–10–20–35; C: 35–50–35. Receta de E9; solo cambia la arquitectura. Aprobadas por el grupo | Autoencoders diap. 6, 21 y 22 |
| D34 | Evaluación | 5 semillas × 20 realizaciones nuevas por patrón en cada celda de la matriz entrenamiento × prueba. Para una semilla y un nivel de prueba, todas las redes ven el mismo ruido | Protocolo del grupo |
| D35 | Comparación con el AE de 1a | Es la fila "p de entrenamiento = 0" de la arquitectura A | — |
| D36 | Confusión entre letras | A cada salida umbralizada se le asigna la letra del dataset más cercana por Hamming (en empates, la primera en el orden de font.h) | Definición nuestra |

## Sesión 5 (2026-11-11): VAE

| # | Tema | Decisión | Sostén en la cátedra |
|---|---|---|---|
| D37 | Dataset del VAE (resuelve P3) | Reemplazado el 2026-11-15 a pedido del grupo (antes: caritas de 7 parámetros). Relojes de agujas construidos en código (`src/data/relojes.py`), 16 × 16 en grises en [0, 1]: esfera fija y la hora, sorteada uniforme en [0, 12) con semilla, fija el ángulo de las dos agujas como en un reloj real. Se probó también variar radio y centro de la esfera: en una corrida de prueba el VAE ordenó el latente por esos corrimientos y dejó las agujas borrosas, así que quedaron fijos. Sin descargas | Enunciado 2a ("elegir o construir"); datos en [0, 1] como el Colab de la diap. 90 |
| D38 | Encoder del VAE | Un MLP cuya última capa tiene 2k neuronas con activación identidad: las primeras k son μ y las últimas k son s = log σ² | Colab de la diap. 90 (dos capas densas lineales sobre la misma capa oculta) |
| D39 | Gradientes del VAE | Decoder: solo reconstrucción. Encoder: δ_μ = ∂L/∂z + λμ y δ_s = ∂L/∂z ⊙ ½ϵ exp(s/2) + λ½(exp(s) − 1), con L y R promediadas sobre el batch | Autoencoders diap. 84–89; doc de cátedra §2.1 |
| D40 | Muestreo de ϵ | `rng.standard_normal` de numpy, un ϵ nuevo por patrón y dimensión en cada actualización | Autoencoders diap. 80–81 ("Pick Random Sample ϵ"); Colab (N(0, 1)). Las diap. 39–42 dan otro método (TCL), que no se usa |

## Sesión 6 (2026-11-11): latente del VAE, generación y elecciones del grupo

| # | Tema | Decisión | Sostén |
|---|---|---|---|
| D41 | AE de comparación (2b) | La arquitectura base del VAE (256–64–2 y decoder espejo, tanh, salida logística) con latente identidad de 2, BCE promediada por batch, el mismo Adam, los mismos lotes y las mismas épocas. Entre el AE y el VAE cambia solo el latente (μ, s, ϵ y λR) | Sesión 6 ("misma arquitectura base"); Autoencoders diap. 3–4 (AE) y 36–37 (el problema de su latente) |
| D42 | Valores de λ | 0,1, 1 y 10, con 5 semillas cada uno | Autoencoders diap. 78, ec. 13 (J = L + λR); la cátedra no da valores. Elección nuestra, propuesta al grupo antes de correr |
| D43 | Generación (2c) | Se decodifican z ~ N(0, I) sorteados con semilla del config, los mismos para todos los modelos. La grilla va de −3 a 3 en z₁ y z₂ (tres desvíos del prior), igual para todos | Autoencoders diap. 34 (especificar z) y 74 (p(z) = N(0, I)); Colab de la diap. 90. La grilla es lineal porque `norm.ppf` no está permitido |
| D44 | Distancia a la imagen más cercana (nuestra) | Media sobre los píxeles de la diferencia absoluta entre la muestra y cada imagen del dataset; se informa la menor. Referencia: la misma distancia entre cada reloj y el otro reloj más cercano | Sesión 6 ("distancia por píxel"); no hay una métrica de la cátedra para esto |
| D45 | Letra nueva que se presenta (1a.4) | Elegida por el grupo el 2026-11-11, sobre una recomendación de Claude: interpolación o→e, semilla 0, paso 3 (`results/e1a3_e1a4/interpolaciones.csv`). Queda en `config/elegidas.json` | Enunciado 1a.4; Autoencoders diap. 33–34 |
| D46 | Relojes generados que se presentan (2c) | Propuestos por Claude el 2026-11-15, falta que el grupo los confirme (antes: caritas 14 y 39, elegidas por el grupo el 2026-11-11): candidatas 14 y 36 del VAE con λ = 1, semilla 0 (`results/e2c/candidatas.csv`), las dos más nítidas de `fig_candidatas_lambda_1.png`. Quedan en `config/elegidas.json` | Enunciado 2c ("que ustedes puedan juzgar") |
| D47 | Semillas de `e1a3_e1a4` | Se subió de 3 a 5 (0 a 4) después de la auditoría de la sesión 7. Las semillas 0, 1 y 2 dan lo mismo que antes; la letra elegida no cambia | CLAUDE.md §4 (mínimo 5 semillas) |

## Pendientes de decisión

| # | Tema | Qué falta |
|---|---|---|
| P1 | η adaptativo | Resuelto en D25: los valores se eligieron con el piloto de la sesión 2 y están en `config/e1a2.json` |

## [EXTRA] no aprobados (no se implementan)

- L-BFGS-B con `jac=True` (no está en ningún PDF).
- Ruido gaussiano o Rayleigh en el DAE (nombrados en Autoencoders 22, sin niveles).
- Early stopping con criterio propio (Clase 13 diap. 14 solo trae el título y un gráfico).
- ReLU, inicialización Xavier o He, Dropout, preentrenamiento greedy por capas (solo nombrados).
- `scipy.stats.norm.ppf` para armar la grilla del latente (lo usa el Colab; no está en la lista de librerías permitidas).
