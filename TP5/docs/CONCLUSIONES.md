# TP5 · Conclusiones

Borrador redactado por Claude el 2026-11-11 a pedido del grupo. **El grupo lo tiene que leer, corregir y hacer propio antes de presentarlo:** cada conclusión hay que poder defenderla sin leerla.

Cada conclusión trae su evidencia (números copiados de `docs/RESULTADOS.md`, que se genera desde los CSV), la parte de la materia que la explica y su límite. Regla usada: se dice que algo es mejor solo si la diferencia supera el desvío entre semillas; si no, se dice que empatan.

## 1a.1 · Arquitectura del autoencoder

**Conclusión.** La arquitectura que proponemos es 35–20–2–20–35, con capas ocultas tanh, latente lineal (identidad) y salida logística. De los tres cambios de arquitectura, el que más pesa es dejar el latente lineal.

**Evidencia.**
- Píxeles incorrectos en total sobre los 32 patrones: base 35–2–35 (E1) 210,0 ± 11,0; con una capa oculta de 20 (E2) 91,4 ± 41,8; con latente identidad (E3) 11,7 ± 3,9; con ocultas tanh (E4) 10,0 ± 2,1.
- Entre E3 y E4 la cantidad de píxeles empata (la diferencia es menor que el desvío), pero solo E4 tiene semillas que cumplen el objetivo: 4 de 10.

**Teoría.** Autoencoders diap. 3–4: dos MLP en espejo y Z = f(X). Con una logística en el latente los códigos quedan encerrados entre 0 y 1 en cada coordenada; con la identidad pueden separarse todo lo que haga falta. Sin capa oculta, el encoder es una sola capa y no puede separar 32 patrones en 2 dimensiones (Clase 10.2 y Clase 11: hace falta una capa oculta no lineal).

**Límite.** La arquitectura sola no alcanza el objetivo de 1 píxel; eso se resuelve en 1a.2.

## 1a.2 · Arquitecturas y técnicas de optimización

**Conclusión.** Se aprende el dataset completo, los 32 patrones, con 0 píxeles incorrectos. Lo que destrabó el objetivo fue cambiar la función de costo a entropía cruzada (BCE); ni el optimizador ni agrandar la red lo lograron.

**Evidencia.**
- Con error cuadrático y el mismo modelo, cambiando solo el optimizador (GD, momentum, η adaptativo, RMSProp, Adam), el total de píxeles incorrectos va de 10,0 ± 2,1 a 18,4 ± 3,5 y ninguno pasa de 4 de 10 semillas.
- Con BCE y Adam (E9): 10 de 10 semillas, con 0,0 ± 0,0 píxeles incorrectos.
- De E4 a E8, 709 de 711 píxeles incorrectos tienen la salida pegada al valor opuesto (|O − X| > 0,99).
- Adam con error cuadrático (E8, 18,4 ± 3,5) queda peor que GD (E4, 10,0 ± 2,1), y la diferencia supera el desvío.
- Powell (E10): 0 de 10, con error final 108,45 ± 6,84 contra 77,93 ± 1,81 de GD sobre la misma red.

**Teoría.** Con error cuadrático el delta de la capa de salida es (ζ − O)·θ′(h) (Clase 11, diap. 40–43). Si una neurona logística queda saturada en el valor equivocado, θ′(h) vale casi 0 y ese píxel deja de corregirse aunque el error sea máximo. Con BCE y salida logística el delta queda proporcional a (O − ζ), sin el factor θ′, así que el píxel más equivocado es el que más se corrige. Los optimizadores de la Clase 12.1 cambian cómo se usa el gradiente, pero no pueden arreglar un gradiente que vale casi 0. Powell no usa el gradiente y con 177 pesos se detiene lejos del mínimo.

**Límite.** La pérdida del AE en las diapositivas es ‖X − X′‖²; la BCE es de la cátedra (Autoencoders diap. 51 y Colab del VAE) pero no es la que la diapositiva del AE usa, y lo declaramos. Los pilotos y el diagnóstico se hicieron con 2 semillas y solo sirvieron para decidir; el resultado es la escalera de 10 semillas.

## 1a.3 · Espacio latente en dos dimensiones

**Conclusión.** El latente de 2 dimensiones alcanza para reconstruir las 32 letras sin errores, cosa que una proyección lineal a 2 dimensiones no logra. Pero la disposición de las letras en el latente cambia con la semilla y refleja solo en parte el parecido entre letras.

**Evidencia.**
- PCA con 2 componentes explica el 27,0 % de la varianza y reconstruye con 230 píxeles incorrectos en total; el autoencoder, con 0.
- Correlación entre semillas de las distancias entre pares de letras en el latente: entre 0,14 y 0,70 según el par de semillas (5 semillas, 10 pares).
- Correlación entre la distancia en el latente y la distancia de Hamming: entre 0,33 y 0,54 según la semilla.

**Teoría.** El autoencoder con capas no lineales es una reducción de dimensión no lineal; PCA es la lineal (Autoencoders diap. 3–4). Nada en la función de costo del AE le pide al latente una forma en particular, así que cada entrenamiento encuentra una distinta.

**Límite.** Las correlaciones son medidas elegidas por nosotros para comparar latentes; la cátedra pide el gráfico, no una métrica.

## 1a.4 · Generar una letra nueva

**Conclusión.** El decoder genera letras que no están en el conjunto de entrenamiento cuando se le da un z que no es el código de ninguna letra. La que mostramos es el punto intermedio entre la "o" y la "e": una "o" con media barra, a 2 píxeles de la "o". El latente del AE tiene zonas que decodifican bien y zonas que no, y eso es lo que motiva el VAE.

**Evidencia.**
- En una grilla de 100 puntos del latente, entre 60 y 73 muestras según la semilla (5 semillas) no son idénticas a ninguna letra del dataset.
- Interpolación o→e, semilla 0: 0·o → 0·o → 0·o → 2·o → 0·e. La elegida es el paso a 2 píxeles (`results/elegidas/fig_letra.png`).
- Interpolación k→q: el camino pasa por letras que no tienen que ver (r, p, y) y por muestras a 5 y 6 píxeles de cualquier letra.

**Teoría.** Autoencoders diap. 33–34: moverse en el latente y generar especificando z directamente. Diap. 36–37: el AE no garantiza que un punto cualquiera del latente decodifique en algo válido.

**Límite.** Que una muestra "sea una letra" es un juicio nuestro; lo que medimos es que no es copia.

## 1b.1 · Arquitectura del Denoising Autoencoder

**Conclusión.** Elegimos 35–20–10–20–35 (B): un cuello de botella intermedio. El latente de 2 es demasiado angosto para sacar ruido y la red sin cuello de botella (35–50–35) saca menos ruido.

**Evidencia.** Píxeles incorrectos a la salida, con el mejor nivel de entrenamiento de cada arquitectura (A: 35–20–2–20–35, C: 35–50–35):
- prueba con p = 0,05: B 0,15 ± 0,03; C 0,65 ± 0,04; A 0,82 ± 0,08;
- prueba con p = 0,1: B 0,68 ± 0,12; C 1,74 ± 0,04; A 2,29 ± 0,19;
- prueba con p = 0,2: B 3,18 ± 0,27; C 4,40 ± 0,06; A 6,14 ± 0,24;
- prueba con p = 0,3: B 6,76 ± 0,12; C 7,31 ± 0,04; A 8,44 ± 0,23.

En los cuatro niveles la diferencia a favor de B supera el desvío. Además, A entrenada con p = 0,1 ya no reconstruye las letras limpias (47,6 ± 9,8 píxeles incorrectos).

**Teoría.** Autoencoders diap. 22: el DAE recibe X̃ y tiene que devolver X, así que necesita capacidad para representar la letra y, a la vez, una restricción que lo obligue a quedarse con lo que la define y descartar el ruido. Con 2 dimensiones no entra la letra más la tolerancia al ruido; con 50 neuronas no hay restricción.

**Límite.** Probamos tres arquitecturas; no afirmamos que 10 sea el mejor tamaño de latente, solo que es el mejor de los tres.

## 1b.2 · Niveles de ruido

**Conclusión.** Un autoencoder entrenado sin ruido no saca ruido: lo empeora. Entrenado con ruido, saca la mayor parte cuando el ruido es bajo y su capacidad cae a medida que el ruido crece. Conviene entrenar con un nivel moderado; entrenar con demasiado ruido arruina también las letras limpias.

**Evidencia** (ruido salt-and-pepper; con p = 0,1 se alteran 3,50 píxeles por letra en promedio).
- El AE de 1a, probado con p = 0,05: entran 1,75 píxeles alterados y salen 6,97 ± 0,25 incorrectos.
- B entrenada con p = 0,1 y probada con p = 0,1: entran 3,50 y salen 0,68 ± 0,12; el 85,2 ± 1,9 % de las reconstrucciones queda con 1 píxel incorrecto o menos.
- B probada con p = 0,3: entran 10,50 y salen 6,76 ± 0,12; solo el 15,4 ± 1,6 % queda con 1 píxel o menos.
- B entrenada con p = 0,3 y probada sin ruido: 2,23 ± 0,10 píxeles incorrectos por letra.
- Las confusiones son entre letras parecidas: la más frecuente en B es '|' → 'l' (25 de 100), dos letras que difieren en 2 píxeles.

**Teoría.** Autoencoders diap. 22: se entrena con la entrada corrompida y la salida esperada limpia. Con mucho ruido la entrada corrompida de una letra queda más cerca de otra letra que de la original, y ahí ningún modelo puede recuperarla.

**Límite.** Un solo tipo de ruido (salt-and-pepper); el gaussiano no lo implementamos.

## 2a · Conjunto de datos nuevo

**Conclusión.** Construimos el dataset en código: 500 caritas de 16 × 16 en grises, definidas por 7 parámetros continuos. Lo construimos nosotros para conocer de qué depende cada imagen y poder juzgar si una muestra generada pertenece al conjunto.

**Evidencia.** 500 caritas distintas de 500; el 44,1 % de los píxeles es gris.

**Teoría.** Datos en [0, 1] con salida logística y BCE, como el Colab del VAE de la cátedra (Autoencoders diap. 90).

**Límite.** Es un dataset sintético y chico; las caritas se parecen mucho entre sí por construcción.

## 2b · Esquema variacional y el problema del latente

**Conclusión.** El VAE resuelve el problema del latente: sus códigos ocupan la zona de la normal estándar, que es de donde después se samplea para generar. El AE con la misma arquitectura deja los códigos desparramados en una zona que depende de la semilla. El precio es una reconstrucción un poco peor, y λ regula ese intercambio.

**Evidencia** (misma arquitectura, mismos datos, 5 semillas).
- Desvío de los códigos: AE 9,55 ± 1,20; VAE con λ = 1: 1,02 ± 0,00. El prior tiene desvío 1.
- Códigos a menos de 3 del origen: AE 0,19 ± 0,05; VAE con λ = 1: 1,00 ± 0,00.
- Reconstrucción L: AE 63,97 ± 0,07; VAE con λ = 1: 65,65 ± 0,04. La diferencia supera el desvío: el VAE reconstruye peor.
- Efecto de λ: con λ = 0,1, R = 6,82 ± 0,16 y L = 64,66 ± 0,11; con λ = 10, R = 0,73 ± 0,04 y L = 72,40 ± 0,21, con σ media 0,696 ± 0,010 y caritas generadas casi iguales entre sí.
- El VAE aprende: L baja de 174,33 ± 1,02 a 66,66 ± 0,07, con un piso de 58,55 para estos datos.

**Teoría.** Autoencoders diap. 74–78: se supone p(z) = N(0, I) y se agrega a la reconstrucción el término R = KL(q‖p), que penaliza que la distribución de cada código se aleje de la normal estándar; J = L + λR (ec. 13). Diap. 80–89: el truco de reparametrización z = μ + σ·ϵ permite hacer backpropagation a través del muestreo. Con λ grande gana R: los códigos se pegan al prior, σ crece y el decoder recibe casi lo mismo para todos los datos, por eso las salidas se parecen. Con λ chico gana L y el modelo se acerca a un AE.

**Límite.** En la distancia de una muestra generada a la imagen más cercana del dataset, el AE (0,056 ± 0,020) y el VAE con λ = 1 (0,037 ± 0,000) empatan según nuestra regla: la diferencia no supera el desvío del AE. La diferencia que sí medimos está en dónde quedan los códigos y en que el AE cambia mucho de una semilla a otra. Con 2 dimensiones de latente para 7 parámetros, la boca sale borrosa.

## 2c · Generar una muestra nueva

**Conclusión.** Decodificando z sorteados de N(0, I) con el VAE (λ = 1) se obtienen caritas nuevas que juzgamos del conjunto: tienen contorno, ojos y boca, y están tan cerca del dataset como las caritas reales entre sí, sin ser copia de ninguna. Mostramos las candidatas 14 y 39.

**Evidencia.**
- Distancia de una generada a la imagen más cercana del dataset: 0,037 ± 0,000. Entre caritas reales, la distancia a la otra más cercana es 0,035 en promedio.
- Ninguna de las 4000 muestras generadas es idéntica a una imagen del dataset.
- Las elegidas, con su vecina del dataset: `results/elegidas/fig_caritas.png` y `elegidas.csv`.

**Teoría.** Autoencoders diap. 34 y 74: se descarta el encoder y se decodifica un z sorteado del prior; funciona porque el entrenamiento llevó los códigos a esa zona (2b).

**Límite.** "Pertenece al conjunto" es un juicio nuestro, como pide la consigna; la distancia por píxel es una métrica propia, no de la cátedra.
