# TP5 · Conclusiones

Borrador redactado por Claude el 2026-11-11 a pedido del grupo. **El grupo lo tiene que leer, corregir y hacer propio antes de presentarlo:** cada conclusión hay que poder defenderla sin leerla.

Cada conclusión trae su evidencia (números copiados de `docs/RESULTADOS.md`, que se genera desde los CSV), la parte de la materia que la explica y su límite. Regla usada: se dice que algo es mejor solo si la diferencia supera el desvío entre semillas; si no, se dice que empatan.

## 1a.1 · Arquitectura del autoencoder

Rehecha el 2026-11-17 por pedido de la cátedra: el latente no puede ser lineal, y hay que variar la cantidad de capas, sus dimensiones y sus activaciones (tanh o logística). Ya no usamos el latente identidad de la versión anterior (E9); la escalera E1–E10 queda como historia de cómo llegamos a la entropía cruzada (1a.2).

**Conclusión.** Para el AE elegimos **35–30–20–2–20–30–35 (T8)**: dos capas ocultas tanh por lado, **latente tanh** y salida logística, con entropía cruzada y Adam (η = 0,01). De las once redes con latente no lineal que probamos, cinco aprenden las 32 letras sin errores en las 10 semillas; entre esas cinco, T8 es la que llega antes, y la diferencia supera el desvío. Más capas no ayudan por sí solas: con tres capas ocultas la red llega en la mitad de las semillas o menos. Con una sola capa importa el ancho: 10 neuronas no alcanzan, 30 sí. Las ocultas tanh andan mejor que las logísticas.

**Evidencia** (`results/e1a2_arquitecturas/`, 10 semillas por celda, mejor η de cada red entre 0,0003, 0,001, 0,003 y 0,01; 0,03 en las que quedaban en el borde).
- Con entropía cruzada llegan en 10 de 10 semillas, con 0,0 ± 0,0 píxeles mal: 35–30–20–2 (T8), 35–30–2 (T6), 35–30–15–2 (T3), 35–20–2 (T1) y 35–20–2 con ocultas logísticas (G1).
- Época en que llegan (mediana; media ± desvío): T8 376 (380 ± 56), T6 942,5 (1116 ± 487), T3 817,5 (861 ± 185), T1 2070,5 (2078 ± 318), G1 2664,5 (2734 ± 435). T8 llega antes que todas, y la diferencia supera el desvío. T6 y T3 empatan (la diferencia es menor que el desvío); T6 queda segunda porque tiene menos pesos (2317 contra 3187). Esas dos primeras son las que usamos en el DAE (1b).
- Las que no llegan en todas: 35–20–10–2 (T2) 8 de 10, la misma con latente logístico (L2) 8 de 10, 35–30–20–10–2 (T4) 5 de 10, 35–30–30–15–2 (T9) 4 de 10, 35–10–2 (T5) 4 de 10, 35–30–15–2 con ocultas logísticas (G3) 3 de 10.
- Activación de las ocultas, misma red: 35–20–2 llega en la época 2070,5 con tanh y en la 2664,5 con logística; 35–30–15–2 llega en 10 de 10 con tanh y en 3 de 10 con logística.
- Con error cuadrático llegan 1 de 440 corridas.
- Con una sola capa oculta, pasar el latente de identidad a tanh no cambia los píxeles (E9: 10 de 10 semillas, 0,0 ± 0,0 px; T1: 10 de 10, 0,0 ± 0,0) pero, con el mismo η = 0,001, llega más tarde: época 2205 (E9) contra 4892,5 (T1).

**Teoría.** Autoencoders diap. 3–6: encoder y decoder son dos MLP en espejo y pueden tener varias capas ocultas. Con tanh en el latente (Clase 10.2 diap. 23) los códigos quedan en el cuadrado (−1, 1)² y no hay ninguna capa lineal. Con entropía cruzada y salida logística el delta de salida es proporcional a (O − ζ), sin el factor θ′ que frena al error cuadrático (1a.2). Que las redes de tres capas tanh por lado aprendan peor, y que la logística en las ocultas frene a la red de dos capas, es consistente con el backprop de la Clase 11 diap. 40–43: el delta de cada capa se multiplica por θ′(h), que vale a lo sumo β con tanh y a lo sumo β/2 con logística (Clase 10.2), así que con más capas o con logística hay más factores chicos.

**Límite.** La cátedra no da una regla para el tamaño de las capas: el orden entre redes es empírico y vale para estas once, no demuestra que 30–20 sea el óptimo. Con η = 0,03 T8 baja a 9 de 10 semillas (25,3 ± 80,0 px), así que 0,01 no está en el borde de lo que funciona. La elección cambió de T3 a T8 cuando agregamos η = 0,01 a la grilla (D62).

## 1a.2 · Arquitecturas y técnicas de optimización

**Conclusión.** Se aprende el dataset completo, los 32 patrones, con 0 píxeles incorrectos. Lo que destrabó el objetivo fue cambiar la función de costo a entropía cruzada (BCE); ni el optimizador ni agrandar la red lo lograron.

**Evidencia.**
- Con error cuadrático y el mismo modelo, cambiando solo el optimizador (GD, momentum, η adaptativo, RMSProp, Adam), el total de píxeles incorrectos va de 10,0 ± 2,1 a 18,4 ± 3,5 y ninguno pasa de 4 de 10 semillas.
- Con BCE y Adam (E9): 10 de 10 semillas, con 0,0 ± 0,0 píxeles incorrectos.
- De E4 a E8, 709 de 711 píxeles incorrectos tienen la salida pegada al valor opuesto (|O − X| > 0,99).
- Adam con error cuadrático (E8, 18,4 ± 3,5) queda peor que GD (E4, 10,0 ± 2,1), y la diferencia supera el desvío.
- Powell (E10): 0 de 10, con error final 108,45 ± 6,84 contra 77,93 ± 1,81 de GD sobre la misma red.

**Teoría.** Con error cuadrático el delta de la capa de salida es (ζ − O)·θ′(h) (Clase 11, diap. 40–43). Si una neurona logística queda saturada en el valor equivocado, θ′(h) vale casi 0 y ese píxel deja de corregirse aunque el error sea máximo. Con BCE y salida logística el delta queda proporcional a (O − ζ), sin el factor θ′, así que el píxel más equivocado es el que más se corrige. Los optimizadores de la Clase 12.1 cambian cómo se usa el gradiente, pero no pueden arreglar un gradiente que vale casi 0. Powell no usa el gradiente y con 177 pesos se detiene lejos del mínimo.

**Con la red nueva (latente tanh, D60).** Repetimos la comparación de optimizadores sobre la red elegida, 35–30–20–2–20–30–35 (`results/e1a2_opt_tanh/`, 10 semillas, mejor η de cada uno entre tres y cinco). La conclusión se mantiene: con error cuadrático ningún optimizador pasa de 2 de 10 semillas (RMSProp; η adaptativo 1 de 10, el resto 0 de 10), y con entropía cruzada llegan Adam y RMSProp en 10 de 10 con 0,0 ± 0,0 píxeles mal. Con entropía cruzada, momentum llega en 5 de 10 (η = 0,0001), descenso por gradiente en 0 de 10 (69,9 ± 47,2 píxeles mal) y η adaptativo en 0 de 10 (375,0 ± 20,4).

**Límite.** La pérdida del AE en las diapositivas es ‖X − X′‖²; la BCE es de la cátedra (Autoencoders diap. 51 y Colab del VAE) pero no es la que la diapositiva del AE usa, y lo declaramos. Los pilotos y el diagnóstico se hicieron con 2 semillas y solo sirvieron para decidir; el resultado es la escalera de 10 semillas. Con la red nueva, donde el mejor η quedó en el borde de la grilla se probó uno más (y uno más para GD y momentum con entropía cruzada); descenso por gradiente con entropía cruzada sigue en el borde (η = 0,0003) y ahí cortamos.

## 1a.3 · Espacio latente en dos dimensiones

**Conclusión.** El latente de 2 dimensiones alcanza para reconstruir las 32 letras sin errores, cosa que una proyección lineal a 2 dimensiones no logra. Pero la disposición de las letras en el latente cambia con la semilla y refleja solo en parte el parecido entre letras.

**Evidencia.**
- PCA con 2 componentes explica el 27,0 % de la varianza y reconstruye con 230 píxeles incorrectos en total; el autoencoder, con 0.
- Modelo: el elegido en 1a.1 (35–30–20–2–20–30–35, latente tanh). Las 5 de 5 semillas reconstruyen con 0 px incorrectos.
- Correlación entre semillas de las distancias entre pares de letras en el latente: entre 0,36 y 0,76 según el par de semillas (5 semillas, 10 pares).
- Correlación entre la distancia en el latente y la distancia de Hamming: entre 0,45 y 0,59 según la semilla.
- Con latente tanh los códigos quedan dentro del cuadrado (−1, 1)² y muchas letras se apoyan en sus bordes (`results/e1a3_e1a4/fig_latente_ae.png`).

**Teoría.** El autoencoder con capas no lineales es una reducción de dimensión no lineal; PCA es la lineal (Autoencoders diap. 3–4). Nada en la función de costo del AE le pide al latente una forma en particular, así que cada entrenamiento encuentra una distinta.

**Límite.** Las correlaciones son medidas elegidas por nosotros para comparar latentes; la cátedra pide el gráfico, no una métrica.

## 1a.4 · Generar una letra nueva

**Conclusión.** El decoder genera letras que no están en el conjunto de entrenamiento cuando se le da un z que no es el código de ninguna letra. La letra que mostrábamos (D45, intermedia entre la "o" y la "e") salía de un modelo anterior; con T8 el camino de la "o" a la "e" pasa directo de una a otra, sin imagen intermedia, así que **la letra nueva hay que volver a elegirla** entre las candidatas de la grilla (`results/e1a4_grilla/fig_candidatas_*.png`). El latente del AE tiene zonas que decodifican bien y zonas que no, y eso es lo que motiva el VAE.

**Evidencia.**
- Con el modelo de 1a.1 (latente tanh), en una grilla de 100 puntos del latente entre 53 y 61 muestras según la semilla (5 semillas) no son idénticas a ninguna letra del dataset.
- Interpolación o→e, semilla 0: 0·o → 0·o → 0·o → 0·o → 0·e → 0·e → 0·e → 0·e → 0·e. No hay paso intermedio.
- Interpolación k→q, semilla 0: 0·k → 0·k → 4·h → 1·m → 2·w → 2·u → 4·a → 5·a → 0·q; el camino pasa por letras que no tienen que ver y por muestras a 5 píxeles de cualquier letra.

**Teoría.** Autoencoders diap. 33–34: moverse en el latente y generar especificando z directamente. Diap. 36–37: el AE no garantiza que un punto cualquiera del latente decodifique en algo válido.

**Límite.** Que una muestra "sea una letra" es un juicio nuestro; lo que medimos es que no es copia.

## 1b.1 · Arquitectura del Denoising Autoencoder

Rehecha el 2026-11-17: el DAE se prueba con las dos mejores redes del AE (puestos 1 y 2 de 1a.1) y siempre con el mismo tipo y nivel de ruido con que se entrenó (`results/e1b_ruidos/`, 10 semillas). La comparación anterior de tres arquitecturas (A, B, C) usaba latente identidad y queda como historia en `results/e1b/`.

**Conclusión.** Las dos mejores redes del AE, **35–30–20–2–20–30–35 (T8)** y **35–30–2–30–35 (T6)**, ambas con latente tanh, sacan la mayor parte del ruido en los niveles bajos y medios. T8 deja menos píxeles mal que T6 con ruido gaussiano medio y alto y con salt-and-pepper p = 0,1 y 0,2; en los demás niveles empatan. Usamos T8 para el DAE, la misma red del AE.

**Evidencia** (píxeles incorrectos por letra a la salida, T8 contra T6).
- T8 queda mejor (diferencia mayor que el desvío) con σ = 0,4 (0,66 ± 0,19 contra 1,16 ± 0,22), σ = 0,5 (1,48 ± 0,26 contra 2,22 ± 0,31) y p = 0,2 (4,14 ± 0,27 contra 4,64 ± 0,40).
- Empatan con σ = 0,1 (0,00 ± 0,00 y 0,21 ± 0,67), σ = 0,2 (0,06 ± 0,07 y 0,06 ± 0,03), σ = 0,3 (0,21 ± 0,09 y 0,50 ± 0,30), p = 0,05 (0,33 ± 0,15 y 0,39 ± 0,17), p = 0,1 (0,90 ± 0,24 y 1,18 ± 0,34) y p = 0,3 (8,15 ± 0,21 y 8,52 ± 0,41).
- La A vieja (35–20–2–20–35, latente identidad), aun con su mejor nivel de entrenamiento, dejaba 2,29 ± 0,19 con p = 0,1.

**Teoría.** Autoencoders diap. 20–22: el DAE es el mismo autoencoder entrenado con X̃ a la entrada y X a la salida; el cuello de botella lo obliga a quedarse con lo que define a la letra. Redes y receta: D57, D59 y D62 (1a.1).

**Límite.** Probamos las dos mejores redes del AE, no una búsqueda de arquitecturas propia del DAE.

## 1b.2 · Niveles y tipos de ruido

**Conclusión.** Probado con el mismo ruido con que se entrenó, el DAE (T8) saca casi todo el ruido gaussiano hasta σ = 0,4 y la mayor parte del salt-and-pepper hasta p = 0,1. Con p = 0,2 saca menos de la mitad, y con p = 0,3 la mejora es chica. A igual cantidad de píxeles alterados en la entrada (σ = 0,4 y p = 0,1), los dos tipos de ruido empatan.

**Evidencia** (T8; píxeles incorrectos por letra, entrada → salida; media ± desvío entre 10 semillas).
- Gaussiano: σ = 0,2: 0,23 ± 0,02 → 0,06 ± 0,07; σ = 0,3: 1,66 ± 0,05 → 0,21 ± 0,09; σ = 0,4: 3,68 ± 0,06 → 0,66 ± 0,19; σ = 0,5: 5,55 ± 0,07 → 1,48 ± 0,26. Con σ = 0,1 el ruido no cruza nunca el umbral (0,00 ± 0,00 píxeles alterados).
- Salt-and-pepper: p = 0,05: 1,74 ± 0,06 → 0,33 ± 0,15; p = 0,1: 3,46 ± 0,08 → 0,90 ± 0,24; p = 0,2: 6,96 ± 0,11 → 4,14 ± 0,27; p = 0,3: 10,51 ± 0,11 → 8,15 ± 0,21.
- Letras que salen exactas: con σ = 0,4, 83,3 ± 5,3 %; con p = 0,1, 79,4 ± 9,7 %; con p = 0,3, 3,0 ± 0,7 %.
- Gaussiano σ = 0,4 contra salt-and-pepper p = 0,1: la entrada tiene casi los mismos píxeles alterados (3,68 y 3,46), pero salen 0,66 ± 0,19 contra 0,90 ± 0,24; empatan (la diferencia no supera el desvío).
- Época a época (`fig_curvas.png`): con p = 0,3 casi no baja después de la época 1000 (9,03 en la época 1000, 8,08 en la 10000).

**Teoría.** Autoencoders diap. 22: se entrena con la entrada corrompida y la salida esperada limpia. Con salt-and-pepper el píxel cambiado vale exactamente lo opuesto, así que nada lo distingue de un píxel bueno. Con ruido gaussiano el píxel que cruza el umbral suele quedar cerca de 0,5, pero a igual cantidad de píxeles alterados eso no se traduce en una diferencia que supere el desvío. Con mucho ruido la entrada de una letra queda más cerca de otra letra que de la original, y ahí ningún modelo puede recuperarla.

**Límite.** Los niveles los fijamos nosotros (la cátedra no da niveles, D58); el gaussiano no se recorta a [0, 1]. Se probó solo con el mismo nivel de entrenamiento, como pidió la cátedra; no medimos qué pasa si el ruido de prueba es otro.

## 2a · Conjunto de datos nuevo

**Conclusión.** Construimos el dataset en código: 500 relojes de agujas de 16 × 16 en grises. La esfera es siempre la misma y lo único que cambia es la hora, que fija las dos agujas como en un reloj real. Lo construimos nosotros para saber qué hora marca cada imagen y poder juzgar si una muestra generada es un reloj del conjunto.

**Evidencia.** 494 de 500 imágenes distintas (las repetidas marcan el mismo minuto); el 36,1 % de los píxeles es gris.

**Teoría.** Datos en [0, 1] con salida logística y BCE, como el Colab del VAE de la cátedra (Autoencoders diap. 90).

**Límite.** Es un dataset sintético y chico, con un solo factor (la hora). Probamos variar también el tamaño y la posición de la esfera, y en una corrida de prueba el VAE usó el latente para eso y dejó las agujas borrosas; por eso quedaron fijos (`DESVIOS.md`).

## 2b · Esquema variacional y el problema del latente

**Conclusión.** El VAE resuelve el problema del latente: sus códigos ocupan la zona de la normal estándar, que es de donde después se samplea para generar. El AE con la misma arquitectura deja los códigos desparramados en una zona que depende de la semilla. El precio es una reconstrucción un poco peor, y λ regula ese intercambio; con λ = 10 el VAE deja de usar el latente.

**Evidencia** (misma arquitectura, mismos datos, 5 semillas).
- Desvío de los códigos: AE 11,89 ± 1,44; VAE con λ = 1: 1,03 ± 0,01. El prior tiene desvío 1.
- Códigos a menos de 3 del origen: AE 0,15 ± 0,05; VAE con λ = 1: 1,00 ± 0,00.
- Reconstrucción L: AE 50,92 ± 0,32; VAE con λ = 1: 53,42 ± 0,28. La diferencia supera el desvío: el VAE reconstruye peor.
- Efecto de λ: con λ = 0,1, R = 7,84 ± 0,17 y L = 51,89 ± 0,29; con λ = 10, R = 0,00 ± 0,00 y L = 62,19 ± 0,00, con σ media 1,000 ± 0,000: todos los relojes reciben el mismo código y el decoder dibuja siempre la esfera con una mancha en el centro.
- El VAE aprende: L baja de 173,83 ± 1,22 a 54,37 ± 0,20, con un piso de 48,45 para estos datos.

**Teoría.** Autoencoders diap. 74–78: se supone p(z) = N(0, I) y se agrega a la reconstrucción el término R = KL(q‖p), que penaliza que la distribución de cada código se aleje de la normal estándar; J = L + λR (ec. 13). Diap. 80–89: el truco de reparametrización z = μ + σ·ϵ permite hacer backpropagation a través del muestreo. Con λ grande gana R: los códigos se pegan al prior, σ crece y el decoder recibe lo mismo para todos los datos. Con λ chico gana L y el modelo se acerca a un AE.

**Límite.** En la grilla del latente las agujas giran de forma continua, pero la de los minutos sale más borrosa que la de las horas: con 2 dimensiones, el VAE ordena mejor la hora que los minutos.

## 2c · Generar una muestra nueva

**Conclusión.** Decodificando z sorteados de N(0, I) con el VAE (λ = 1) se obtienen relojes nuevos que juzgamos del conjunto: tienen esfera, marcas y dos agujas, y no son copia de ningún reloj del dataset. A diferencia del AE, las muestras del VAE recorren todas las horas. Mostramos las candidatas 14 y 36.

**Evidencia.**
- Horas del reloj (de 12) en las que cae el más cercano de alguna de las 200 muestras: VAE con λ = 1, 12,0 ± 0,0; AE, 7,4 ± 1,5. Con z ~ N(0, I) el AE solo visita la zona de su latente cercana al origen.
- Distancia de una generada al reloj más cercano del dataset: VAE con λ = 1, 0,018 ± 0,000; AE, 0,013 ± 0,001. Entre relojes reales, la distancia al otro más cercano es 0,006 en promedio.
- Ninguna de las 4000 muestras generadas es idéntica a una imagen del dataset.
- Las elegidas, con su vecino del dataset y la hora que marca: `results/elegidas/fig_relojes.png` y `elegidas.csv`.

**Teoría.** Autoencoders diap. 34 y 74: se descarta el encoder y se decodifica un z sorteado del prior; funciona porque el entrenamiento llevó los códigos a esa zona (2b).

**Límite.** Las muestras del VAE son más borrosas que los relojes reales (su distancia al más cercano es mayor que la del AE, y la diferencia supera el desvío). "Pertenece al conjunto" es un juicio nuestro, como pide la consigna; la distancia por píxel y las horas cubiertas son métricas propias, no de la cátedra.
