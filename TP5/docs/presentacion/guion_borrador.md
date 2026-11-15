# TP5 · Borrador del guion

Generado desde `laminas.json`. Es un borrador: quien presenta cada lámina lo reescribe con sus palabras.

## 1. TP5 · Deep Learning · presenta Louis

Tres partes: comprimir letras con un autoencoder, limpiar ruido con un DAE y generar relojes con un VAE. Todo implementado a mano en numpy.

## 2. El autoencoder son dos perceptrones multicapa en espejo · presenta Louis

Autoencoders diap. 3–4. El encoder lleva X al espacio latente Z; el decoder tiene las capas invertidas y devuelve X′. En el ejercicio 1, X son 35 píxeles y Z tiene 2 dimensiones.

## 3. Cambiar solo la arquitectura baja los errores de 210 a 12, pero no alcanza el objetivo · presenta Louis

Cada barra agrega un cambio: una capa oculta de 20 y después latente identidad (error cuadrático, descenso por gradiente). El cambio que más pesa es sacar la logística del latente: con logística los códigos quedan encerrados entre 0 y 1. Sin capa oculta (35-2-35) quedan 210 píxeles mal, cerca de los 230 de PCA. La cátedra muestra que un autoencoder lineal equivale a PCA (Autoencoders diap. 7–16); el nuestro tiene activación logística, así que es una comparación y no la misma cosa. Fuente: results/e1a2/resumen.csv y results/e1a3_e1a4/pca.csv.

## 4. Con entropía cruzada, todos los optimizadores salvo η adaptativo aprenden las 32 letras en todas las semillas · presenta Gago

Mismo modelo 35-20-2-20-35; cambian el optimizador, su η y la función de costo (Clase 12.1 diap. 5–13; Autoencoders diap. 51). 10 semillas por barra. Mejores η, error cuadrático: GD 0,01; momentum 0,003; η adaptativo 0,03; RMSProp 0,01; Adam 0,001. Entropía cruzada: GD 0,001; momentum 0,0001; RMSProp 0,001; Adam, los tres valores llegan. Donde el mejor η quedaba en el borde se probó un valor más; solo η adaptativo con entropía cruzada sigue sin llegar con ningún η. Por qué: sube η de a 0,002 hasta 0,009–0,015, donde GD con η fijo tampoco llega; el error salta y queda en una meseta. No diverge ni η se va a 0. Powell (scipy, red 35-2-35) no llega en ninguna semilla. Control: una red más profunda (35-30-20-10-2-10-20-30-35) con Adam y error cuadrático, batch completo, llega en 0 de 10 semillas con 10.000 épocas y en 0 de 10 con 40.000. Fuente: results/e1a2_optimizadores/ y results/e1a2_eta_adaptativo/.

## 5. Los píxeles equivocados se saturan y dejan de aprender · presenta Gago

Clase 11, diap. 40–43: el delta de la capa de salida lleva el factor θ′(h). Con salida logística, si la neurona queda saturada en el valor equivocado, θ′ vale casi 0 y el píxel con más error es el que menos se corrige. Con entropía cruzada (Autoencoders diap. 51) el factor θ′ se cancela. Conteo sobre los escalones con ocultas tanh y error cuadrático. Fuente: results/e1a2/resumen.csv.

## 6. El latente ubica las letras, pero su forma cambia con la semilla · presenta Gago

Cada punto es el código Z de una letra. Las dos redes reconstruyen sin errores, pero ubican las letras distinto: la función de costo no le pide ninguna forma al latente. La correlación es de Pearson entre las 496 distancias entre pares de letras en un latente y en el otro. Con 5 semillas va de 0,14 a 0,70. Fuente: results/e1a3_e1a4/estabilidad.csv.

## 7. Una proyección lineal a dos dimensiones deja 230 píxeles mal · presenta Gago

Línea base: PCA con 2 componentes (calculado con SVD, sin librerías). El autoencoder es una reducción de dimensión no lineal y por eso entra lo que PCA no. Fuente: results/e1a3_e1a4/pca.csv.

## 8. En el latente del autoencoder aparecen formas que no están en el dataset, a 3 píxeles o más de toda letra · presenta Gago

Autoencoders diap. 33–34: moverse en el latente y decodificar un z que no es el código de ninguna letra. Criterio: 2 píxeles es la distancia mínima entre dos letras distintas del propio dataset (“l” y “|”), así que a 3 píxeles o más ya no es la misma letra con ruido. Una grilla de 50 × 50 sobre el latente da 526 salidas que no son letras; 472 quedan a 2 píxeles o más. Elegimos las que se leen como un glifo y salen de una región amplia del latente: una “S” (a 6 píxeles de la “g”), una “o” con un punto y una “u” con un punto. Caen en el borde del latente, lejos de las letras. Entre la “o” y la “e”, en cambio, ningún paso pasa de 2 píxeles. Fuente: results/e1a4_grilla/unicas.csv y config/elegidas.json.

## 9. Dos números alcanzan para guardar cada letra · presenta Gago

Consecuencia: antes de probar optimizadores más sofisticados conviene mirar si el gradiente está llegando. Y un autoencoder común comprime bien, pero no es un generador confiable.

## 10. El DAE recibe la letra con ruido y tiene que devolverla limpia · presenta Agus

Autoencoders diap. 22. Es la misma red; cambia la entrada del entrenamiento. El ruido es salt-and-pepper y se sortea de nuevo en cada época.

## 11. Entre las tres arquitecturas, la del cuello de botella intermedio (B) saca más ruido · presenta Agus

Tres arquitecturas: A con latente de 2, B con latente de 10 (la elegida), C sin cuello de botella (35–50–35). Filas: ruido de entrenamiento; columnas: ruido de prueba. B gana en los cuatro niveles de prueba por más que el desvío. El latente de 2 no tiene lugar para la letra más la tolerancia al ruido; sin cuello de botella no hay restricción que obligue a descartar el ruido. A (latente de 2) deja 2,6. No probamos una red profunda sin cuello de botella (35-20-50-20-35): C cambia a la vez la profundidad y el cuello, por eso el título dice "entre las tres". Las tres con el entrenamiento corto (10.000 épocas, una réplica por letra): con p = 0,1, B reconstruye exactas el 75 %; con el entrenamiento largo llega al 90 % (lámina 13). Fuente: results/e1b/resumen.csv, 5 semillas.

## 12. Sin entrenar con ruido, el autoencoder lo empeora · presenta Agus

El DAE es el del entrenamiento corto (75 % exactas con p = 0,1). Línea punteada: los píxeles alterados que trae la entrada (35·p). Por encima de esa línea, la red empeora la letra. El autoencoder de 1a queda siempre por encima; el DAE, por debajo. Fuente: results/e1b/resumen.csv.

## 13. Con más entrenamiento el DAE mejora, pero no alcanza a la letra más cercana · presenta Agus

Arquitectura B entrenada con p = 0,1. Con el entrenamiento actual reconstruye exactas el 75 %. Se probaron 1 o 5 réplicas ruidosas de cada letra por época (data augmentation, Clase 13) y 10.000 o 40.000 épocas. Declarar: con 5 réplicas también hay 5 veces más actualizaciones por época. Línea punteada: devolver la letra del dataset más cercana a la entrada ruidosa, con el mismo ruido. Con p = 0,3: DAE actual 10 %, letra más cercana 53 %. Con a lo sumo un píxel incorrecto, con p = 0,1: 85 % el entrenamiento corto, 94 % el largo y 98 % la letra más cercana. Fuente: results/e1b_replicas/resumen.csv y results/e1b/referencia.csv.

## 14. Los errores son confusiones entre letras parecidas · presenta Agus

El DAE es el del entrenamiento corto (75 % exactas con p = 0,1). A cada salida se le asigna la letra del dataset más cercana. Con p = 0,1, solo el 0,6 % de las entradas queda más cerca de otra letra: casi todas estas confusiones son errores del modelo y se podrían evitar. Con p = 0,3, ya es el 37 %. Fuente: results/e1b/confusion.csv y referencia.csv.

## 15. Entrenar con ruido limpia las letras, pero elegir la letra más cercana sigue ganando · presenta Agus

Consecuencia: el nivel de ruido del entrenamiento se elige según el ruido que se espera en uso. Entre las tres arquitecturas probadas, la del cuello de botella intermedio sacó más ruido. Con p = 0,1, el DAE con el entrenamiento más largo llega a 90 % y la letra más cercana a 98 %. El límite de las letras parecidas pesa recién con p = 0,3.

## 16. El VAE aprende una distribución por dato y samplea de ella · presenta Canzo

Autoencoders diap. 74–89. L es la reconstrucción y R penaliza que la distribución de cada código se aleje de la normal estándar. El truco de reparametrización saca el sorteo afuera (ϵ) para poder hacer backpropagation. El encoder devuelve μ y s = log σ². Verificado con gradient check con ϵ fijo.

## 17. Construimos 500 relojes de agujas: lo único que cambia es la hora · presenta Canzo

Relojes de 16 × 16. La esfera es fija; la hora, sorteada al azar, fija las dos agujas como en un reloj real (la de las horas avanza también con los minutos). Probamos variar también el tamaño y la posición de la esfera, pero el VAE gastaba el latente en eso y dejaba las agujas borrosas. Lo construimos para poder juzgar si una muestra generada es un reloj del conjunto.

## 18. El VAE aprende: la reconstrucción baja y la regularización se estabiliza · presenta Canzo

Encoder 256–64–(μ y s de 2), decoder 2–64–256, λ = 1, Adam, 2000 épocas, 5 semillas (una línea por semilla). Línea punteada: el piso de L. L no puede llegar a 0: con datos en grises tiene un piso de 48,4 (la BCE de cada reloj consigo mismo). Empieza en 174. Fuente: results/e2b/curvas.csv.

## 19. El VAE lleva los códigos a la normal estándar; el autoencoder los desparrama · presenta Canzo

Misma arquitectura y mismos datos; cambia solo el latente. El color es la hora que marca cada reloj. La línea punteada es ‖z‖ = 3. En el autoencoder solo el 15 % de los códigos cae adentro; en el VAE con λ = 1, todos. Qué guarda el latente: la hora. Una grilla de 8 × 8 sobre μ explica 0,53 de la variación del ángulo de la hora, contra 0,11 por azar. No queda en un círculo ordenado: Pearson solo ve una parte (|r| ≤ 0,51). Fuente: results/e2c/corridas.csv, results/e2c_parametros/celdas.csv y results/e2c_justa/correlaciones.csv, 5 semillas.

## 20. λ intercambia nitidez por orden en el latente · presenta Bruno

El decoder evaluado sobre la misma grilla del latente. J = L + λR (diap. 78, ec. 13). Con λ = 10 gana R: todos los códigos quedan en el origen con σ = 1 (R = 0) y el decoder dibuja la esfera con una mancha gris. Con λ = 0,1 gana L. Números para decir: L pasa de 51,89 a 62,19 y R de 7,84 a 0,00 entre λ = 0,1 y λ = 10. Fuente: results/e2c/corridas.csv.

## 21. Muestreando donde viven sus códigos, los dos modelos recorren las doce horas, pero el autoencoder falla más · presenta Bruno

Grillas: cada modelo sobre el rango de sus propios códigos (z en el autoencoder, μ(x) en el VAE). La esquina del autoencoder decodifica manchas: su latente tiene zonas vacías. Muestreo: el VAE de N(0, I), su prior; el autoencoder de una normal con la media y la covarianza de sus códigos (D50). Falla: cada modelo reconstruye 500 relojes reales nuevos; una muestra generada falla si queda más lejos de un banco de 20.000 relojes nuevos que el percentil 99 de esas reconstrucciones. El umbral no sale de las muestras que se juzgan. Con el percentil 95: 18,2 % contra 6,9 %; con el máximo: 10,0 % contra 0,3 %. Los dos cubren las 12 horas y ninguno copia. Fuente: results/e2c_justa/fallas.csv y relojes.csv, 5 semillas.

## 22. Los relojes del VAE no son copias, pero son más borrosos: distancia típica al banco 0,017, contra 0,013 del autoencoder · presenta Bruno

Se descarta el encoder y se decodifica z sorteado de N(0, I). Los juzgamos del conjunto: tienen esfera, marcas y dos agujas. Distancia típica: la mediana de la diferencia absoluta media por píxel al reloj más cercano de un banco de 20.000 relojes nuevos (métrica nuestra). Con N(0, I) para los dos, el autoencoder cubría 7,4 de las 12 horas (e2c); muestreado donde viven sus códigos, cubre las 12 como el VAE. Fuente: results/e2c_justa/resumen.csv, results/e2c/corridas.csv y results/elegidas/elegidas.csv.

## 23. El VAE ordena el latente: sus relojes son más borrosos, pero falla menos que el autoencoder · presenta Bruno

Cierre: la misma red de dos MLP en espejo sirve para tres tareas según cómo se la entrena. Para generar hay que pedirle una forma al latente, y eso es lo que agrega el término R.
