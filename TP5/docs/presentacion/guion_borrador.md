# TP5 · Borrador del guion

Generado desde `laminas.json`. Es un borrador: quien presenta cada lámina lo reescribe con sus palabras.

## 1. TP5 · Deep Learning

Tres partes: comprimir letras con un autoencoder, limpiar ruido con un DAE y generar relojes con un VAE. Todo implementado a mano en numpy.

## 2. El autoencoder son dos perceptrones multicapa en espejo

Autoencoders diap. 3–4. El encoder lleva X al espacio latente Z; el decoder tiene las capas invertidas y devuelve X′. En el ejercicio 1, X son 35 píxeles y Z tiene 2 dimensiones.

## 3. Con latente lineal y una capa oculta, la red representa las letras en dos dimensiones

Arquitectura final: 35–20–2–20–35, ocultas tanh, latente identidad, salida logística. El cambio que más pesa es sacar la logística del latente: con logística los códigos quedan encerrados entre 0 y 1. Entre latente lineal y ocultas tanh la cantidad de píxeles empata (la diferencia es menor que el desvío). Fuente: results/e1a2/resumen.csv.

## 4. Ningún optimizador alcanza un píxel de error con error cuadrático

Mismo modelo, cambia solo el optimizador (Clase 12.1). Momentum, η adaptativo, RMSProp y Adam no mejoran al descenso por gradiente. Powell (scipy) no usa el gradiente y con 177 pesos se detiene lejos del mínimo. Fuente: results/e1a2/resumen.csv, 10 semillas por barra.

## 5. Los píxeles equivocados se saturan y dejan de aprender

Clase 11, diap. 40–43: el delta de la capa de salida lleva el factor θ′(h). Con salida logística, si la neurona queda saturada en el valor equivocado, θ′ vale casi 0 y el píxel con más error es el que menos se corrige. Con entropía cruzada (Autoencoders diap. 51) el factor θ′ se cancela. Conteo sobre los escalones con ocultas tanh y error cuadrático. Fuente: results/e1a2/resumen.csv.

## 6. Con entropía cruzada la red aprende las 32 letras sin errores

Mismo modelo y Adam; cambia solo la función de costo. Declarar: la diapositiva del autoencoder usa el error cuadrático; la entropía cruzada es de la cátedra (diap. 51 y Colab del VAE). Fuente: results/e1a2/resumen.csv.

## 7. El latente ubica las letras, pero su forma cambia con la semilla

Cada punto es el código Z de una letra. Las dos redes reconstruyen sin errores, pero ubican las letras distinto: la función de costo no le pide ninguna forma al latente. La correlación es de Pearson entre las 496 distancias entre pares de letras en un latente y en el otro. Con 5 semillas va de 0,14 a 0,70. Fuente: results/e1a3_e1a4/estabilidad.csv.

## 8. Una proyección lineal a dos dimensiones deja 230 píxeles mal

Línea base: PCA con 2 componentes (calculado con SVD, sin librerías). El autoencoder es una reducción de dimensión no lineal y por eso entra lo que PCA no. Fuente: results/e1a3_e1a4/pca.csv.

## 9. Entre la “o” y la “e” la red genera una letra que no está en el dataset

Autoencoders diap. 33–34: moverse en el latente y decodificar un z que no es el código de ninguna letra. En una grilla de 100 puntos, entre 60 y 73 muestras no son letras del dataset, pero no todas se leen como una letra: el latente del autoencoder tiene zonas que no decodifican bien, y eso motiva el VAE. Fuente: results/elegidas/elegidas.csv.

## 10. Dos números alcanzan para guardar cada letra

Consecuencia: antes de probar optimizadores más sofisticados conviene mirar si el gradiente está llegando. Y un autoencoder común comprime bien, pero no es un generador confiable.

## 11. El DAE recibe la letra con ruido y tiene que devolverla limpia

Autoencoders diap. 22. Es la misma red; cambia la entrada del entrenamiento. El ruido es salt-and-pepper y se sortea de nuevo en cada época.

## 12. Un cuello de botella de diez neuronas saca más ruido que uno de dos o que ninguno

Tres arquitecturas: A con latente de 2, B con latente de 10 (la elegida), C sin cuello de botella (35–50–35). Filas: ruido de entrenamiento; columnas: ruido de prueba. B gana en los cuatro niveles de prueba por más que el desvío. El latente de 2 no tiene lugar para la letra más la tolerancia al ruido; sin cuello de botella no hay restricción que obligue a descartar el ruido. Los tres números son con entrenamiento y prueba en p = 0,1. Fuente: results/e1b/resumen.csv, 5 semillas.

## 13. Sin entrenar con ruido, el autoencoder lo empeora

Línea punteada: los píxeles alterados que trae la entrada (35·p). Por encima de esa línea, la red empeora la letra. El autoencoder de 1a queda siempre por encima; el DAE, por debajo. Fuente: results/e1b/resumen.csv.

## 14. El DAE corrige casi todo con ruido bajo y cae con ruido alto

Arquitectura B entrenada con p = 0,1: probada con p = 0,1 y con p = 0,3. Entrenar con demasiado ruido (p = 0,3) arruina también las letras limpias. Conviene un nivel de entrenamiento moderado. Fuente: results/e1b/resumen.csv.

## 15. Los errores son confusiones entre letras parecidas

A cada salida se le asigna la letra del dataset más cercana. Con ruido, la entrada corrompida de una letra puede quedar más cerca de otra letra que de la original, y ahí ningún modelo la recupera. Fuente: results/e1b/confusion.csv.

## 16. Para limpiar ruido hay que entrenar con ruido y dejar un cuello de botella intermedio

Consecuencia: el nivel de ruido del entrenamiento se elige según el ruido que se espera en uso, y hay un límite que no depende del modelo.

## 17. El VAE aprende una distribución por dato y samplea de ella

Autoencoders diap. 74–89. L es la reconstrucción y R penaliza que la distribución de cada código se aleje de la normal estándar. El truco de reparametrización saca el sorteo afuera (ϵ) para poder hacer backpropagation. El encoder devuelve μ y s = log σ². Verificado con gradient check con ϵ fijo.

## 18. Construimos 500 relojes de agujas: lo único que cambia es la hora

Relojes de 16 × 16. La esfera es fija; la hora, sorteada al azar, fija las dos agujas como en un reloj real (la de las horas avanza también con los minutos). Probamos variar también el tamaño y la posición de la esfera, pero el VAE gastaba el latente en eso y dejaba las agujas borrosas. Lo construimos para poder juzgar si una muestra generada es un reloj del conjunto.

## 19. El VAE aprende: la reconstrucción baja y la regularización se estabiliza

Encoder 256–64–(μ y s de 2), decoder 2–64–256, λ = 1, Adam, 2000 épocas, 5 semillas (una línea por semilla). L no puede llegar a 0: con datos en grises tiene un piso de 48,45. Fuente: results/e2b/curvas.csv.

## 20. El VAE lleva los códigos a la normal estándar; el autoencoder los desparrama

Misma arquitectura y mismos datos; cambia solo el latente. El color es la hora que marca cada reloj. La línea punteada es ‖z‖ = 3. En el autoencoder solo el 15 % de los códigos cae adentro; en el VAE con λ = 1, todos. Fuente: results/e2c/corridas.csv, 5 semillas.

## 21. λ intercambia nitidez por orden en el latente

El decoder evaluado sobre la misma grilla del latente. J = L + λR (diap. 78, ec. 13). Con λ = 10 gana R: todos los códigos quedan en el origen con σ = 1 (R = 0) y el decoder dibuja la esfera con una mancha gris. Con λ = 0,1 gana L. Números para decir: L pasa de 51,89 a 62,19 y R de 7,84 a 0,00 entre λ = 0,1 y λ = 10. Fuente: results/e2c/corridas.csv.

## 22. Los relojes generados son nuevos y el VAE recorre todas las horas

Se descarta el encoder y se decodifica z sorteado de N(0, I). Los juzgamos del conjunto: tienen esfera, marcas y dos agujas. Ninguna de las 4000 generadas es copia de una del dataset. Para medir variedad miramos la hora del reloj del dataset más cercano a cada muestra. Límite: las del VAE son más borrosas; su distancia por píxel al más cercano (0,018) es mayor que la del autoencoder y que la de los relojes reales entre sí (0,006). Fuente: results/e2c/corridas.csv y results/elegidas/elegidas.csv.

## 23. El VAE ordena el latente, y por eso permite generar

Cierre: la misma red de dos MLP en espejo sirve para tres tareas según cómo se la entrena. Para generar hay que pedirle una forma al latente, y eso es lo que agrega el término R.
