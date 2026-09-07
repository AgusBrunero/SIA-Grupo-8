# Caso final — La noche estrellada

> Cierre del análisis. Los operadores **se heredan** del barrido (el ganador de cada
> eje, promediado entre las tres imágenes); ver [`config.md`](config.md) con la tabla
> completa y si cada ganador está respaldado o es un empate. Acá se barre sólo lo que
> depende del tamaño del problema.

## Por qué esta imagen

Es la que la cátedra mostró como ejemplo, y es el caso opuesto a los targets del
barrido: `japan` es una región plana, `pika` tiene detalle acotado sobre fondo liso.
La noche estrellada **no tiene una sola región plana**: es textura en cada píxel, sin
bordes duros. Sirve para dos cosas — mostrar dónde deja de alcanzar el método, y poner
a prueba decisiones tomadas con otras imágenes sobre una que nunca vimos.

## El problema de transferir la tasa de mutación

`pm` es una probabilidad **por gen**. El eje [`08-tasa-mutacion`](../08-tasa-mutacion/informe.md) encontró que la mejor carga es **2 genes mutados por individuo y generación**, medida con 500 genes. Acá el cromosoma es mucho más largo, y hay dos maneras incompatibles de trasladar ese resultado:

| Triángulos | Genes | `pm fijo` → carga | `carga fija` → pm |
|---|---|---|---|
| 100 | 1,000 | pm=0.0040 → carga 4 | carga 2 → pm=0.00200 |
| 200 | 2,000 | pm=0.0040 → carga 8 | carga 2 → pm=0.00100 |
| 400 | 4,000 | pm=0.0040 → carga 16 | carga 2 → pm=0.00050 |
| 800 | 8,000 | pm=0.0040 → carga 32 | carga 2 → pm=0.00025 |

Con 800 triángulos, copiar `pm` muta **16 veces más genes por individuo** que en la corrida donde ese valor ganó. Copiar el número no es heredar la configuración: es cambiarla. Por eso el régimen es una dimensión del barrido y no un supuesto.

## Resultados

4 cantidades × 2 regímenes × 3 semillas × 1200 generaciones.

| # | Triángulos | Régimen | pm | Fitness | RMSE | Tiempo/corrida |
|---|---|---|---|---|---|---|
| 1 | 800 | carga fija | 0.00025 | 0.9356 ± 0.0004 | 16.43 | 550 s |
| 2 | 400 | carga fija | 0.00050 | 0.9300 ± 0.0008 | 17.84 | 287 s |
| 3 | 200 | carga fija | 0.00100 | 0.9259 ± 0.0016 | 18.89 | 161 s |
| 4 | 100 | carga fija | 0.00200 | 0.9214 ± 0.0010 | 20.04 | 70 s |
| 5 | 200 | pm fijo | 0.00400 | 0.9169 ± 0.0008 | 21.19 | 173 s |
| 6 | 100 | pm fijo | 0.00400 | 0.9162 ± 0.0001 | 21.37 | 84 s |
| 7 | 400 | pm fijo | 0.00400 | 0.9137 ± 0.0008 | 22.01 | 329 s |
| 8 | 800 | pm fijo | 0.00400 | 0.9105 ± 0.0004 | 22.83 | 655 s |

**Gana 800 triángulos con carga fija** (0.9356), contra 0.9105 de la peor combinación (800 tri · pm fijo).

### El régimen de mutación decide más que la cantidad de triángulos

| Triángulos | pm fijo | carga fija | Diferencia |
|---|---|---|---|
| 100 | 0.9162 | 0.9214 | +0.0052 |
| 200 | 0.9169 | 0.9259 | +0.0090 |
| 400 | 0.9137 | 0.9300 | +0.0164 |
| 800 | 0.9105 | 0.9356 | +0.0251 |

**Mantener la carga constante mejora en 4 de las 4 cantidades**, y la ventaja crece con los triángulos (+0.0251 con 800). Es la conclusión más transferible del trabajo:

> **Un hiperparámetro medido a una escala no se copia a otra: se traduce.** Lo que
> hay que mantener constante al cambiar el largo del cromosoma no es la
> probabilidad por gen sino la cantidad esperada de genes mutados por individuo.

Y tiene una consecuencia hacia atrás: el eje [`11-triangulos`](../11-triangulos/informe.md) mide exactamente este cruce sobre las otras imágenes, y es el motivo por el que ahí también se corren los dos regímenes en vez de uno solo.

### Capacidad contra costo de búsqueda: las curvas se cruzan

En la generación 1 arriba está `800 tri · pm fijo` (0.6341) y abajo `200 tri · carga fija` (0.5395). Al final gana `800 tri · carga fija`.

| Combinación | Fitness gen. 1 | Fitness gen. 1200 | Ganancia |
|---|---|---|---|
| 100 tri · carga fija | 0.5685 | 0.9214 | +0.3529 |
| 100 tri · pm fijo | 0.5667 | 0.9162 | +0.3495 |
| 200 tri · carga fija | 0.5395 | 0.9259 | +0.3864 |
| 200 tri · pm fijo | 0.5429 | 0.9169 | +0.3741 |
| 400 tri · carga fija | 0.6123 | 0.9300 | +0.3178 |
| 400 tri · pm fijo | 0.6179 | 0.9137 | +0.2957 |
| 800 tri · carga fija | 0.6291 | 0.9356 | +0.3065 |
| 800 tri · pm fijo | 0.6341 | 0.9105 | +0.2763 |

El cruce ocurre alrededor de la **generación 137**.

Se ven los dos efectos separados:

- **La inicialización informada premia la capacidad**: más triángulos es una
  grilla más fina, así que el punto de partida ya aproxima mejor el target.
- **La búsqueda la castiga**: cada triángulo son 10 genes más que optimizar. A
  presupuesto fijo, el espacio más grande se explora peor.

Cuál de los dos gana depende del presupuesto, así que **la cantidad de triángulos
no se elige por capacidad sino por cuántas generaciones se van a correr.**

## La corrida final

- **800 triángulos**, régimen `carga fija` (pm = 0.00025), 3000 generaciones, 240,080 evaluaciones, 804 s
- **Fitness 0.9442** (RMSE 14.23)
- Canvas de evaluación 96×76px; la imagen
  entregada se renderiza a 900×712px **desde el
  mismo genotipo** — el individuo no tiene resolución propia, y eso es una
  propiedad del formato de salida, no un detalle de implementación

### Como compresor

El enunciado plantea el TP como un compresor con pérdida. Acá está el número:

| | Tamaño |
|---|---|
| Original (`noche_estrellada.jpg`) | 604 KB |
| `triangles.json` (800 triángulos) | 80 KB |
| **Relación** | **7.6×** |

Con la salvedad de que el JSON es texto sin comprimir y el original es un JPEG ya
comprimido. La comparación honesta es *cuánta información hace falta para describir
la imagen*: 800 × 10 = 8,000 números reales.

## Qué mirar

- `imagenes/comparacion.png` — original y resultado lado a lado. Es la slide de cierre.
- `imagenes/<n>-tri-<regimen>.png` — las 8 combinaciones. **Acá se ve lo que los
  números no dicen**: con `pm fijo` y muchos triángulos la imagen queda ruidosa,
  porque se están mutando decenas de genes por individuo y por generación.
- `imagenes/evolucion.png` — la corrida en 8 momentos. Se ve el orden en que el
  algoritmo construye: primero el reparto de masas de color, después el cielo, y el
  ciprés y el pueblo al final. Los detalles chicos son los últimos porque aportan poco
  al RMSE hasta que lo grueso está resuelto.
- `figuras/convergencia_compleja.png` — que ninguna curva esté plana al final es la
  advertencia honesta: **la corrida no convergió, se quedó sin presupuesto.**

## Limitaciones, dichas antes de que las pregunten

- **El fitness es RMSE en RGB, que no es percepción.** En una imagen con esta textura
  el óptimo del RMSE es emborronar: promediar cada zona minimiza el error cuadrático
  mejor que intentar reproducir la pincelada. Buena parte de lo que se ve «lavado» no
  es una falla del AG — **está optimizando exactamente lo que le pedimos**. Un fitness
  perceptual (ΔE en Lab) es la continuación natural.
- La configuración combina ganadores de ejes distintos. Cuánto cuesta eso está medido
  en [`14-interaccion`](../14-interaccion/informe.md); esta corrida no es «la mejor
  configuración posible» sino la mejor que se puede *justificar* con lo medido.
- `population_size` se fijó por presupuesto, no por medición a esta escala.
- 3 semillas en el barrido: alcanza para la tendencia, no para diferencias
  chicas.

## Archivos

| Archivo | Qué es |
|---|---|
| `imagenes/comparacion.png` | Original y resultado, lado a lado |
| `imagenes/resultado.png` | Sólo el resultado |
| `imagenes/target.png` | Sólo el original, al mismo tamaño |
| `imagenes/evolucion.png` | Tira de 8 momentos de la corrida |
| `imagenes/gen_*.png` | Cada momento por separado, para componer la slide |
| `imagenes/<n>-tri-<regimen>.png` | El mejor de cada combinación del barrido |
| `imagenes/triangles.json` | La enumeración de triángulos: la «compresión» |
| `datos/barrido.csv` | Una fila por generación, combinación y semilla |
| `datos/final.json` | Métricas de la corrida final |
