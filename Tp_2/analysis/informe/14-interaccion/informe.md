# ¿Los ganadores de cada eje componen?

> Todo el análisis varía **un factor por vez**. Eso aísla bien el efecto de cada
> operador, pero no responde si armar la configuración con el ganador de cada eje da
> la mejor configuración. Este experimento lo mide.

## Método

1. Se toma de `summary.csv` la variante ganadora de cada eje, **promediando entre las
   3 imágenes** (elegir por una sola sería otra forma de sobreajustar).
2. Se corren tres configuraciones completas: la **base** del barrido, la **compuesta**
   (todos los ganadores juntos) y la **inversa** (todos los perdedores), como cota.
3. Desde la base se reemplaza **un operador por vez** por el de la compuesta, y se mide
   la mejora de cada reemplazo aislado.
4. Se compara la **suma de las mejoras individuales** contra la **mejora real** de
   aplicarlas todas juntas.

Si los efectos fueran aditivos, las dos cantidades coincidirían.

## La configuración compuesta

| Eje | Ganador (promedio entre imágenes) | Perdedor |
|---|---|---|
| `seleccion` | **torneo det** | ruleta |
| `supervivencia` | **exclusiva K=2N** | exclusiva K=N/2 |
| `cruza` | **uniforme** | espacial |
| `granularidad` | **corte por triangulo** | corte por componente |
| `tasa_cruza` | **pc=1.00** | pc=0.00 (solo mutacion) |
| `mutacion` | **gen (carga 1)** | no uniforme (carga 4 -> 0.4) |
| `tasa_mutacion` | **carga 2** | carga 50 |
| `sigma` | **sigma=0.20** | sigma=0.02 |
| `poblacion` | **N=K=120** | N=K=15 |
| `inicializacion` | **grilla informada** | al azar |

## Resultados

| Configuración | Fitness (promedio entre imágenes) | Δ vs. base |
|---|---|---|
| base | 0.9104 | — |
| compuesta (mejor de cada eje) | 0.9267 | +0.0164 |
| inversa (peor de cada eje) | 0.6444 | -0.2659 |

### Reemplazando un operador por vez

| Operador reemplazado | Fitness | Δ vs. base |
|---|---|---|
| seleccion | 0.9104 | +0.0000 |
| supervivencia | 0.9188 | +0.0085 |
| cruza | 0.9152 | +0.0049 |
| granularidad | 0.9153 | +0.0050 |
| tasa_cruza | 0.9127 | +0.0023 |
| mutacion | 0.9105 | +0.0002 |
| tasa_mutacion | 0.9114 | +0.0011 |
| sigma | 0.9126 | +0.0022 |
| poblacion | 0.9153 | +0.0050 |
| inicializacion | 0.9227 | +0.0123 |

| **Suma de las mejoras individuales** | | **+0.0414** |
| **Mejora real de aplicarlas todas juntas** | | **+0.0164** |

## Lectura

**Los efectos no se suman.** Individualmente los reemplazos aportan
+0.0414, pero aplicados juntos rinden +0.0164: se pierde
0.0251 en el camino. Hay **interacción** entre operadores — parte de
lo que cada uno aporta por separado es la misma mejora, o directamente se estorban.

Consecuencia metodológica, y es la conclusión que vale la pena defender:
**optimizar cada componente por separado y juntar los ganadores no da la mejor
configuración.** Un barrido de un factor por vez sirve para *entender* qué hace
cada operador; para *elegir* una configuración hay que medir configuraciones
completas.

La configuración **inversa** (el peor de cada eje) queda -0.2659 respecto
de la base. La distancia entre la compuesta y la inversa es el rango total que abarcan
las decisiones de operador: **0.2823
de fitness**. Sirve para dimensionar cuánto importa realmente afinar los operadores
comparado con, por ejemplo, cambiar la cantidad de triángulos o el presupuesto.

## Figuras

| Archivo | Qué muestra |
|---|---|
| `figuras/ofat.png` | La mejora de cada reemplazo aislado, con la suma y el efecto conjunto marcados como líneas verticales |

## Datos

- `datos/configuraciones.csv` — fitness por configuración, imagen y semilla
- `datos/resumen.json` — los ganadores elegidos y los totales
