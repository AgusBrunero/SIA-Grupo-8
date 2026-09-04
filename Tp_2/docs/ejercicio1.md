# Ejercicio 1 — ASCII art con Algoritmos Genéticos

El enunciado pide **pensar** (no implementar) cómo representar una imagen cuadrada en
un mapa de NxN caracteres ASCII usando AG.

## Representación

| | |
|---|---|
| **Individuo** | Una matriz NxN de caracteres: un ASCII art candidato |
| **Gen** | Un índice entero en el alfabeto `A` (por ejemplo `" .:-=+*#%@"`, 10 caracteres, o los 95 ASCII imprimibles) |
| **Cromosoma** | Vector de `N²` enteros en `[0, |A|)` |

Diferencia con el Ejercicio 2: acá el cromosoma es **entero y discreto**, no real. Eso
cambia los operadores: la mutación no puede ser una perturbación gaussiana con clamp,
tiene que ser un reemplazo por otro símbolo del alfabeto.

## Fitness

El mismo esquema que en el Ejercicio 2, y por las mismas razones:

1. Se renderiza el individuo con una fuente monoespaciada a un bitmap del tamaño de la
   imagen original.
2. Se compara píxel a píxel: `fitness = 1 - RMSE/255`, en `[0,1]` y a maximizar.

Renderizar la matriz completa en cada evaluación sería carísimo. Como la fuente es
monoespaciada, el bitmap de cada carácter se puede **precomputar una sola vez**, y con
eso armar una tabla de error `N × N × |A|`: cuánto error aporta poner el carácter `a`
en la celda `(i,j)`. Evaluar un individuo pasa a ser sumar `N²` lookups en vez de
rasterizar texto.

## Operadores

- **Cruza**: uniforme o de dos puntos sobre el vector de `N²` genes. Pero conviene
  además una **cruza 2D**: intercambiar un rectángulo de la matriz entre los padres.
  Preserva la vecindad espacial, que es donde está la información — es el mismo
  razonamiento que nos llevó a la cruza "por triángulo" en el Ejercicio 2.
- **Mutación**: reemplazar el carácter de una celda por otro del alfabeto. Una variante
  dirigida —reemplazar por un carácter de **densidad de tinta parecida**— converge
  mucho más rápido que elegir uniformemente, porque no destruye el tono de la celda.
- **Inicialización informada**: asignarle a cada celda el carácter cuya densidad de
  tinta más se acerca al brillo medio de su bloque. Arranca cerca del óptimo en vez de
  arrancar en ruido.

## El punto importante: acá el AG no hace falta

El problema es **separable**. Cada celda `(i,j)` afecta exactamente a su propio bloque
de píxeles y a ningún otro: no hay superposición. Entonces el error total es la suma
de los errores por celda, y **el óptimo global se obtiene celda por celda**, probando
los `|A|` caracteres y quedándose con el mejor. Son `N² × |A|` evaluaciones y da la
solución exacta — un AG sólo puede empatarla, nunca mejorarla, y con más cómputo.

El AG recupera sentido en cuanto se rompe la separabilidad, por ejemplo:

- penalizar cambios bruscos entre celdas vecinas (que el resultado se "lea" bien);
- usar una métrica perceptual global en vez del error por píxel;
- limitar cuántos caracteres distintos se pueden usar en total;
- permitir que el mapeo carácter → tono sea parte del individuo.

Esto es justamente lo que hace interesante al Ejercicio 2: los triángulos **sí**
interactúan entre sí —se superponen, el orden importa, el alfa mezcla colores— así que
el problema no se descompone y un método de búsqueda global tiene algo que aportar.
