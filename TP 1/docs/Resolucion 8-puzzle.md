# Ejercicio 1: 8-puzzle

## 1. Datos iniciales y modelado del problema

El 8-puzzle está formado por ocho fichas numeradas y un espacio vacío en un tablero de
3 × 3. En cada paso se puede intercambiar el espacio vacío con una ficha ubicada arriba,
abajo, a la izquierda o a la derecha. Todos los movimientos tienen costo 1; por lo tanto,
el costo de una solución es su cantidad de movimientos.

### Representación del estado

Se representa cada tablero mediante una tupla inmutable de nueve enteros, recorrida por
filas. El valor `0` representa el espacio vacío. Por ejemplo, el tablero inicial mostrado
en el enunciado se representa como:

```text
(5, 7, 3,
 8, 2, 0,
 1, 6, 4)
```

Esta representación es compacta y _hasheable_, por lo que permite almacenar los estados
en conjuntos y diccionarios y consultar si un tablero ya fue visitado en tiempo promedio
O(1).

La posición del vacío puede guardarse junto al nodo de búsqueda para no tener que
recorrer la tupla en cada expansión. Los datos propios del algoritmo —estado padre,
movimiento aplicado, profundidad y costos `g`, `h` y `f`— pertenecen al nodo de búsqueda
y no a la identidad del estado.

### Estado inicial, acciones y transición

- **Estado inicial:** el tablero aleatorio recibido por el problema.
- **Acciones:** mover el espacio vacío hacia arriba, abajo, izquierda o derecha, siempre
  que no se salga del tablero.
- **Transición:** intercambiar el `0` con la ficha adyacente en la dirección elegida.
- **Costo:** cada acción cuesta 1.
- **Factor de ramificación:** según la posición del vacío, un estado tiene entre 2 y 4
  sucesores.

### Estados objetivo

El enunciado presenta las siguientes tres configuraciones objetivo:

```text
G1 = (1, 2, 3,    G2 = (3, 2, 1,    G3 = (3, 6, 0,
      4, 5, 6,          6, 5, 4,          2, 5, 8,
      7, 8, 0)          0, 8, 7)          1, 4, 7)
```

Estas configuraciones representan el mismo patrón geométrico con orientaciones
distintas. En particular, `G3` se obtiene rotando `G1` 90° en sentido antihorario,
mientras que `G2` se obtiene reflejando `G1` respecto de su eje vertical. Por lo tanto,
las tres pertenecen a la misma órbita de simetría del cuadrado, descrita por el grupo
diédrico `D4`.

Sin embargo, una rotación o una reflexión completa del tablero no es una acción válida
del 8-puzzle: las únicas acciones permitidas son los intercambios del espacio vacío con
una ficha adyacente. Por eso `G1`, `G2` y `G3` siguen siendo tres estados diferentes para
el algoritmo y deben aparecer individualmente en la prueba de meta.

La prueba de meta consiste en verificar si el estado actual pertenece al conjunto de
metas alcanzables. Al tratarse de una colección pequeña, esta consulta se realiza en
tiempo promedio O(1) usando un `set`.

## 2. Simetrías y alcanzabilidad por paridad

No todas las configuraciones de un 8-puzzle pueden alcanzarse entre sí. Para un tablero
de ancho impar, como este, cada movimiento conserva la paridad de la permutación de las
fichas. La paridad se obtiene contando las inversiones de la tupla luego de eliminar el
`0`: una inversión es un par de fichas que aparece en el orden contrario al numérico.

Las metas del enunciado tienen estas paridades:

| Meta | Inversiones | Paridad |
| --- | ---: | --- |
| `G1` | 0 | par |
| `G2` | 7 | impar |
| `G3` | 12 | par |

Aunque geométricamente las tres metas son versiones rotadas o reflejadas del mismo
patrón, desde el punto de vista de los movimientos legales quedan separadas en dos
clases de alcanzabilidad:

```text
Clase par:   {G1, G3}
Clase impar: {G2}
```

`G1` puede transformarse en `G3` mediante una secuencia de movimientos legales porque
ambas configuraciones tienen la misma paridad. En cambio, ninguna secuencia de
movimientos permite transformar `G1` o `G3` en `G2`, ya que `G2` tiene la paridad
opuesta. En este sentido técnico puede hablarse de **dos tipos de solución según su
alcanzabilidad**, aunque el enunciado muestre tres tableros objetivo concretos.

Antes de iniciar la búsqueda se calcula la paridad del estado inicial y se descartan las
metas de paridad opuesta. En consecuencia:

- si el tablero inicial tiene paridad par, las metas alcanzables son `G1` y `G3`;
- si tiene paridad impar, la única meta alcanzable es `G2`.

El tablero inicial usado como ejemplo en el enunciado posee 17 inversiones, por lo que
solo puede llegar a `G2`. Este filtro evita orientar la búsqueda hacia objetivos
imposibles. No se deben generar ocho metas mediante rotaciones y reflexiones: a
diferencia de otras versiones del ejercicio, este enunciado define tres metas explícitas.

## 3. Heurísticas admisibles

Como existen varias metas posibles, una heurística individual se calcula contra cada
meta alcanzable y luego se toma el mínimo:

```text
h(s) = min(h_G(s) para cada meta alcanzable G)
```

Esto estima el costo hasta la meta válida más cercana y conserva la admisibilidad.
El espacio vacío no se incluye en ninguna de las heurísticas.

### 3.1. Cantidad de fichas fuera de lugar

Para cada meta `G`, se cuenta cuántas fichas no están en su posición objetivo:

```text
h_fuera_G(s) = cantidad de fichas t tales que posición_s(t) != posición_G(t)
```

Es admisible porque cada ficha fuera de lugar deberá moverse al menos una vez antes de
alcanzar esa meta. También es consistente: un movimiento solo puede modificar en una
unidad la cantidad de fichas correctamente ubicadas.

### 3.2. Distancia Manhattan

Para cada ficha se suma la distancia horizontal y vertical entre su posición actual y
la posición que ocupa en la meta:

```text
h_Manhattan_G(s) = Σ_t (|fila_s(t) - fila_G(t)|
                         + |col_s(t) - col_G(t)|)
```

Es admisible porque un movimiento desplaza una única ficha una sola celda y la
heurística ignora los obstáculos producidos por las demás fichas. Por ello nunca puede
superar la cantidad real de movimientos necesarios. Es más informada que la cantidad
de fichas fuera de lugar, ya que una ficha alejada aporta más de una unidad.

### 3.3. Distancia Manhattan con conflictos lineales

Se puede mejorar la heurística anterior detectando pares de fichas que están en su fila
o columna objetivo, pero aparecen en el orden inverso al requerido por una meta. Aunque
sus distancias Manhattan parezcan independientes, una de ellas deberá apartarse y
regresar para que ambas puedan ordenarse, lo que exige al menos dos movimientos extra.

```text
h_CL_G(s) = h_Manhattan_G(s) + 2 × cantidad_de_conflictos_lineales_independientes
```

Para no contar de más, se selecciona un conjunto de conflictos sin superposición, es
decir, ninguna ficha participa en más de uno. Así definida, la heurística es admisible y
domina a Manhattan. Para las tres metas se utiliza nuevamente
`min_G h_CL_G(s)`.

## 4. Métodos de búsqueda elegidos

### A* como método principal

El método recomendado es **A\*** en su variante de búsqueda en grafo, con prioridad:

```text
f(n) = g(n) + h(n)
```

Se usaría distancia Manhattan con conflictos lineales; Manhattan sola constituye una
alternativa más sencilla. Como el costo de cada movimiento es 1 y las heurísticas
propuestas son admisibles, A* encuentra una solución de costo mínimo. Para preservar
esta garantía, la implementación debe almacenar el mejor `g` conocido de cada estado y
actualizar o reabrir un estado si se descubre un camino más barato.

### BFS como referencia

**BFS** también es completa y óptima en este problema porque todas las acciones cuestan
1. Es útil como línea de base para verificar el costo obtenido por A*, pero expande
estados sin información sobre la meta y suele consumir mucha más memoria.

### Greedy para comparar velocidad y calidad

**Greedy Best-First Search** puede usar la misma heurística y ordenar la frontera solo
por `h(n)`. Suele avanzar rápidamente hacia una meta, pero ignora el costo ya recorrido,
por lo que no garantiza una solución óptima. Resulta útil para comparar cantidad de
nodos, tiempo y calidad de la solución contra BFS y A*.

### DFS no recomendado como solucionador principal

**DFS** requiere poca memoria, pero puede internarse en ramas muy largas y devolver una
solución muy alejada de la óptima. Debe utilizar un conjunto de visitados para evitar
ciclos. Aunque el espacio del 8-puzzle es finito, no aprovecha las heurísticas ni la
estructura de costos, por lo que se incluiría solo con fines comparativos.

## 5. Relación con el motor de Sokoban implementado

La organización propuesta sigue la misma idea general de `sokoban_solver.py`:

- un estado inmutable y hasheable;
- una frontera cuya estructura depende del método;
- un conjunto de estados visitados;
- una prueba de meta al extraer un nodo;
- reconstrucción del camino como una secuencia de desplazamientos `(dx, dy)`;
- BFS, DFS, Greedy y A* dentro de un mismo motor de búsqueda.

En Sokoban, el estado es `(posición_del_jugador, frozenset(posiciones_de_cajas))` y las
heurísticas estiman la asignación de cajas a objetivos. En el 8-puzzle, el estado pasa a
ser la tupla de nueve posiciones y la correspondencia entre cada ficha y su lugar en
cada meta ya es única, por lo que no hace falta aplicar el algoritmo Húngaro.

La fórmula utilizada por el solver actual para A* es
`0.5 × h + 0.5 × g`. Esta expresión produce el mismo orden de prioridad que `g + h`,
pues solo está multiplicada por una constante positiva. Para documentar o implementar
el 8-puzzle conviene escribir directamente la forma canónica `f = g + h`.

## 6. Conclusión

La solución propuesta utiliza una tupla inmutable como estado, elimina de antemano las
metas inalcanzables mediante la paridad y ejecuta A* en grafo. La heurística principal es
la distancia Manhattan con conflictos lineales, evaluada contra todas las metas válidas
y minimizada. Esta combinación reduce el espacio explorado sin perder la garantía de
encontrar el camino con la menor cantidad posible de movimientos.
