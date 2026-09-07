# Criterios de corte — cuál elegir y por qué

> El enunciado pide **decidir y justificar** el criterio de corte. Acá se decide con
> datos: las 945 corridas del barrido se corrieron siempre hasta el tope de
> 800 generaciones, y sobre esos registros se reconstruye, para cada
> criterio candidato, **en qué generación habría disparado** y **cuánto fitness se
> habría perdido** por frenar ahí.

## Por qué se mide así y no corriendo cada criterio

Si cada criterio se midiera con su propia corrida, cada uno cortaría en un punto
distinto y las corridas no serían comparables entre sí — además de costar una tanda
completa por criterio. El motor registra por generación exactamente lo que los cinco
criterios necesitan mirar:

| Criterio | Qué mira | Columna registrada |
|---|---|---|
| máximo de generaciones | la generación actual | `generation` |
| tiempo máximo | segundos acumulados | `elapsed` |
| entorno a la solución | el mejor fitness alcanzado | `best_global_fitness` |
| **contenido** | generaciones seguidas sin mejorar | `stalled` |
| **estructura** | fracción de la población sin cambios | `share_unchanged` |

Con eso, evaluar un criterio es recorrer la serie y buscar la primera generación que
cumple la condición. Es exacto, no una simulación aproximada.

## Resultados

Sobre las 945 corridas del barrido:

| Criterio | ¿Dispara? | Generación (mediana) | Rango | Cómputo ahorrado | Fitness perdido |
|---|---|---|---|---|---|
| contenido G=20 | 5% (45 de 945) | 441 | 139–799 | 42% | 0.0083 (máx 0.0445) |
| contenido G=50 | 2% (20 de 945) | 574 | 218–785 | 30% | 0.0045 (máx 0.0195) |
| contenido G=100 | 0% (1 de 945) | 683 | 683–683 | 15% | 0.0042 (máx 0.0042) |
| estructura G=20 (ε=0.01) | 0% (2 de 945) | 602 | 559–644 | 25% | 0.0002 (máx 0.0002) |
| estructura G=50 (ε=0.01) | **nunca** (0 de 945) | — | — | — | — |
| entorno fitness≥0.85 | 98% (930 de 945) | 128 | 20–722 | 79% | 0.0571 (máx 0.1191) |
| entorno fitness≥0.90 | 64% (602 de 945) | 322 | 62–790 | 58% | 0.0251 (máx 0.0708) |
| entorno fitness≥0.95 | 5% (45 de 945) | 633 | 206–786 | 24% | 0.0060 (máx 0.0213) |

## Lectura

### 1. El criterio de estructura no dispara nunca, y hay una razón de diseño

En las 945 corridas, la fracción de la población que no cambia de una generación a
la siguiente promedia **0.607** y su máximo es **1.000**; el umbral del criterio es 0.99. Sólo el 0.44% de las
generaciones lo alcanza.

No es un error de implementación: es una consecuencia de la representación. Los genes
son **reales**, y el criterio compara genomas por igualdad exacta de bytes. Con
mutación gaussiana activa, cualquier perturbación —por chica que sea— cuenta como
cambio. En un AG binario dos individuos convergen a cadenas idénticas; acá convergen
a cadenas *parecidas*, que nunca son iguales.

**Conclusión defendible:** el criterio de estructura, tal como lo define la cátedra
(recambio de la población), **no es aplicable a una representación real con mutación
continua** salvo que se lo redefina con una tolerancia — por ejemplo, considerar dos
individuos iguales si su distancia es menor a un ε. Lo implementamos, lo medimos, y
esto es lo que encontramos.

> La métrica que sí captura el fenómeno es la **diversidad genética** (desvío promedio
> por gen), que está en todos los ejes y sí colapsa. Es el reemplazo natural del
> criterio de estructura para esta representación.

### 2. El criterio de contenido tampoco alcanza a disparar con umbrales razonables

La racha más larga sin mejorar tiene mediana **5** generaciones, percentil 90 en **11** y máximo **106**.

La causa es la forma del problema: con 500 genes reales y mutación permanente, el
algoritmo casi siempre encuentra **alguna** mejora marginal. El mejor global es
monótono y sigue subiendo de a milésimas hasta el final. Un criterio de contenido
sólo sirve acá si el umbral se define sobre la *magnitud* de la mejora y no sobre su
existencia (por ejemplo: cortar si no mejoró más de 0.001 en G generaciones).

### 3. El criterio útil es el entorno a la solución, y sirve para otra cosa

Es el único que dispara de forma consistente, pero su valor no es ahorrar cómputo
sino **fijar un objetivo de calidad**: permite responder «cuántas generaciones hacen
falta para llegar a fitness X» en vez de «qué fitness da en X generaciones». Es la
forma correcta de comparar métodos a calidad igualada, y es lo que el enunciado
menciona como opcional («error mínimo como condición de corte»).

## Qué criterio usamos, y por qué

| Criterio | Estado | Justificación |
|---|---|---|
| **máximo de generaciones** | **activo, siempre** | Es el único que garantiza terminación
y el único que hace comparables dos corridas: a presupuesto fijo, lo que se compara es
la calidad alcanzada. Todo el análisis lo usa. |
| **entorno a la solución** | disponible, se usa para medir | No para cortar antes, sino
para el análisis inverso: generaciones necesarias para alcanzar una calidad dada. |
| tiempo máximo | implementado, no se usa en el análisis | Hace las corridas no
reproducibles: el mismo experimento en otra máquina corta en otra generación. Sirve
como red de seguridad en producción, no como criterio experimental. |
| contenido | implementado, medido, **descartado** | Con umbrales razonables no dispara
(ver punto 2). Necesitaría redefinirse sobre la magnitud de la mejora. |
| estructura | implementado, medido, **descartado** | No dispara nunca con genes reales
(ver punto 1). |

Que dos de los cinco se descarten **no es un resultado negativo**: están implementados
y el motor reporta cuál cortó. Lo que se aporta es la evidencia de por qué, en este
problema y con esta representación, no son los adecuados — que es exactamente lo que
el enunciado pide justificar.

## Figuras

| Archivo | Qué muestra |
|---|---|
| `figuras/estancamiento.png` | Distribución de la racha máxima sin mejorar, con los umbrales de contenido marcados. Se ve que casi ninguna corrida los alcanza |
| `figuras/recambio_poblacion.png` | Fracción de la población sin cambios por generación, contra el umbral del criterio de estructura |
| `figuras/costo_beneficio.png` | Cómputo ahorrado contra fitness perdido, por criterio |

## Datos

- `datos/criterios.csv` — la tabla de arriba, en CSV
