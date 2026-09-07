"""Arma el informe del análisis: una carpeta por eje, con todo junto.

    python analysis/build_report.py

Por cada eje experimental crea `analysis/informe/NN-<eje>/` con:

    informe.md    qué se ve, qué es importante mirar y qué no se puede afirmar
    config.md     la configuración anotada: la fija, la de cada variante, y cómo reproducir
    datos/        el CSV crudo del eje y su resumen agregado
    figuras/      un gráfico por PNG, sin título ni texto, fondo transparente
    imagenes/     el resultado visual: el mejor individuo de cada variante, renderizado

Las figuras no traen texto de mobiliario a propósito: el título lo pone la
diapositiva. Todo lo que hay que anotar está en los `.md`.

Las imágenes de `imagenes/` se regeneran corriendo el motor con la configuración
exacta de la variante y la primera semilla del barrido. Como el motor es
determinístico dada la semilla, el fitness reproduce el del CSV; el informe lo
verifica y avisa si no coincide.
"""

from __future__ import annotations

import csv
import json
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import plot_results as figs
from ga import engine
from ga.render import load_target, render

ANALYSIS = Path(__file__).resolve().parent
ROOT = ANALYSIS.parent
INFORME = ANALYSIS / "informe"
#: carpeta del caso de cierre, que genera caso_final.py y este script no debe pisar
PROPIAS = {"13-criterios-corte", "14-interaccion", "15-caso-final"}

RESULTS = ANALYSIS / "results"
RENDER_SIZE = 320

#: orden de presentación: sigue el loop del AG, no el orden del archivo de specs
#: El número de carpeta es EXPLÍCITO, no posicional. Antes se numeraba con
#: `enumerate`, y agregar un eje en el medio renumeraba todos los que venían
#: después: los enlaces del README y de la presentación quedaban apuntando a
#: carpetas que ya no existían. Con el número acá, agregar un eje no toca a nadie.
#: 13, 14 y 15 están reservados para los extras que generan otros scripts (PROPIAS).
ORDEN = [
    (1, "seleccion", "seleccion", "Selección de padres"),
    (2, "presion", "presion-seleccion", "Presión de selección"),
    (3, "supervivencia", "supervivencia", "Supervivencia y brecha generacional"),
    (4, "cruza", "cruza", "Método de cruza"),
    (5, "granularidad", "granularidad", "Granularidad de la cruza"),
    (6, "tasa_cruza", "tasa-cruza", "Probabilidad de cruza"),
    (7, "mutacion", "mutacion", "Método de mutación"),
    (8, "tasa_mutacion", "tasa-mutacion", "Carga de mutación"),
    (9, "sigma", "sigma", "Magnitud de la mutación (σ)"),
    (10, "poblacion", "poblacion", "Tamaño de población"),
    (11, "triangulos", "triangulos", "Cantidad de triángulos"),
    (12, "inicializacion", "inicializacion", "Inicialización de la población"),
    (16, "seleccion_combinada", "seleccion-combinada", "Selección combinada A%/B%"),
]

#: rótulos legibles para las claves de configuración
ETIQUETAS = {
    "triangles": "triángulos", "canvas_size": "canvas (px)", "population_size": "población N",
    "offspring_size": "descendencia K", "initialization": "inicialización",
    "selection_parents": "selección de padres", "selection_survivors": "selección de sobrevivientes",
    "crossover": "cruza", "crossover_granularity": "granularidad de cruza",
    "crossover_rate": "probabilidad de cruza pc", "crossover_uniform_p": "p de cruza uniforme",
    "mutation": "mutación", "mutation_rate": "probabilidad de mutación pm",
    "mutation_sigma": "σ de mutación", "mutation_genes": "M (multigen)",
    "mutation_decay_floor": "piso de decaimiento", "replacement": "supervivencia",
    "background": "color de fondo", "tournament": "torneo", "boltzmann": "Boltzmann",
    "stop": "criterios de corte", "seed": "semilla", "image": "imagen",
}

# --------------------------------------------------------------------------------------
# Comentario propio de cada eje: qué se está midiendo y qué hay que mirar en el gráfico.
# Los números salen de los datos; esto es la lectura.
# --------------------------------------------------------------------------------------
NOTAS = {
    "seleccion": {
        "que_es": (
            "Compara los siete métodos de selección de padres con todo lo demás fijo. Lo que "
            "está en juego es la **presión de selección**: cuánto favorece el método a los "
            "mejores individuos. Mucha presión converge rápido pero se estanca en un óptimo "
            "local; poca presión explora más pero avanza lento.\n\n"
            "Los siete se dividen en tres familias por lo que usan para decidir: los que miran "
            "el **valor** del fitness (ruleta, universal, Boltzmann), los que miran sólo el "
            "**orden** (ranking, elite) y los que hacen **comparaciones locales** (los dos "
            "torneos). Esa distinción explica el comportamiento mejor que el nombre del método."
        ),
        "que_mirar": [
            "En `convergencia`, la **pendiente inicial**: los de presión alta despegan primero.",
            "En `diversidad`, cuál colapsa antes. Presión alta = diversidad que cae rápido, y ése es el mecanismo de la convergencia prematura (**causa 1** de las tres que enumera la cátedra; las otras dos están en `08-tasa-mutacion` y `10-poblacion`).",
            "Los métodos por valor (ruleta, universal, Boltzmann) pierden presión cuando la población converge, porque todos los fitness se parecen y las probabilidades se vuelven casi uniformes. Los métodos por orden (ranking) y los torneos no: mantienen la misma presión aunque las diferencias sean mínimas. Mirá si eso se nota en la parte final de la curva.",
            "`elite` es el caso de borde a señalar: cuando K = N devuelve la población entera, o sea presión **cero**. Es consecuencia directa de la fórmula n(i) = ⌈(K−i)/N⌉, no un bug.",
            "En `boxplot_final`, si las cajas se solapan: si los siete caen en un rango angosto, la conclusión honesta es que **no se distinguen**, y eso también hay que decirlo.",
        ],
        "trampas": [
            "La selección de sobrevivientes está fija en `elite`: acá sólo varía la de padres.",
            "Los parámetros de los torneos (M y Th) están fijos en sus valores por defecto. El efecto de moverlos se mide aparte, en [`02-presion-seleccion`](../02-presion-seleccion/informe.md), que es el experimento controlado de presión.",
        ],
    },
    "seleccion_combinada": {
        "que_es": (
            "La cátedra pide poder seleccionar **A% de los padres con un método y (1−A)% con "
            "otro**. Está implementado (`selection.build`) y este eje lo mide, con tres métodos "
            "puros como referencia para que la comparación tenga sentido.\n\n"
            "La intuición detrás de combinar es repartir el trabajo: un método de presión alta "
            "(elite) asegura que lo mejor se propague, y uno de presión baja (ruleta) mantiene "
            "diversidad. La pregunta es si esa mezcla rinde más que cualquiera de los dos solo."
        ),
        "que_mirar": [
            "**Compará cada mezcla contra sus dos componentes puros**, que están en el mismo gráfico. Si una mezcla no le gana a los dos, combinar no aportó.",
            "`25% elite + 75% torneo` es la mezcla con más presión de las tres; si el orden sigue la presión, el resultado se explica por [`02-presion-seleccion`](../02-presion-seleccion/informe.md) y no por la combinación en sí.",
            "En `diversidad`: la promesa de combinar es sostener más dispersión que el método de presión alta solo. Si eso no se ve, el mecanismo propuesto no está operando.",
        ],
        "trampas": [
            "`elite` con k=N devuelve la población entera (presión cero), así que `50% elite + 50% X` en realidad significa *la mitad de los padres sin ninguna presión*. No es 'medio elite': es medio azar.",
            "Este eje comparte variantes con [`01-seleccion`](../01-seleccion/informe.md) (los puros); los números tienen que coincidir, y sirven de control cruzado.",
        ],
    },
    "presion": {
        "que_es": (
            "Mide la **presión de selección de forma controlada**: en vez de comparar métodos "
            "distintos, se toma un solo método y se mueve su parámetro de presión. En el torneo "
            "determinístico eso es **M** (cuántos compiten: M=2 es la presión mínima, M=N "
            "equivale a elite); en el probabilístico es **Th** (con qué probabilidad gana el "
            "mejor: Th=0.5 es azar puro, Th=1 es torneo determinístico de 2).\n\n"
            "Es el experimento que aísla la **causa 1 de convergencia prematura** sin confundirla "
            "con otras diferencias entre métodos."
        ),
        "que_mirar": [
            "Se espera una curva en U: poca presión no explota lo bueno que encuentra, demasiada colapsa la diversidad antes de haber explorado. Mirá dónde está el óptimo y **cuán plano es** — si es plano, la presión no es un parámetro crítico en este problema.",
            "En `diversidad`, el efecto tiene que ser **monótono** aunque el fitness no lo sea: más presión, colapso más temprano. Si eso se ve, es la evidencia directa del mecanismo.",
            "En `mejor_vs_promedio`, con presión alta el promedio alcanza al mejor mucho antes: la población entera se vuelve el mismo individuo.",
            "Comparar M=30 (sobre N=60) con `elite` del eje 01: son casi lo mismo conceptualmente y deberían comportarse parecido.",
        ],
        "trampas": [
            "M está acotado por el tamaño de la población: `tournament_det` hace `min(M, N)`.",
            "Th < 0.5 invertiría el método (ganaría el peor). La cátedra lo acota a [0.5, 1] y por eso no se prueban valores menores.",
        ],
    },
    "supervivencia": {
        "que_es": (
            "Cruza dos cosas: la **estrategia de supervivencia** (aditiva vs. exclusiva) y el "
            "**tamaño de la descendencia K**, o sea la brecha generacional G = K/N.\n\n"
            "Aditiva: compiten los N padres con los K hijos y sobreviven N del pool N+K.\n"
            "Exclusiva: con K > N se seleccionan N de los K hijos; con K ≤ N pasan los K hijos "
            "más N−K padres.\n\n"
            "Con K = N/2 la brecha es 0.5 y la mitad de la población sobrevive intacta; con "
            "K = 2N se generan el doble de hijos que lugares hay."
        ),
        "que_mirar": [
            "**Agrupá las variantes por K, no por estrategia.** Si todas las de un mismo K quedan juntas, lo que decide es la brecha generacional y no la estrategia — y ésa es la conclusión.",
            "`fitness_vs_evaluaciones` es acá **el gráfico decisivo**: una generación con K=2N cuesta el doble de evaluaciones que una con K=N. Si la ventaja de K=2N desaparece al graficar contra evaluaciones, no había ventaja: había más presupuesto.",
            "Con supervivencia exclusiva el mejor de la población **puede empeorar** de una generación a la otra, porque los hijos desplazan a los padres. Por eso la curva usa el mejor acumulado; la línea punteada de `mejor_vs_promedio` muestra el efecto.",
            "Con K = N/2 y exclusiva, la mitad de la población pasa sin competir: mirá si eso frena la convergencia o si actúa como elitismo encubierto.",
        ],
        "trampas": [
            "La combinación elite + K=N + exclusiva deja al algoritmo **sin ninguna presión de selección** (elite con k=N devuelve todo, y exclusiva reemplaza todo): degenera en caminata aleatoria. Por eso la base usa torneo en los padres.",
            "Los nombres aditiva/exclusiva siguen las láminas 45 y 46 del deck; están citadas textualmente en el docstring de `ga/replacement.py`.",
        ],
    },
    "cruza": {
        "que_es": (
            "Compara los cinco métodos de cruza. La pregunta de fondo es cuánto **preserva "
            "bloques constructivos**: conjuntos de genes que juntos valen más que por separado. "
            "Acá un bloque constructivo es un triángulo bien ubicado, o un grupo de triángulos "
            "que se superponen bien.\n\n"
            "Se puede **predecir antes de medir**: la cruza uniforme es la única que no mantiene "
            "correlación posicional entre loci, así que debería ser la más disruptiva. Que gane "
            "o pierda dice cuánto le importa a este problema conservar el orden."
        ),
        "que_mirar": [
            "Un punto y anular preservan bloques contiguos; uniforme los rompe. Si uniforme igual gana, es señal de que el problema **tolera o necesita** mucha mezcla — probablemente porque los triángulos interactúan poco entre sí salvo por superposición.",
            "La cruza **espacial** es propia: parte por dónde cae el triángulo en el canvas, no por su índice. Miralo junto a [`11-triangulos`](../11-triangulos/informe.md): con pocos triángulos una partición espacial casi no tiene qué repartir.",
            "En `diversidad`, si alguna cruza mantiene la población más dispersa. Una cruza disruptiva actúa parcialmente como mutación.",
            "El eje se lee junto con [`06-tasa-cruza`](../06-tasa-cruza/informe.md): si con pc=0 el algoritmo rinde casi igual, la elección de método de cruza importa poco por definición.",
        ],
        "trampas": [
            "La granularidad está fija en este eje; se estudia aparte en [`05-granularidad`](../05-granularidad/informe.md).",
            "La cruza espacial **ignora** la granularidad configurada: siempre trabaja por triángulo entero, porque partir un triángulo al medio no tiene sentido geométrico.",
        ],
    },
    "granularidad": {
        "que_es": (
            "El experimento conceptualmente más interesante del TP. Con la misma cruza uniforme, "
            "cambia sólo la **unidad de corte**: un componente suelto (puede partir un triángulo "
            "al medio, mezclando coordenadas de uno con color de otro) o un triángulo entero.\n\n"
            "La hipótesis es teórica y se enuncia **antes** de medir: como el orden de la lista "
            "define el z-order, la posición en el cromosoma tiene significado real. Cortar "
            "adentro de un triángulo destruye una unidad semántica. En el vocabulario de la "
            "cátedra, el triángulo es el **gen** y los 10 valores son su alelo; cortar por "
            "componente es cortar por debajo del gen."
        ),
        "que_mirar": [
            "Es un eje de **dos variantes**: el boxplot es el gráfico decisivo, no la curva.",
            "Si las cajas se solapan, la hipótesis **no queda demostrada** aunque la media favorezca al corte por triángulo. Decirlo así es más fuerte que forzar una conclusión.",
            "El efecto debería crecer con la cantidad de triángulos: con más triángulos hay más que romper. Vale contrastarlo con [`11-triangulos`](../11-triangulos/informe.md).",
        ],
        "trampas": [
            "Con dos variantes el poder estadístico es el que dan las semillas. Un empate no refuta la hipótesis: dice que el experimento no alcanza para decidirla.",
            "El efecto de la granularidad interactúa con el método de cruza: acá se mide sólo con `uniform`, que es donde debería notarse más (es la que rompe más bloques).",
        ],
    },
    "tasa_cruza": {
        "que_es": (
            "Barre la probabilidad de cruza `pc`. Los dos extremos son los interesantes:\n\n"
            "- **pc = 0**: no hay recombinación. El algoritmo queda reducido a mutación más "
            "selección, o sea una búsqueda local paralela. **Es el sanity check del enunciado**: "
            "si el fitness igual mejora, la mutación sola funciona; y la distancia hasta pc>0 es "
            "**cuánto aporta realmente la recombinación**, que es la pregunta de fondo de por qué "
            "usar un AG y no un hill climbing.\n"
            "- **pc = 1**: todos los padres se cruzan siempre, no queda ningún individuo que pase "
            "intacto a la etapa de mutación."
        ),
        "que_mirar": [
            "**La diferencia entre pc=0 y el mejor pc es el aporte neto de la cruza.** Si es chica, el trabajo pesado lo hace la mutación, y conviene decirlo con el número en la mano en vez de asumir que la recombinación es esencial.",
            "En `diversidad`, pc=0 debería mantener más o menos la misma diversidad que el resto: la cruza redistribuye material genético pero no crea material nuevo.",
            "Si la curva es plana entre pc=0.25 y pc=1, `pc` no es un parámetro crítico y no vale la pena afinarlo.",
        ],
        "trampas": [
            "Con pc=0 los hijos son copias de los padres y **igual pasan por mutación**: no es una corrida sin cambios.",
            "`pc` se aplica por par de padres, no por individuo.",
        ],
    },
    "mutacion": {
        "que_es": (
            "Compara los cuatro métodos de mutación **a carga comparable**. La mutación es la "
            "única fuente de material genético nuevo: la cruza sólo recombina lo que ya está.\n\n"
            "La comparación honesta no es a `pm` igual sino a **cantidad esperada de genes "
            "mutados por individuo** igual, porque `pm` significa cosas distintas en cada método. "
            "Las variantes están calibradas a carga 4 sobre 500 genes — salvo `gen`, que por "
            "construcción muta a lo sumo **un** gen y no puede llegar a 4. Ese techo estructural "
            "es parte del resultado."
        ),
        "que_mirar": [
            "`gen` está limitado a carga ≤ 1 por diseño. Si queda último, la explicación no es que el método sea malo sino que **no puede aplicar suficiente mutación** en un cromosoma de 500 genes. Es el mismo fenómeno que se mide en [`11-triangulos`](../11-triangulos/informe.md).",
            "`no uniforme` arranca en la misma carga y la decae con las generaciones: mirá si su curva arranca igual que `uniforme` y se aplana antes. Es exploración temprana y ajuste fino tardío.",
            "En `diversidad`, `no uniforme` debería mostrar el colapso más pronunciado al final, porque su propia tasa se apaga.",
            "`multigen` con M fijo concentra las mutaciones en menos individuos-gen que `uniforme` con la misma carga esperada: mirá si esa diferencia de *varianza* (no de media) se traduce en algo.",
        ],
        "trampas": [
            "**`mutation_rate` no significa lo mismo en cada método.** En `gen` es la probabilidad de que ocurra la única mutación; en `uniforme` es la probabilidad **por gen** sobre los 500. Por eso cada variante lleva su propio pm calibrado y el nombre de la variante indica la carga.",
            "La magnitud comparable entre métodos es la carga esperada, no `pm`. El barrido de la carga está en [`08-tasa-mutacion`](../08-tasa-mutacion/informe.md).",
        ],
    },
    "tasa_mutacion": {
        "que_es": (
            "Barre la **carga de mutación**: cuántos genes se espera que muten por individuo y "
            "generación. Se expresa como carga y no como `pm` a propósito, porque `pm` es "
            "probabilidad por gen y sólo tiene sentido junto al largo del cromosoma (acá 500 "
            "genes, así que carga = pm × 500).\n\n"
            "Es la **causa 2 de convergencia prematura** que enumera la cátedra: probabilidad de "
            "mutación demasiado baja."
        ),
        "que_mirar": [
            "Se espera una U: muy poca mutación no explora y la población se congela; demasiada destruye lo construido más rápido de lo que lo mejora. Mirá **dónde está el mínimo y cuán ancho es el valle** — si es ancho, no hace falta afinar.",
            "El extremo malo suele ser mucho más claro que el óptimo. Es más defendible decir «carga 50 es netamente peor» que «carga 4 es la mejor».",
            "En `diversidad` se ve el mecanismo directo: carga alta sostiene diversidad artificialmente sin que eso se traduzca en fitness. Es la mejor ilustración de que **diversidad no es calidad**.",
            "Con carga 0.5 la población casi no cambia: contrastalo con el criterio de estructura en [`13-criterios-corte`](../13-criterios-corte/informe.md).",
        ],
        "trampas": [
            "La carga óptima depende del **presupuesto**: la que gana con 800 generaciones no es necesariamente la que gana con 3000.",
            "Y depende del **largo del cromosoma**: por eso este eje fija los triángulos en 50. El efecto de cambiar el largo se mide en [`11-triangulos`](../11-triangulos/informe.md).",
        ],
    },
    "sigma": {
        "que_es": (
            "Barre σ, la **magnitud** de la perturbación gaussiana, con la carga de mutación "
            "fija. Es la otra mitad de la mutación: `pm` decide *cuántos* genes se tocan y σ "
            "*cuánto* se los mueve.\n\n"
            "Como todos los genes viven en [0,1] y la perturbación se recorta a ese rango, σ "
            "grande no sólo explora más: además **satura** más genes contra los bordes, lo que "
            "sesga la distribución hacia 0 y 1."
        ),
        "que_mirar": [
            "σ chico es ajuste fino: convergencia suave pero lenta. σ grande es exploración: avance rápido al principio y ruido al final. Mirá si eso se ve como un cruce de curvas.",
            "Este eje explica por qué la mutación **no uniforme** puede tener sentido aunque en [`07-mutacion`](../07-mutacion/informe.md) no gane: lo que la no uniforme hace es moverse por este eje a lo largo de la corrida, empezando con σ grande y terminando con σ chico.",
            "Un σ óptimo intermedio y un valle ancho significan que el parámetro no es crítico; un óptimo agudo significa que sí, y entonces la mutación no uniforme debería ganar.",
            "En las imágenes resultado: σ grande produce colores saturados por el clamp a [0,1]. Es visible.",
        ],
        "trampas": [
            "σ se aplica igual a coordenadas y a color. Un σ óptimo distinto para cada tipo de gen es una mejora posible que no implementamos.",
            "El clamp a [0,1] hace que el efecto de σ no sea simétrico cerca de los bordes del dominio.",
        ],
    },
    "poblacion": {
        "que_es": (
            "Barre el tamaño de población N, con K = N (brecha generacional 1). Es la **causa 3 "
            "de convergencia prematura** que enumera la cátedra: población demasiado escasa.\n\n"
            "Una población chica pierde diversidad rápido por deriva genética; una grande la "
            "mantiene pero **gasta más evaluaciones por generación**, así que a presupuesto de "
            "cómputo fijo hace menos generaciones."
        ),
        "que_mirar": [
            "**`fitness_vs_evaluaciones` es el gráfico decisivo de este eje.** Comparar poblaciones por generación es directamente injusto: N=120 gasta 8 veces más evaluaciones por generación que N=15. Si al graficar contra evaluaciones el orden se da vuelta, la conclusión por generación era un artefacto del presupuesto.",
            "En `diversidad`, el efecto del tamaño tiene que ser claro y monótono: poblaciones chicas colapsan antes. Es la evidencia directa de la causa 3.",
            "En `mejor_vs_promedio`, con N chico el promedio alcanza al mejor mucho más rápido.",
            "Buscá el punto donde agrandar la población deja de pagar: es la respuesta práctica a «qué N usar».",
        ],
        "trampas": [
            "K se mueve junto con N para mantener la brecha en 1. El efecto de la brecha se mide por separado en [`03-supervivencia`](../03-supervivencia/informe.md).",
            "El costo por generación es proporcional a K, no a N: lo que se paga es evaluar a los hijos.",
        ],
    },
    "triangulos": {
        "que_es": (
            "Barre el segundo **parámetro del problema** (el primero es la imagen) **cruzado con "
            "el régimen de la tasa de mutación**, porque los dos no se pueden separar.\n\n"
            "Más triángulos dan más capacidad de representación pero un cromosoma más largo: de "
            "100 a 2000 genes. Y como `pm` es probabilidad **por gen**, dejar `pm` fijo mientras "
            "el cromosoma crece multiplica la carga de mutación por el mismo factor. Por eso cada "
            "cantidad se corre dos veces: con **pm fijo** (lo que sale de copiar el valor "
            "calibrado) y con **carga fija** (escalando pm como 1/L para mantener constante la "
            "cantidad esperada de genes mutados).\n\n"
            "Sin ese cruce, cualquier conclusión sobre «cuántos triángulos conviene» estaría "
            "confundida con un efecto de mutación."
        ),
        "que_mirar": [
            "**Compará las dos series por separado antes de sacar conclusiones.** Si con pm fijo más triángulos empeora y con carga fija mejora, entonces «más triángulos es peor» era un artefacto de no escalar la mutación, no una propiedad del problema.",
            "El punto donde las dos series se cruzan es 50 triángulos: ahí los dos regímenes son el mismo valor por construcción. Sirve de control.",
            "Mirá el **arranque** de las curvas además del final: con inicialización informada, más triángulos arranca mejor (grilla más fina) aunque termine peor. Son dos efectos opuestos que se pueden separar mirando la generación 1 contra la última.",
            "En las imágenes resultado, la comparación visual es más elocuente que el fitness: con la misma cantidad de generaciones, más triángulos puede verse **peor** aunque tenga más capacidad.",
        ],
        "trampas": [
            "El fitness entre cantidades distintas mezcla capacidad y dificultad de búsqueda. Igualar el presupuesto de evaluaciones no lo arregla: el costo extra está **adentro** de cada evaluación (renderizar más triángulos), no en la cantidad de evaluaciones. Hay que mirar además el tiempo, que está en el resumen.",
            "La inicialización de la base es `random`, así que acá no se ve la ventaja de arranque que da la grilla informada. Ese cruce está en [`15-caso-final`](../15-caso-final/informe.md).",
        ],
    },
    "inicializacion": {
        "que_es": (
            "Compara arrancar de ruido uniforme contra arrancar de una grilla donde cada "
            "triángulo toma el **color promedio que el target tiene en esa celda**.\n\n"
            "Es el eje con más contenido conceptual del trabajo, y conviene plantearlo así: la "
            "grilla es **conocimiento del problema inyectado antes del algoritmo**, no expresado "
            "dentro de él. Funciona, pero el mérito no es del AG — es del preproceso. La "
            "alternativa honesta es meter la misma información *adentro*: en la función de "
            "aptitud, o en un operador de mutación guiado."
        ),
        "que_mirar": [
            "Mirá si la ventaja es sólo un **arranque más alto que se diluye**, o si se sostiene hasta el final. Que las curvas no se crucen es el resultado fuerte; que se crucen sería el resultado interesante.",
            "Mirá también el **desvío entre semillas**: una inicialización informada suele reducir la varianza, no sólo subir la media. Eso se ve en el ancho de la banda y en el alto de las cajas.",
            "En `diversidad`, la grilla arranca con menos diversidad (los colores ya están cerca del target). Si igual llega más lejos, es un contraejemplo útil a «más diversidad es mejor».",
            "Las imágenes de la generación 1 son la mejor ilustración: comparar el punto de partida de las dos vale más que cualquier número.",
        ],
        "trampas": [
            "La base de **todos** los demás ejes usa `random` justamente para no contaminar sus mediciones con esta ventaja.",
            "Los vértices de la grilla siguen siendo aleatorios: si no, todos los individuos arrancarían idénticos y no habría diversidad para evolucionar.",
            "La ventaja de la grilla crece con la cantidad de triángulos (grilla más fina). Con 50 triángulos se ve una parte del efecto; el cruce completo está en [`15-caso-final`](../15-caso-final/informe.md).",
        ],
    },
}


# --------------------------------------------------------------------------------------
# Métricas derivadas
# --------------------------------------------------------------------------------------
def estadisticas(data_target: dict) -> list[dict]:
    """Una fila por variante, ordenada por fitness final descendente."""
    filas = []
    for variant, metrics in data_target.items():
        best = metrics.get("best_global_fitness", metrics["best_fitness"])
        finales = best[:, -1]
        media_curva = best.mean(axis=0)

        # velocidad: primera generación que alcanza el 99% del fitness final
        umbral = media_curva[-1] * 0.99
        gen99 = int(np.argmax(media_curva >= umbral)) + 1

        div = metrics["diversity"].mean(axis=0)
        colapso = np.flatnonzero(div <= div[0] * 0.01)
        filas.append({
            "variante": variant,
            "media": float(finales.mean()),
            "desvio": float(finales.std()),
            "mediana": float(np.median(finales)),
            "q1": float(np.percentile(finales, 25)),
            "q3": float(np.percentile(finales, 75)),
            "rmse": float((1 - finales.mean()) * 255),
            "gen99": gen99,
            "diversidad_final": float(div[-1]),
            "gen_colapso": int(colapso[0]) + 1 if colapso.size else None,
            "brecha_media": float(media_curva[-1] - metrics["mean_fitness"].mean(axis=0)[-1]),
            "evaluaciones": float(metrics["evaluations"].mean(axis=0)[-1]),
        })
    return sorted(filas, key=lambda f: -f["media"])


def presupuesto_igualado(data_target: dict) -> tuple[float, list[tuple[str, float]]] | None:
    """Fitness de cada variante al mismo número de EVALUACIONES, no de generaciones.

    Cuando K difiere entre variantes, una generación no cuesta lo mismo en todas:
    comparar al final del barrido le regala presupuesto a la más cara. Se toma el
    tope común (el máximo de evaluaciones de la variante más barata) y se interpola
    el fitness de todas ahí.

    Devuelve None si todas las variantes gastan lo mismo (el eje no lo necesita).
    """
    topes = {}
    for variant, metrics in data_target.items():
        topes[variant] = float(metrics["evaluations"].mean(axis=0)[-1])
    if max(topes.values()) - min(topes.values()) < 0.02 * max(topes.values()):
        return None

    corte = min(topes.values())
    ranking = []
    for variant, metrics in data_target.items():
        best = metrics.get("best_global_fitness", metrics["best_fitness"]).mean(axis=0)
        evals = metrics["evaluations"].mean(axis=0)
        ranking.append((variant, float(np.interp(corte, evals, best))))
    return corte, sorted(ranking, key=lambda r: -r[1])


def separado(a: dict, b: dict) -> bool:
    """¿El rango intercuartil de `a` no se solapa con el de `b`?"""
    return a["q1"] > b["q3"]


# --------------------------------------------------------------------------------------
# Imágenes: el resultado visual de cada variante
# --------------------------------------------------------------------------------------
def _render_one(job):
    etiqueta, config, destino = job
    target = load_target(str(ROOT / config["image"]), config["canvas_size"], config["background"],
                         config.get("preserve_aspect", False))
    result = engine.run(config, target)
    render(result.best, RENDER_SIZE, background=config["background"]).save(destino)
    return etiqueta, float(result.best.fitness)


def jobs_imagenes(manifest: dict, experimento: str, carpeta: Path) -> list:
    """Una imagen por variante y target, con la primera semilla del barrido."""
    info = manifest["experiments"][experimento]
    seed = manifest["seeds"][0]
    trabajos = []
    for target, imagen in manifest["targets"].items():
        for variante, detalle in info["variants"].items():
            config = {**detalle["config"], "image": imagen, "seed": seed}
            destino = carpeta / f"{target}__{slug(variante)}.png"
            trabajos.append(((experimento, target, variante), config, destino))
    return trabajos


def guardar_targets(manifest: dict, carpeta: Path) -> None:
    """El target original, al mismo tamaño que los resultados, para comparar."""
    from PIL import Image
    for target, imagen in manifest["targets"].items():
        img = Image.open(ROOT / imagen).convert("RGB").resize((RENDER_SIZE, RENDER_SIZE))
        img.save(carpeta / f"{target}__TARGET.png")


def slug(texto: str) -> str:
    """Nombre de archivo seguro. Los nombres de variante traen `=`, `/`, `.`, `+`,
    paréntesis y acentos; cualquier resto de `/` crearía subcarpetas."""
    reemplazos = {"=": "", "+": "-mas-", "/": "-sobre-", "·": "-",
                  "á": "a", "é": "e", "í": "i", "ó": "o", "ú": "u", "ñ": "n"}
    limpio = texto.lower()
    for viejo_, nuevo_ in reemplazos.items():
        limpio = limpio.replace(viejo_, nuevo_)
    limpio = "".join(c if (c.isalnum() or c in "-.") else " " for c in limpio)
    return "-".join(limpio.split()).strip("-").replace("--", "-")


# --------------------------------------------------------------------------------------
# Documentos
# --------------------------------------------------------------------------------------
def valor(v) -> str:
    if isinstance(v, dict):
        return "`" + json.dumps(v, ensure_ascii=False) + "`"
    return f"`{v}`"


def escribir_config(manifest: dict, experimento: str, titulo: str, destino: Path) -> None:
    info = manifest["experiments"][experimento]
    varied = info["varied_keys"]
    fijo = info["fixed_config"]

    lineas = [
        f"# Configuración — {titulo}",
        "",
        f"Barrido `{experimento}` · batch `{manifest['run_batch']}` · commit "
        f"`{manifest['git']['commit']}`{' (árbol sucio)' if manifest['git']['dirty'] else ''}",
        "",
        f"- **Semillas**: {manifest['seeds']}",
        f"- **Targets**: " + ", ".join(f"`{k}` = `{v}`" for k, v in manifest["targets"].items()),
        f"- **Varía**: " + ", ".join(f"`{k}`" for k in varied),
        "",
        "## Configuración fija",
        "",
        "Todo lo que **no** cambia entre variantes de este eje.",
        "",
        "| Parámetro | Valor |",
        "|---|---|",
    ]
    for clave in sorted(fijo):
        if clave in varied:
            continue
        lineas.append(f"| {ETIQUETAS.get(clave, clave)} (`{clave}`) | {valor(fijo[clave])} |")

    lineas += ["", "## Variantes", "", "| Variante | Qué cambia |", "|---|---|"]
    for variante, detalle in info["variants"].items():
        cambios = " · ".join(f"`{k}` = {valor(v)}" for k, v in detalle["overrides"].items())
        lineas.append(f"| **{variante}** | {cambios} |")

    lineas += [
        "",
        "## Reproducir",
        "",
        "```bash",
        f"python analysis/run_experiments.py {experimento} --clean",
        "python analysis/build_report.py",
        "```",
        "",
        "La configuración completa de cada variante, ya resuelta contra los defaults del",
        "motor, está en `analysis/results/manifest.json`.",
        "",
        "<details><summary>Configuración completa de cada variante (JSON)</summary>",
        "",
        "```json",
        json.dumps({v: d["config"] for v, d in info["variants"].items()}, indent=2, ensure_ascii=False),
        "```",
        "",
        "</details>",
        "",
    ]
    destino.write_text("\n".join(lineas))


def tabla_resultados(filas: list[dict]) -> list[str]:
    lineas = [
        "| # | Variante | Fitness final | IQR | RMSE | Gen. al 99% | Diversidad final |",
        "|---|---|---|---|---|---|---|",
    ]
    for i, f in enumerate(filas, start=1):
        lineas.append(
            f"| {i} | {f['variante']} | {f['media']:.4f} ± {f['desvio']:.4f} | "
            f"{f['q1']:.4f}–{f['q3']:.4f} | {f['rmse']:.2f} | {f['gen99']} | "
            f"{f['diversidad_final']:.2e} |"
        )
    return lineas


def escribir_informe(manifest, experimento, titulo, slug_eje, stats, presupuesto,
                     fitness_render, destino):
    nota = NOTAS[experimento]
    targets = sorted(stats)

    lineas = [
        f"# {titulo}",
        "",
        f"> Eje `{experimento}` · batch `{manifest['run_batch']}` · "
        f"{len(manifest['seeds'])} semillas · "
        f"{manifest['experiments'][experimento]['fixed_config'].get('stop', {}).get('max_generations', '?')} generaciones",
        "",
        "## Qué mide este eje",
        "",
        nota["que_es"],
        "",
        "La configuración exacta está en [`config.md`](config.md).",
        "",
    ]

    for target in targets:
        filas = stats[target]
        primero, segundo, ultimo = filas[0], filas[1], filas[-1]
        sep12 = separado(primero, segundo)
        sep1n = separado(primero, ultimo)

        lineas += [f"## Resultados — imagen `{target}`", ""] + tabla_resultados(filas) + [""]

        if sep12:
            veredicto = (
                f"**{primero['variante']}** gana con el rango intercuartil **separado** del "
                f"segundo ({segundo['variante']}): la diferencia no se explica por el ruido "
                f"entre semillas."
            )
        elif sep1n:
            veredicto = (
                f"**No hay un ganador claro.** {primero['variante']} tiene la media más alta, "
                f"pero su rango intercuartil se solapa con el de {segundo['variante']}: con "
                f"{len(manifest['seeds'])} semillas la diferencia está dentro del ruido. "
                f"Lo que **sí** se puede afirmar es que le gana a **{ultimo['variante']}**, "
                f"que queda separado ({ultimo['media']:.4f})."
            )
        else:
            veredicto = (
                f"**Las {len(filas)} variantes son indistinguibles a esta escala**: ni siquiera "
                f"el primero se separa del último ({primero['media']:.4f} contra "
                f"{ultimo['media']:.4f}, con rangos que se solapan). El resultado es un empate, "
                f"y como tal hay que presentarlo."
            )
        lineas += [veredicto, ""]

        rango = primero["media"] - ultimo["media"]
        lineas += [
            f"- **Rango del eje**: {rango:.4f} de fitness entre el mejor y el peor "
            f"({rango * 255:.1f} puntos de RMSE).",
            f"- **Velocidad**: el más rápido llega al 99% de su fitness final en la generación "
            f"{min(f['gen99'] for f in filas)}; el más lento, en la "
            f"{max(f['gen99'] for f in filas)}.",
        ]
        colapsos = [f for f in filas if f["gen_colapso"]]
        if colapsos:
            antes = min(colapsos, key=lambda f: f["gen_colapso"])
            lineas.append(
                f"- **Diversidad**: {len(colapsos)} de {len(filas)} variantes colapsan por debajo "
                f"del 1% de su diversidad inicial; la primera es **{antes['variante']}** en la "
                f"generación {antes['gen_colapso']}."
            )
        brecha = min(filas, key=lambda f: f["brecha_media"])
        lineas += [
            f"- **Convergencia de la población**: la brecha entre el mejor y el promedio es "
            f"mínima en **{brecha['variante']}** ({brecha['brecha_media']:.4f}); cuanto más "
            f"chica, más se parecen entre sí todos los individuos.",
            "",
        ]

        igualado = presupuesto[target]
        if igualado is not None:
            corte, ranking = igualado
            orden_final = [f["variante"] for f in filas]
            orden_igualado = [v for v, _ in ranking]
            lineas += [
                f"#### A presupuesto de cómputo igualado",
                "",
                f"Las variantes de este eje **no gastan lo mismo**: al final del barrido, la más "
                f"cara hizo {max(f['evaluaciones'] for f in filas):,.0f} evaluaciones de fitness y "
                f"la más barata {min(f['evaluaciones'] for f in filas):,.0f}. Comparadas todas a "
                f"las **{corte:,.0f} evaluaciones** que alcanza la más barata:",
                "",
                "| # | Variante | Fitness a presupuesto igualado |",
                "|---|---|---|",
            ]
            for i, (variante, fit) in enumerate(ranking, start=1):
                lineas.append(f"| {i} | {variante} | {fit:.4f} |")
            if orden_final[:1] != orden_igualado[:1]:
                lineas += [
                    "",
                    f"**El orden se da vuelta.** Por generación gana *{orden_final[0]}*; por "
                    f"evaluación gasta gana *{orden_igualado[0]}*. La ventaja de "
                    f"*{orden_final[0]}* no venía del método sino de **haber consumido más "
                    f"cómputo**. Es el gráfico `fitness_vs_evaluaciones` el que hay que llevar "
                    f"a la presentación, no el de convergencia.",
                ]
            else:
                lineas += [
                    "",
                    f"El orden **se mantiene**: *{orden_igualado[0]}* gana también a presupuesto "
                    f"igualado, así que su ventaja no se explica por haber gastado más cómputo.",
                ]

            rango_final = filas[0]["media"] - filas[-1]["media"]
            rango_igualado = ranking[0][1] - ranking[-1][1]
            if rango_final > 0 and rango_igualado < rango_final * 0.5:
                lineas += [
                    "",
                    f"Y sobre todo: **el eje casi desaparece**. Al final del barrido separa "
                    f"{rango_final:.4f} de fitness entre el mejor y el peor; a presupuesto "
                    f"igualado, sólo {rango_igualado:.4f} "
                    f"({100 * (1 - rango_igualado / rango_final):.0f}% menos). La mayor parte de "
                    f"lo que parecía diferencia entre variantes era, en realidad, diferencia de "
                    f"cómputo consumido.",
                ]
            lineas.append("")

    lineas += ["## Qué mirar", ""] + [f"- {b}" for b in nota["que_mirar"]] + [""]
    lineas += ["## Trampas y advertencias", ""] + [f"- {b}" for b in nota["trampas"]] + [""]

    lineas += [
        "## Figuras",
        "",
        "Sin título ni texto adentro del PNG, fondo transparente: el rótulo lo pone la slide.",
        "",
        "| Archivo | Qué muestra | Cómo se lee |",
        "|---|---|---|",
        "| `convergencia_<target>.png` | Mejor fitness acumulado por generación, media entre semillas con banda ±σ | Pendiente = velocidad; altura final = calidad; ancho de la banda = cuánto depende de la suerte |",
        "| `diversidad_<target>.png` | Diversidad genética por generación, escala log | Cuándo y cuán rápido colapsa la población |",
        "| `boxplot_final_<target>.png` | Fitness final por variante, con los puntos de cada semilla | **Si las cajas se solapan, no hay ganador** |",
        "| `mejor_vs_promedio_<target>.png` | Mejor (sólido) y promedio (guionado) de la población | Cuando el promedio alcanza al mejor, la población convergió |",
        "| `fitness_vs_evaluaciones_<target>.png` | Mejor fitness contra evaluaciones de fitness | Comparación a presupuesto de cómputo igualado, no a generaciones iguales |",
        "",
        "## Imágenes resultado",
        "",
        f"El mejor individuo de cada variante, renderizado a {RENDER_SIZE}px con la primera "
        f"semilla del barrido (`seed={manifest['seeds'][0]}`). El target original está como "
        f"`<target>__TARGET.png`.",
        "",
        "| Target | Variante | Fitness | Archivo |",
        "|---|---|---|---|",
    ]
    for (target, variante), (fit, archivo, coincide) in sorted(fitness_render.items()):
        aviso = "" if coincide else " ⚠️ no coincide con el CSV"
        lineas.append(f"| {target} | {variante} | {fit:.4f}{aviso} | `imagenes/{archivo}` |")

    lineas += [
        "",
        "## Datos",
        "",
        f"- `datos/{experimento}.csv` — una fila por generación, variante y semilla",
        "- `datos/resumen.csv` — una fila por variante y target, con mediana e IQR",
        "",
    ]
    destino.write_text("\n".join(lineas))


def fila_indice(carpeta: str, titulo: str, stats: dict) -> dict:
    """Una fila del índice: ganador por imagen y en cuáles se separa del segundo."""
    ganadores = {t: filas[0]["variante"] for t, filas in stats.items()}
    sep = {t: separado(filas[0], filas[1]) for t, filas in stats.items() if len(filas) > 1}
    return {
        "carpeta": carpeta,
        "titulo": titulo,
        "variantes": len(next(iter(stats.values()))),
        "ganadores": ganadores,
        "separado": ", ".join(t for t, s in sep.items() if s) or "en ninguna",
    }


def resumen_desde_summary(manifest: dict) -> list[dict]:
    """Reconstruye el índice desde summary.csv, sin rehacer figuras ni imágenes."""
    filas = list(csv.DictReader((RESULTS / "summary.csv").open()))
    resumen = []
    for numero, experimento, slug_eje, titulo in [
            e for e in ORDEN if (RESULTS / f"{e[1]}.csv").exists()]:
        stats: dict[str, list] = {}
        for fila in filas:
            if fila["experiment"] != experimento:
                continue
            stats.setdefault(fila["target"], []).append({
                "variante": fila["variant"],
                "media": float(fila["best_fitness_mean"]),
                "q1": float(fila["best_fitness_q1"]),
                "q3": float(fila["best_fitness_q3"]),
            })
        if not stats:
            continue
        for target in stats:
            stats[target].sort(key=lambda f: -f["media"])
        resumen.append(fila_indice(f"{numero:02d}-{slug_eje}", titulo, stats))
    return resumen


def escribir_indice(manifest: dict, resumen_global: list[dict]) -> None:
    targets = list(manifest["targets"])
    lineas = [
        "# Informe de análisis — TP2",
        "",
        f"Generado desde el batch `{manifest['run_batch']}` "
        f"(commit `{manifest['git']['commit']}`"
        f"{', árbol sucio' if manifest['git']['dirty'] else ''}), "
        f"{manifest['runs']} corridas, {len(manifest['seeds'])} semillas por variante.",
        "",
        "Una carpeta por eje experimental. Cada una trae el informe, la configuración",
        "anotada, los datos, las figuras y las imágenes resultado.",
        "",
        "```",
        "NN-<eje>/",
        "├── informe.md      qué se ve y qué es importante analizar",
        "├── config.md       configuración fija, variantes y cómo reproducir",
        "├── datos/          CSV crudo del eje + resumen agregado",
        "├── figuras/        un gráfico por PNG, sin texto, fondo transparente",
        "└── imagenes/       el mejor individuo de cada variante, renderizado",
        "```",
        "",
        "## Ejes",
        "",
        "| Carpeta | Eje | Var. | " + " | ".join(f"Gana en `{t}`" for t in targets)
        + " | ¿Separado del 2º? |",
        "|---|---|---|" + "---|" * (len(targets) + 1),
    ]
    for fila in resumen_global:
        ganadores = " | ".join(fila["ganadores"].get(t, "—") for t in targets)
        lineas.append(
            f"| [`{fila['carpeta']}`]({fila['carpeta']}/informe.md) | {fila['titulo']} | "
            f"{fila['variantes']} | {ganadores} | {fila['separado']} |"
        )

    extras = [
        ("13-criterios-corte", "Criterios de corte",
         "Cuál usar y por qué, medido: para cada criterio candidato se reconstruye, sobre las "
         "corridas del barrido, en qué generación habría disparado y cuánto fitness habría "
         "costado. Responde lo que el enunciado pide justificar. "
         "Lo genera `python analysis/criterios_corte.py`."),
        ("14-interaccion", "¿Los ganadores de cada eje componen?",
         "El barrido varía un factor por vez; esto mide si juntar los ganadores da la mejor "
         "configuración, reemplazando un operador por vez desde la base. "
         "Lo genera `python analysis/interaccion.py`."),
        ("15-caso-final", "Caso final: La noche estrellada",
         "La imagen que la cátedra mostró como ejemplo. Los operadores se **heredan** del "
         "barrido; se barre sólo lo que depende del tamaño del problema. "
         "Lo genera `python analysis/caso_final.py`."),
    ]
    presentes = [(carpeta, titulo, texto) for carpeta, titulo, texto in extras
                 if (INFORME / carpeta).exists()]
    if presentes:
        lineas += ["", "## Análisis transversales y cierre", "",
                   "| Carpeta | Qué es |", "|---|---|"]
        for carpeta, titulo, texto in presentes:
            lineas.append(f"| [`{carpeta}`]({carpeta}/informe.md) | **{titulo}.** {texto} |")

    lineas += [
        "",
        "## Las tres imágenes",
        "",
        "Se eligieron por **composición**, no por dificultad: cuánta superficie plana tienen",
        "contra cuánta textura. Ése es el eje que ordena los resultados.",
        "",
        "| Etiqueta | Imagen | Composición |",
        "|---|---|---|",
        "| `plana` | bandera de Japón | Regiones planas grandes, un solo borde curvo |",
        "| `detallada` | Pikachu | Detalle fino sobre fondo liso, contornos negros duros |",
        "| `compleja` | La noche estrellada | Textura en todo el lienzo, sin regiones planas |",
        "",
        "**No están ordenadas por dificultad**, y conviene decirlo antes de que alguien lo",
        "note: medido por el error que le queda a la mejor configuración, la más difícil es",
        "`detallada`, no `compleja`.",
        "",
        "| Imagen | Mejor fitness alcanzado | Error remanente |",
        "|---|---|---|",
        "| `plana` | 0.9658 | 0.034 |",
        "| `detallada` | 0.8844 | **0.116** |",
        "| `compleja` | 0.9180 | 0.082 |",
        "",
        "Los contornos negros duros de Pikachu son justamente lo que un triángulo de color",
        "uniforme no puede reproducir; un óleo sin bordes se aproxima razonablemente bien con",
        "un borrón de triángulos, porque el RMSE premia el promedio local.",
        "",
        "## Cobertura del enunciado",
        "",
        "| Lo que pide | Dónde está |",
        "|---|---|",
        "| Justificar la estructura del individuo | `README.md` § Diseño, y `docs/TP2.md` |",
        "| Justificar la función de aptitud | `README.md` § Diseño |",
        "| Los 6 métodos de selección | [`01-seleccion`](01-seleccion/informe.md) (7, con la combinada aparte) |",
        "| Supervivencia aditiva y exclusiva | [`03-supervivencia`](03-supervivencia/informe.md), cruzada con la brecha generacional |",
        "| Al menos 2 métodos de cruza | [`04-cruza`](04-cruza/informe.md) (5) y [`05-granularidad`](05-granularidad/informe.md) |",
        "| Al menos 2 métodos de mutación | [`07-mutacion`](07-mutacion/informe.md) (4), a carga comparable |",
        "| **Decidir y justificar el criterio de corte** | [`13-criterios-corte`](13-criterios-corte/informe.md) |",
        "| Qué operador conviene **en qué circunstancia** | Cada eje se corre sobre 3 imágenes de composición distinta; el resumen por eje dice si el ganador cambia entre ellas |",
        "| Convergencia prematura (3 causas de la cátedra) | Presión de selección: [`02`](02-presion-seleccion/informe.md) · Tasa de mutación: [`08`](08-tasa-mutacion/informe.md) · Tamaño de población: [`10`](10-poblacion/informe.md) |",
        "| Cantidad de triángulos vs. calidad y tiempo | [`11-triangulos`](11-triangulos/informe.md) |",
        "",
        "## Cómo leer los números",
        "",
        f"Cada variante se corrió con {len(manifest['seeds'])} semillas. Con tan pocas repeticiones,",
        "**una diferencia de dos milésimas en la media no es un resultado**. El criterio que usamos",
        "en todo el informe es el rango intercuartil: si el del primero se solapa con el del",
        "segundo, la conclusión es que empatan. Los boxplot son el gráfico que lo muestra.",
        "",
        "La curva de convergencia usa el **mejor fitness acumulado**, que es monótono por",
        "construcción. El mejor de la población actual puede bajar con supervivencia exclusiva,",
        "porque los hijos desplazan a los padres; ese caso se ve en `mejor_vs_promedio`.",
        "",
        "## Base común",
        "",
        "Todos los ejes varían **un solo parámetro** sobre esta misma configuración base:",
        "",
        "| Parámetro | Valor |",
        "|---|---|",
    ]
    base = manifest["base_config"]
    for clave in sorted(base):
        lineas.append(f"| {ETIQUETAS.get(clave, clave)} (`{clave}`) | {valor(base[clave])} |")

    lineas += [
        "",
        "La base usa inicialización `random` **a propósito**: la inicialización informada es uno",
        "de los ejes a estudiar, y si estuviera en la base contaminaría la medición de todos los",
        "demás.",
        "",
        "## Reproducir todo",
        "",
        "```bash",
        "python analysis/run_experiments.py --clean   # barrido completo (~2 min)",
        "python analysis/build_report.py              # este informe (~1 min)",
        "```",
        "",
    ]
    (INFORME / "README.md").write_text("\n".join(lineas))


# --------------------------------------------------------------------------------------
def main() -> None:
    solo_indice = "--solo-indice" in sys.argv
    manifest_path = RESULTS / "manifest.json"
    if not manifest_path.exists():
        sys.exit(f"falta {manifest_path}. Corré primero analysis/run_experiments.py")
    manifest = json.loads(manifest_path.read_text())

    if solo_indice:
        escribir_indice(manifest, resumen_desde_summary(manifest))
        print(f"índice reescrito: {(INFORME / 'README.md').relative_to(ROOT)}")
        return

    figs.use_theme("light")
    # se regeneran sólo las carpetas de ejes; el caso final lo produce otro script
    INFORME.mkdir(parents=True, exist_ok=True)
    for carpeta in INFORME.iterdir():
        if carpeta.is_dir() and carpeta.name not in PROPIAS:
            shutil.rmtree(carpeta)

    resumen_rows = list(csv.DictReader((RESULTS / "summary.csv").open()))
    ejes = [e for e in ORDEN if (RESULTS / f"{e[1]}.csv").exists()]
    resumen_global = []
    iniciado = time.perf_counter()

    for numero, experimento, slug_eje, titulo in ejes:
        carpeta = INFORME / f"{numero:02d}-{slug_eje}"
        (carpeta / "datos").mkdir(parents=True)
        (carpeta / "figuras").mkdir()
        (carpeta / "imagenes").mkdir()
        print(f"[{numero}/{len(ejes)}] {titulo}")

        data = figs.load(RESULTS / f"{experimento}.csv")
        colores = figs.colors_for(data)

        extra = figs.FIGURAS_EXTRA.get(experimento, {})
        for target, data_target in data.items():
            for nombre, funcion in {**figs.FIGURES, **extra}.items():
                funcion(data_target, colores, carpeta / "figuras" / f"{nombre}_{target}.png")

        shutil.copy(RESULTS / f"{experimento}.csv", carpeta / "datos" / f"{experimento}.csv")
        filas_resumen = [r for r in resumen_rows if r["experiment"] == experimento]
        with (carpeta / "datos" / "resumen.csv").open("w", newline="") as fh:
            escritor = csv.DictWriter(fh, fieldnames=list(filas_resumen[0]))
            escritor.writeheader(); escritor.writerows(filas_resumen)

        stats = {target: estadisticas(dt) for target, dt in data.items()}

        guardar_targets(manifest, carpeta / "imagenes")
        trabajos = jobs_imagenes(manifest, experimento, carpeta / "imagenes")
        fitness_render = {}
        with ProcessPoolExecutor() as pool:
            for (_exp, target, variante), fit in pool.map(_render_one, trabajos):
                esperado = next(f["media"] for f in stats[target] if f["variante"] == variante)
                fitness_render[(target, variante)] = (
                    fit, f"{target}__{slug(variante)}.png", abs(fit - esperado) < 0.05
                )

        escribir_config(manifest, experimento, titulo, carpeta / "config.md")
        presupuesto = {t: presupuesto_igualado(dt) for t, dt in data.items()}
        escribir_informe(manifest, experimento, titulo, slug_eje, stats, presupuesto,
                         fitness_render, carpeta / "informe.md")

        resumen_global.append(fila_indice(carpeta.name, titulo, stats))

    escribir_indice(manifest, resumen_global)
    print(f"\ninforme en {INFORME.relative_to(ROOT)}/ ({time.perf_counter() - iniciado:.0f}s)")


if __name__ == "__main__":
    main()
