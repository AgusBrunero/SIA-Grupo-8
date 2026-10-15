# Ejercicio 2 · Memoria asociativa con Hopfield

Se completaron el caso sencillo de letras 5×5 (2.2) y la recuperación de los cinco QR de
Trattoria Aurora (2.1). Los experimentos usan el mismo modelo implementado con NumPy.

## Qué hace la red

Cada patrón se representa con valores −1 y +1. El aprendizaje de Hebb construye una
matriz de conexiones que combina los patrones almacenados, sin autoconexiones. La red
recibe una versión alterada y actualiza una neurona por vez según las demás, hasta que
un barrido completo no cambia nada. Los empates conservan el estado anterior.

La energía permite comprobar la dinámica: con los pesos simétricos y esta actualización
asincrónica no aumenta. Pero un punto fijo puede ser una memoria distinta o un estado
espurio. Recuperar correctamente exige comparar la salida con el patrón consultado.

## Letras: aprender a evaluar una memoria asociativa

Se definieron 16 letras de 25 bits y se enumeraron sus 1.820 cuartetos. Se eligieron
por productos internos absolutos los extremos FILX (media normalizada 0,0933) y BEFP
(0,6533). Se observó que la similitud entre memorias importa tanto como su cantidad.

FILX conserva sus cuatro letras sin ruido; BEFP conserva solo dos. Con 20 % de bits
invertidos, la recuperación exacta fue 88,5 % frente a 39,75 %. Se hicieron 6.400 consultas
con diferentes niveles de ruido y se evaluó además estabilidad con entre una y seis letras.
La referencia `0,138N` sirve como orientación para otro régimen de patrones, no como
garantía sobre estas letras pequeñas y correlacionadas.

Con F y 40 % de ruido se obtuvo un espurio reproducible: terminó a 6, 5, 5 y 13 bits
de F, I, L y X. La energía bajó de 0,32 a −10,40, aunque no recuperó ninguna memoria.
El desarrollo, las tablas y los gráficos están en
[conclusiones de letras](hopfield/letras/conclusiones.md).

## QR: el límite de Hebb con patrones muy correlacionados

Cada QR tiene 37×37 módulos: 1.369 neuronas. Se almacenaron cinco simultáneamente.
Comparten el 69,76 % de los píxeles y sus solapamientos entre pares llegan a 0,86.
**Ninguno de los cinco es estable con Hebb**, aun sin ruido. Las consultas originales
terminan en el mismo estado espurio, modificando entre 71 y 225 bits.

Se hicieron 3.000 recuperaciones entre dos reglas, con ruido aleatorio de 0–50 % y daños
localizados: esquinas borradas, manchas negras y rayas blancas. Hebb obtuvo 0 % de
recuperación exacta en todas las condiciones probadas. Al almacenar cada QR por separado,
sí resulta estable: el problema principal es la interferencia entre memorias.

Como mejora opcional se implementó la regla de proyección por pseudoinversa, manteniendo
la misma dinámica. Logró 100 % de recuperación exacta en los niveles probados hasta 30 %
de ruido, 149/150 consultas al 40 % y 0/150 cerca del 50 %. También recuperó todos los daños
localizados ensayados. Estos porcentajes describen las muestras realizadas, no cualquier
deterioro posible. Hebb sigue siendo el caso clásico analizado, no se reemplazó su resultado.

Se agregó un lector real, ZXing-C++, para contrastar el texto recuperado con la URL
original. Se preparan las imágenes con borde blanco y escala entera; eso ocurre fuera
de la red. Los cinco originales se leen correctamente. La lectura de las salidas en
estos ensayos coincide con los aciertos exactos, pero ambos criterios son distintos. En el fallo al 40 %, una consulta de food terminó en
wine: el lector leyó un QR válido de otro menú. Detectar un QR no basta para declarar éxito.

Ver [conclusiones de QR](hopfield/qr/conclusiones.md),
[ejemplos visuales](hopfield/qr/ejemplos.png) y [curvas de recuperación](hopfield/qr/recuperacion.png).

## Conclusión del ejercicio

Hopfield permite recuperar patrones conocidos a partir de consultas deterioradas cuando
las memorias y sus cuencas de atracción lo permiten. Almacenar pocos patrones respecto de N
no basta: los productos internos, la regla de aprendizaje y el deterioro cambian el resultado.
La caída de energía certifica una propiedad de la dinámica, no el éxito de la consulta.

En el restaurante, Hebb no es adecuado para estos cinco QR simultáneos. La pseudoinversa
mejora mucho el caso estudiado, pero el sistema sigue limitado a las cartas almacenadas y
a consultas ya alineadas y binarizadas. No es un lector universal de QR nuevos o fotografías.

## Cómo reproducirlo

Desde `TP4/`, con un entorno Python activado:

```bash
python -m pip install -r requirements-qr.txt
python letras.py
OPENBLAS_NUM_THREADS=1 python qr.py
OPENBLAS_NUM_THREADS=1 python -m unittest discover -s tests -v
```

Los scripts generan CSV, modelos y PNG en `resultados/hopfield/letras/` y
`resultados/hopfield/qr/`. Pasaron doce pruebas del modelo y de la integración de QR.
Se inspeccionaron los gráficos y se verificó la energía durante las corridas.
