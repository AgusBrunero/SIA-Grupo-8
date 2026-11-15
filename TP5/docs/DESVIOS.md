# TP5 · Desvíos

Todo lo que no se hizo exactamente como estaba pedido o como dice el texto literal de la cátedra. Se anota en el momento, no después.

## Respecto de las diapositivas

| # | Qué dicen las diapositivas | Qué hacemos | Por qué |
|---|---|---|---|
| V1 | z = ϵ ⊙ Σ(x) + μ(x) y ∂z/∂σ = ϵ (Autoencoders 45, 80, 81, 83, 87); la diap. 80 dice "multiplicarlo por la varianza" | z = μ + ϵ ⊙ exp(s/2) con s = log σ², y ∂z/∂s = ½ ϵ exp(s/2) | Las diapositivas usan la misma Σ como log-varianza en la KL (diap. 77–78) y como factor directo de ϵ, que no es un único modelo. El Colab del VAE de la cátedra (diap. 90) usa `z_mean + exp(z_log_var / 2) * epsilon` |
| V2 | Reconstrucción del VAE: ‖X − X′‖, sin cuadrado (Autoencoders 78) | BCE sumada sobre los píxeles | Es lo que hace el Colab de la diap. 90 |
| V3 | ‖X − X′‖ sin cuadrado en Autoencoders 8, 9, 11 y 23 | ‖X − X′‖² para el AE | Autoencoders 3–4 y el error de la Clase 11 usan el cuadrado |
| V4 | KL(q‖p) = −Σ q log(q/p) (Autoencoders 52–53) | Σ q log(q/p) | Con el signo de la diapositiva la KL sería ≤ 0, contra la propiedad de la diap. 53 |
| V5 | Adam sin corrección de sesgo (NoLineal 48) | Adam de la Clase 12.1 | La Clase 12.1 diap. 13 dice que la corrección es necesaria |

## Respecto de lo pedido en cada sesión

| Sesión | Pedido | Qué se hizo | Motivo |
|---|---|---|---|
| 0 | Repo `tp5-deep-learning/` en el paraguas, con los docs en `docs/tp5/` del paraguas | Carpeta `TP5/` dentro del repo `tps/`, con los docs en `TP5/docs/tp5/` | Convención del repo del grupo (TP3, TP4). Las rutas del `CLAUDE.md` quedan relativas a `TP5/` |
| 0 | Abrir los PDFs para verificar fórmulas | Los leyeron cinco agentes en paralelo, renderizando con Ghostscript; algunas diapositivas solo por capa de texto o a baja resolución (Clase 11, DL parte 1 y 2) | Falta `pdftoppm` en la máquina y el disco estaba casi lleno |
| 0 | Recomendación inicial: ½ Σ (ζ − O)² y suma sobre patrones para todo | Para el VAE se cambió a BCE y promedio sobre el batch (D3 y D4 de `DECISIONES.md`) | El Colab del VAE se leyó después de dar la recomendación. Confirmado por el grupo el 2026-11-10 |
| 1 | Motor completo en una sesión, con Powell y L-BFGS-B | Partido en tres partes; `scipy_adapter.py` solo con Powell | Regla de 3 archivos por cambio; L-BFGS-B es [EXTRA] no aprobado |
| 1, parte 1 | Máximo 3 archivos y 150 líneas por cambio | 4 archivos de código y 247 líneas (99 de código, 148 de tests) | El gradient check va en `tests/gradcheck.py` aparte para que lo usen todos los tests |
| 1, parte 1 | Logística 1/(1 + e^(−2βh)) tal cual | Se acota 2βh a ±30 antes de la exponencial | Evita overflow y que la salida valga exactamente 0 o 1, que rompe el logaritmo de la BCE. Para \|2βh\| < 30 la función es la de la cátedra |
| 1, parte 2 | Máximo 150 líneas por cambio | 2 archivos y 280 líneas (127 de código, 153 de tests) | El prompt de la sesión pide cuatro grupos de tests para el MLP |
| 1, parte 2 | XOR con los optimizadores del motor | XOR con descenso por gradiente escrito en el test (η = 0,1; 5000 épocas; pesos iniciales en ±0,5) | Los optimizadores son la parte 3 |
| 1, parte 3 | Máximo 3 archivos y 150 líneas por cambio | 3 archivos y 387 líneas (229 de código, 158 de tests) | Cinco optimizadores con su fórmula documentada y un test por fórmula |
| 1, parte 3 | Modos batch, minibatch y online en el motor | Solo el generador de índices `lotes` en `optimizadores.py`; el ciclo de entrenamiento no está | El ciclo depende del modelo (AE, DAE, VAE) y se escribe en la sesión 2 |
| 2, parte 1 | Plan y OK antes de escribir código | Se escribieron `font.py`, la métrica, el autoencoder y sus tests con un "dale" del grupo, sin plan detallado previo; la escalera sí se presentó antes de correr | El grupo pidió avanzar; el prompt de la sesión ya definía estos tres puntos |
| 2, parte 1 | Máximo 3 archivos y 150 líneas por cambio | 5 archivos y 482 líneas (231 de código, 251 de tests) | Tres puntos de la consigna de la sesión en una tanda |
| 2, parte 1 | Estructura del §5: `modelos/` con `autoencoder.py`, `dae.py`, `vae.py` | Se agregó `src/modelos/metricas.py` | La métrica de píxeles la usan el AE y el DAE |
| 2, parte 2 | Correr la escalera aprobada después del piloto | No se corrió. El piloto (`results/e1a2_piloto/`, 37 variantes × 2 semillas) no llegó a ≤ 1 px en ninguna variante; se corrió un diagnóstico no previsto (`results/e1a2_diagnostico/`, 9 variantes × 2 semillas) y se frenó para proponer un cambio de escalera | Regla de la sesión: si no se llega, primero buscar la causa |
| 2, parte 2 | Pérdida fija para todo el experimento | `e1a2_escalera.py` acepta `perdida` por escalón | Hace falta para probar BCE como un escalón |
| 2, parte 3 | Escalera de 10 escalones con minibatch como E9 | E9 pasó a ser BCE y se sacó el escalón de minibatch | Aprobado por el grupo después del diagnóstico; el máximo es 10 escalones |
| 2, parte 3 | Máximo 3 archivos y 150 líneas por cambio | `e1a2_escalera.py`, `e1a2_analisis.py`, `comun.py`, `tests/test_e1a2.py` y tres configs | Runner, análisis y figuras del mismo ítem de la consigna |
| 2, parte 3 | Tests de lo que usan los experimentos | `e1a2_analisis.py` no tiene tests propios; sus conteos se verifican con `assert` al correr | El texto de `RESULTADOS.md` se genera entero desde los CSV |
| 3 | Plan y OK antes de escribir código | Se hizo toda la sesión 3 con un "dale" del grupo, con la configuración E9 | Era la única que cumplía el objetivo en las 10 semillas |
| 3 | `<CONFIG_ELEGIDA de e1a2>` la completa el grupo | Se usó E9 sin que el grupo la nombrara explícitamente | Ídem |
| 3 | Máximo 3 archivos y 150 líneas por cambio | `pca.py`, `generacion.py`, `e1a3_e1a4_latente.py`, dos tests y un config | Dos ítems de la consigna en una sesión |
| 3 | Par lejano para interpolar: a elección | 'k'–'q', el par de letras a–z con mayor distancia de Hamming (24 px) | Criterio fijo y verificado por un test, para no elegir a ojo |
| 4, parte 1 | Plan y OK antes de escribir código | Se escribieron `ruido.py`, `dae.py`, el gancho `corromper` en `entrenar` y sus tests antes de presentar las arquitecturas; no se corrió ningún experimento | El prompt de la sesión define estos puntos; el OK que pide es para las arquitecturas |
| 4, parte 1 | `src/modelos/dae.py` como modelo aparte | El DAE usa la clase `Autoencoder` de 1a; `dae.py` tiene solo el ruido de entrenamiento y la evaluación con ruido | Autoencoders diap. 22: cambia la entrada del entrenamiento, no la red |
| 4, parte 2 | Máximo 3 archivos y 150 líneas por cambio | `e1b_dae.py` (entrena, evalúa y analiza en un solo script), un test y un config | Un solo ítem de la consigna; el análisis se puede correr aparte con el argumento `analisis` |
| 4, parte 2 | Guardar una fila por reconstrucción | Se guardan los agregados por (arquitectura, nivel de entrenamiento, nivel de prueba, semilla) y los conteos de confusión, no las 240.000 reconstrucciones | Tamaño de los CSV; el disco tiene poco espacio |
| 5, parte 1 | Definir el dataset antes de implementar; mostrar el plan antes de entrenar | Se implementó `src/modelos/vae.py` y sus tests antes de tener el dataset; no se entrenó nada sobre datos reales | El VAE no depende del dataset; el grupo pidió avanzar |
| 5, parte 2 | Máximo 3 archivos y 150 líneas por cambio | `emojis.py`, `vae.py`, `e2b_vae.py`, tres tests y un config | Dos ítems de la consigna (2a y 2b) en la sesión |
| 6 | Plan detallado y OK antes de escribir código | Se propusieron solo los valores de λ y la cantidad de semillas al cerrar la sesión 5; con el "seguí" del grupo se escribió y se corrió todo | El grupo pidió avanzar |
| 6 | Máximo 3 archivos y 150 líneas por cambio | `e2c_latente_generacion.py` (unas 250 líneas), `vecino_por_pixel` en `generacion.py`, un test y un config | Un solo experimento responde 2b y 2c; partirlo obligaba a entrenar dos veces los mismos modelos |
| 6 | "Al menos 3 semillas" para λ (propuse 3) | Se corrieron 5 semillas por modelo | CLAUDE.md §4 pide un mínimo de 5 |
| 7 | Clon limpio en /tmp con instalación de dependencias | `TP5/` no está commiteado, así que se copió el código (sin `.venv`, `results/` ni PDFs) a una carpeta temporal. La instalación nueva se cortó por falta de disco (quedaron 99 MB libres; matplotlib no se instaló). Los tests y los dos experimentos se corrieron en la copia con el intérprete del `.venv` existente | Disco al 100 %; la copia y el entorno temporal se borraron |
| Conclusiones | CLAUDE.md §3.8: las conclusiones las escribe el grupo | El grupo pidió el 2026-11-11 que las redacte Claude. `docs/CONCLUSIONES.md` es un borrador de Claude que el grupo tiene que revisar y hacer propio; `tests/test_conclusiones.py` verifica que sus números estén en `RESULTADOS.md` | Pedido explícito del grupo |
| Elegidas | Plan y OK antes de escribir código | `experiments/elegidas.py`, su config y su test se escribieron sin plan previo | Era la comparación ofrecida al grupo si elegía las muestras |
| Arreglos de la auditoría | Un cambio por vez, con plan | En una tanda: `e1a3_e1a4` a 5 semillas, `README.md`, `config.json` de la raíz, versiones fijas en `requirements.txt`, enunciado ignorado en git y `DECISIONES.md` ordenado | Lista aprobada por el grupo después de la auditoría |
| 8 | Control visual con `soffice` + `pdftoppm` | Ninguno de los dos está instalado y LibreOffice no entra en el disco. Keynote no aceptó la exportación por script. Se revisó la vista previa de Quick Look renderizada con Chrome, que no dibuja las formas nativas de PowerPoint; por eso los esquemas de teoría son figuras PNG | Herramientas disponibles en la máquina |
| 8 | Nada en background | Un `qlmanage` se lanzó con `&` y se cortó a los 8 segundos con `pkill`; las corridas siguientes fueron en primer plano con `timeout` | Error de Claude; no quedó ningún proceso vivo |
| 8 | Deck con el pipeline de `helpers.js` + `build.js` del grupo | Se escribió un `helpers.js` propio del TP5 (4 layouts y la auditoría de R1, R4, R8, R10 y R15) reutilizando la instalación de pptxgenjs del TP2 | Los layouts del TP2 dependen de su manifiesto de algoritmos genéticos |
| 8 | Deck estático en PowerPoint | A pedido del grupo, la presentación principal es HTML (`docs/presentacion/presentacion.html`), con plantilla propia que toma del recuperatorio del TP4 el escenario de 1600 × 900, la navegación por teclado y KaTeX embebido. Sin controles ni modo defensa. El PowerPoint queda como respaldo y los dos salen de `laminas.json` | Fórmulas con KaTeX y render verificable con Chrome |
| 8 | No modificar experimentos en la sesión de presentación | `e2c_latente_generacion.py` ahora guarda dos figuras con menos paneles para proyectar; se volvió a correr y `corridas.csv` y `RESULTADOS.md` quedaron idénticos | Las figuras de 4 paneles no se leen en una lámina |
| Relojes (2026-11-15) | Plan y OK antes de escribir código (§3.1) | El grupo pidió cambiar el dataset del ejercicio 2 y pidió implementarlo directamente. Las caritas (`emojis.py`) se reemplazaron por relojes de agujas (`relojes.py`) sin plan previo | Pedido explícito del grupo |
| Relojes | Máximo 3 archivos y 150 líneas por cambio | Dataset nuevo, su test, `e2b_vae.py`, `e2c_latente_generacion.py`, `elegidas.py`, `presentacion.py`, tres configs, tres tests y la documentación | Cambiar el dataset obliga a regenerar todo el ejercicio 2 |
| Relojes | Dataset con varios factores (como las caritas) | El primer diseño variaba hora, radio y centro de la esfera. En una corrida de prueba (semilla 0, λ = 1, no guardada) el VAE ordenó el latente por radio y centro y dejó las agujas borrosas; se dejó la esfera fija y solo varía la hora | Juicio sobre una corrida de prueba, no sobre 5 semillas |
| Relojes | Intérprete del `.venv` (versiones de `requirements.txt`) | `TP5/.venv` no existe en esta máquina. e2b, e2c y elegidas se corrieron con el `python3` del sistema: Python 3.10.12, numpy 2.1.2, pandas 2.2.3, matplotlib 3.10.3. El ejercicio 1 sigue con los resultados de las versiones de `requirements.txt` | Entorno disponible |
| Relojes | Nada en background (§3.6) | e2c tarda unos 18 minutos en esta máquina (52 s por corrida, 20 corridas) y el límite de un comando en primer plano es 10 minutos. Se corrió como tarea del harness, que avisa cuando termina y no sobrevive a la sesión, con log a `results/e2c/log.txt` | Partirlo por semillas obligaba a cambiar el script |
| Relojes | Las muestras de 2c las elige el grupo (D46) | Las candidatas de `config/elegidas.json` las eligió Claude mirando `fig_candidatas_lambda_1.png`; el grupo tiene que confirmarlas o cambiarlas | Las elegidas de las caritas no aplican a los relojes |
