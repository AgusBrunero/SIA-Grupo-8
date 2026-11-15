# TP5 · Deep Learning (AE, DAE, VAE) — Reglas para Claude Code

Grupo 8 · SIA ITBA 2C 2026. Este archivo manda sobre tus costumbres. Si un pedido de una sesión choca con algo de acá, **frená y preguntá**.

## 0. Fuente de verdad (en este orden)

1. `docs/tp5/enunciado_y_catedra_tp5.md`: el enunciado textual y el contenido de la cátedra, con diapositivas.
2. Los PDFs originales en `docs/tp5/clases/`, para verificar fórmulas. Si el `.md` y el PDF no coinciden, gana el PDF y **lo reportás**.
3. Nada más:
   - ni tu memoria de "cómo se hace normalmente un VAE";
   - ni tutoriales;
   - ni repos públicos de TP5 de otros grupos (no copiar código ajeno).

## 1. Regla de oro: solo lo que enseña la cátedra

- **Toda técnica, fórmula, métrica o arquitectura lleva su referencia** (archivo y diapositiva) en un comentario del código y en `docs/DECISIONES.md`.
- **Lo que no está desarrollado en la cátedra no se implementa.** Ver §6 del doc de cátedra.
  - Si creés que hace falta, **proponelo antes**, etiquetado **[EXTRA]**, con tres cosas: qué problema resuelve, la evidencia de que ese problema aparece en *nuestros* resultados y qué diapositiva es lo más cercano.
  - Lo decide el grupo.
- **Notación de la cátedra:**
  - **MLP:** x (entrada), ζ (salida esperada), O (salida), h (excitación), V (salida de capa oculta), θ (activación), η, δ, μ (un dato), p (cantidad de datos), m (capa), M (cantidad de capas).
  - **AE:** X, Z, X′.
  - **VAE:** q_ϕ(z|x), p_θ(x|z), μ(x), s(x) = log σ², ϵ, J = L + λR.
- **Inconsistencias de las diapositivas** (§7 del doc de cátedra): usás la versión indicada ahí. **No corregís nada en silencio.**

## 2. Librerías

- **Permitido:**
  - numpy, pandas, matplotlib, plotly;
  - `scipy.optimize.minimize` (la cátedra nombra Powell, BFGS y Newton, y dice que usa Powell para el TP5; L-BFGS-B no está en la cátedra y es [EXTRA]);
  - PIL, solo para leer o escribir imágenes del dataset.
- **Prohibido** ("NO UTILIZAR DEEP LEARNING FRAMEWORKS ni ninguna otra librería relacionada con los algoritmos que vemos en la materia"):
  - torch, tensorflow, keras, jax, autograd;
  - sklearn completo (si hace falta PCA, se hace con la SVD de numpy).
- **Se escriben a mano en numpy:** el backprop, los optimizadores, la reparametrización y la KL. Todo en float64.

## 3. Forma de trabajo

1. **Plan antes de código.** Cada sesión arranca en plan mode con:
   - qué vas a hacer;
   - qué archivos tocás;
   - qué test lo prueba;
   - qué diapositivas lo sostienen.

   Esperás un OK.
2. **Cambios chicos.**
   - Una sesión, un objetivo.
   - Si un cambio toca más de 3 archivos o más de 150 líneas, lo partís y avisás.
   - Nada de refactors no pedidos.
3. **Explicar cada decisión:** qué hiciste, por qué, y qué parte de la teoría lo justifica. Si no hay diapositiva que lo sostenga, es [EXTRA].
4. **Desvíos declarados, nunca racionalizados.** Si no hiciste exactamente lo pedido (otro parámetro, otra definición, otro criterio, algo que no anduvo):
   - lo escribís **primero** en tu respuesta, bajo el título `Desvíos`;
   - lo agregás a `docs/DESVIOS.md`.

   Nunca lo presentes después como si fuera lo pedido.
5. **No inventes resultados.**
   - Todo número que escribas (chat, README, gráfico, lámina) sale de un CSV de `results/` generado por un script del repo.
   - Si algo no se corrió, decís "no corrido".
6. **Nada en background:** ni `nohup`, ni `&`, ni daemons, ni servidores vivos. Los experimentos corren en primer plano, con log a archivo. Si algo es largo, partilo por semillas o por configuraciones.
7. **Los tests prueban lo real.**
   - Importan los mismos módulos que usan los experimentos, no copias.
   - Después de mover o renombrar algo: correr **todos** los tests y hacer grep de las rutas viejas.
8. **Las conclusiones son del grupo.**
   - Vos generás datos, figuras y **observaciones factuales** (qué se ve, con números).
   - No escribís conclusiones, títulos-afirmación ni narrativa de presentación, salvo en la sesión de presentación, partiendo de `docs/CONCLUSIONES.md`, que escribe el grupo.

## 4. Verificación obligatoria antes de decir "listo"

- **Gradient check** por diferencias finitas centradas (paso de 1e-5, error relativo < 1e-6 en float64). Cubre:
  - cada activación;
  - MSE y BCE;
  - la KL;
  - el VAE completo con ϵ fijo.

  **Sin gradient check en verde no se entrena nada nuevo.**
- **MLP base:** XOR con [2,2,1] y [2,3,2,1] aprende (ejercicio de validación del TP3).
- **Dataset:** `font.h` se parsea desde el archivo, sin copiar números a mano. Un test compara 'a', 'l' y '|' con arte ASCII escrito en el test. Orden de bits: 0x10 es la columna izquierda.
- **Semillas:**
  - Todo experimento lee la semilla desde `config/*.json`.
  - Se repite con N semillas: **mínimo 5**, y 10 si la comparación es ajustada.
  - Se reporta media ± desvío y **la fracción de semillas que logra el objetivo**.
  - Nunca "el mejor run".
- **"A es mejor que B"** solo si la diferencia supera el desvío entre semillas. Si no, se escribe "empatan".
- **Conteos:** toda tabla agregada tiene un `assert` sobre cuántas corridas entran por celda (evita el doble conteo).
- **Texto contra datos:** todo epígrafe o frase con un número se genera desde el CSV o tiene un `assert` que lo verifica.

## 5. Estructura del repo y entregable

La cátedra pide: repo con la ppt, `README.md` y `config.json`; presentación de 25 minutos, orientada a resultados, con promedio de ejecuciones y seed fija.

```
TP5/
  CLAUDE.md
  config/            un .json por experimento; nada hardcodeado
  src/
    data/            font.py, ruido.py, relojes.py
    nn/              activaciones.py, perdidas.py, mlp.py, optimizadores.py, scipy_adapter.py
    modelos/         autoencoder.py, dae.py, vae.py
  experiments/       un script por ítem de la consigna (e1a2_*.py, e1b2_*.py, e2b_*.py…)
  results/<exp>/     CSV + figuras + copia del config usado
  tests/
  docs/              DECISIONES.md, DESVIOS.md, RESULTADOS.md, CONCLUSIONES.md (este lo escribe el grupo)
  README.md
```

**Docstring de cada experimento:** arriba de todo, qué ítem de la consigna responde, qué pregunta concreta y qué diapositivas lo sostienen.

**Figuras:**
- sin título interno;
- ejes en español y con unidades;
- leyenda legible;
- mismos colores para el mismo modelo en todo el TP.

## 6. Errores que ya cometimos (no repetir)

| TP | Error | Regla que lo evita |
|---|---|---|
| TP4 | Se mostraron métodos que no están en la cátedra (Jolliffe B2/B4) y nadie supo definirlos | §1 |
| TP4 | Notación ajena: 0,138·N en vez de 0,15·N, E en vez de H | §1, notación |
| TP4 | Muchos experimentos y ninguna conclusión con dueño | §3.8 y docstrings de §5 |
| TP4 | La IA decidió qué era importante | §3.8 |
| TP3 | Un epígrafe decía lo contrario que los datos | §4, texto contra datos |
| TP2 | Doble conteo en una figura | §4, conteos |
| Sesiones previas | Un proceso en background sobrevivió a la sesión | §3.6 |
| Sesiones previas | Tests que verificaban la carpeta equivocada después de renumerar | §3.7 |
| Sesiones previas | Desvíos justificados a posteriori | §3.4 |