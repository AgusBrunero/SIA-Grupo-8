# TP5 · Deep Learning: Autoencoder, Denoising Autoencoder y VAE

Grupo 8 · Sistemas de Inteligencia Artificial · ITBA, 2C 2026.

Todo está implementado a mano en numpy: el perceptrón multicapa, el backpropagation, los optimizadores, el truco de reparametrización y la divergencia KL. No se usa ningún framework de deep learning. De scipy se usa únicamente `scipy.optimize.minimize` con el método de Powell, que es el que nombra la cátedra.

## Requisitos e instalación

Python 3.14. Desde la carpeta `TP5/`:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Cómo correr

Todos los comandos se corren desde `TP5/`.

**Tests** (incluyen el gradient check por diferencias finitas de cada activación, cada pérdida, la KL y el VAE completo):

```bash
.venv/bin/python -m unittest discover -s tests -t . -v
```

**Experimentos.** Cada uno lee su configuración de un archivo de `config/` (semillas, arquitectura, optimizador, épocas) y escribe en `results/<experimento>/` sus CSV, sus figuras y una copia del config usado. El archivo `config.json` de la raíz es el índice de todos.

| Consigna | Comando | Salida |
|---|---|---|
| 1a.1 y 1a.2 · arquitecturas y optimización | `.venv/bin/python -m experiments.e1a2_escalera config/e1a2.json` y después `.venv/bin/python -m experiments.e1a2_analisis` | `results/e1a2/` |
| 1a.3 y 1a.4 · latente y letra nueva | `.venv/bin/python -m experiments.e1a3_e1a4_latente config/e1a3_e1a4.json` | `results/e1a3_e1a4/` |
| 1b · Denoising Autoencoder | `.venv/bin/python -m experiments.e1b_dae config/e1b.json` | `results/e1b/` |
| 2a y 2b · dataset y VAE | `.venv/bin/python -m experiments.e2b_vae config/e2b.json` | `results/e2b/` |
| 2b y 2c · AE contra VAE, λ y generación | `.venv/bin/python -m experiments.e2c_latente_generacion config/e2c.json` | `results/e2c/` |
| 1a.4 y 2c · muestras elegidas | `.venv/bin/python -m experiments.elegidas config/elegidas.json` | `results/elegidas/` |

`experiments.elegidas` lee los CSV de `e1a3_e1a4` y de `e2c`, así que va después de esos dos.

Los resultados son reproducibles: con el mismo config y la misma semilla, los CSV salen idénticos.

## Presentación

`docs/presentacion/presentacion.html`: un solo archivo, se abre con doble clic en cualquier navegador y funciona sin internet. Teclas: `→` o espacio avanza, `←` retrocede, `F` pantalla completa, `N` muestra las notas del orador. Hay un respaldo en PowerPoint, `docs/presentacion/TP5_Grupo8.pptx`.

Para regenerarla, desde `TP5/`:

```bash
.venv/bin/python -m experiments.presentacion      # números, figuras de las láminas y el HTML
node docs/presentacion/build.js                   # respaldo en PowerPoint (antes: npm install en esa carpeta)
```

- Los textos de las láminas están en `docs/presentacion/laminas.json` y el estilo en `plantilla.html`.
- Cada número en pantalla sale de `docs/presentacion/manifest.json`, que registra el CSV de origen.
- La generación falla si una lámina tiene más de 35 palabras, más de 3 números o un número escrito a mano.
- `docs/presentacion/guion_borrador.md` trae las notas de cada lámina para que quien presenta las reescriba.

## Estructura

```
TP5/
  config.json        índice de las configuraciones
  config/            un .json por experimento
  src/
    data/            font.py (lee font.h), ruido.py (salt-and-pepper), emojis.py (dataset de caritas)
    nn/              activaciones, pérdidas, MLP con backpropagation, optimizadores, adaptador de Powell
    modelos/         autoencoder, DAE, VAE, PCA con SVD, generación y métricas
  experiments/       un script por ítem de la consigna
  results/           CSV, figuras y copia del config de cada experimento
  tests/
  docs/
```

## Datos

- **Ejercicio 1:** los 32 caracteres de 5 × 7 de `docs/tp5/font.h`, leídos desde el archivo.
- **Ejercicio 2:** un dataset propio de caritas de 16 × 16 en grises, construido en código con semilla (`src/data/emojis.py`). No se descarga nada.

## Documentación

- `docs/RESULTADOS.md`: las observaciones con números. Lo generan los scripts desde los CSV; no se edita a mano.
- `docs/CONCLUSIONES.md`: la conclusión de cada ítem de la consigna, con su evidencia y la parte de la materia que la explica.
- `docs/DECISIONES.md`: cada decisión de implementación con la clase y la diapositiva que la sostiene.
- `docs/DESVIOS.md`: en qué nos apartamos de las diapositivas o de lo planeado, y por qué.

## Protocolo de los experimentos

- Cada experimento se repite con al menos 5 semillas, leídas del config; se informa media ± desvío entre semillas y cuántas semillas logran el objetivo.
- Los archivos `config/e1a2_piloto*.json` y `config/e1a2_diagnostico.json` son corridas exploratorias con 2 semillas que sirvieron para elegir hiperparámetros; no son resultados.
