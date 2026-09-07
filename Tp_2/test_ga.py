"""Sanity checks del motor (los 'Definition of Done' de los steps 1-4)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from ga import artifact
from ga import crossover as crossover_mod
from ga import engine
from ga import mutation as mutation_mod
from ga import selection as selection_mod
from ga import stopping
from ga.context import Context
from ga.crossover import one_point
from ga.fitness import FitnessEvaluator
from ga.individual import GENES_PER_TRIANGLE, Individual, grid_individual, random_individual
from ga.mutation import gene
from ga.render import canvas_size, load_target, render, render_array
from ga.replacement import additive, exclusive
from ga.selection import elite

CANVAS = 32


def make_ctx(**params) -> Context:
    return Context(rng=np.random.default_rng(0), params=params)


def solid_target(color) -> np.ndarray:
    return np.full((CANVAS, CANVAS, 3), color, dtype=np.float64)


class TestRender(unittest.TestCase):
    def test_canvas_vacio_es_el_fondo(self):
        empty = Individual(np.zeros(0))
        rendered = render_array(empty, CANVAS)
        self.assertEqual(rendered.shape, (CANVAS, CANVAS, 3))
        np.testing.assert_array_equal(rendered, 255.0)

    def test_triangulo_opaco_pinta(self):
        genes = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])
        rendered = render_array(Individual(genes), CANVAS)
        self.assertLess(rendered.mean(), 255.0)


class TestCanvasRectangular(unittest.TestCase):
    """`preserve_aspect`: una imagen apaisada no se puede comparar contra un target
    aplastado a cuadrado. Con la opción activa, `canvas_size` es el lado LARGO."""

    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.path = str(Path(self.dir.name) / "apaisada.png")
        Image.new("RGB", (200, 100), (10, 20, 30)).save(self.path)

    def test_por_defecto_fuerza_el_cuadrado(self):
        self.assertEqual(canvas_size(self.path, 64, False), (64, 64))
        self.assertEqual(load_target(self.path, 64).shape, (64, 64, 3))

    def test_con_preserve_aspect_respeta_la_proporcion(self):
        self.assertEqual(canvas_size(self.path, 64, True), (64, 32))
        self.assertEqual(load_target(self.path, 64, preserve_aspect=True).shape, (32, 64, 3))

    def test_tambien_para_imagenes_verticales(self):
        vertical = str(Path(self.dir.name) / "vertical.png")
        Image.new("RGB", (100, 200)).save(vertical)
        self.assertEqual(canvas_size(vertical, 64, True), (32, 64))

    def test_el_render_acepta_alto_distinto_del_ancho(self):
        ind = random_individual(4, np.random.default_rng(0))
        self.assertEqual(render_array(ind, 40, 20).shape, (20, 40, 3))
        self.assertEqual(render_array(ind, 40).shape, (40, 40, 3))  # cuadrado sigue andando

    def test_el_motor_corre_sobre_un_target_rectangular(self):
        target = load_target(self.path, 32, preserve_aspect=True)
        config = {"triangles": 4, "population_size": 6, "offspring_size": 6, "seed": 1,
                  "stop": {"max_generations": 5}}
        result = engine.run(config, target)
        self.assertEqual(result.generations, 5)
        # el fitness se evalúa contra el target rectangular, sin deformarlo
        self.assertEqual(FitnessEvaluator(target).rmse(result.best),
                         FitnessEvaluator(target).rmse(result.best))
        self.assertGreater(result.best.fitness, 0.0)


class TestInitialization(unittest.TestCase):
    """Inicialización informada: arrancar de una grilla con los colores del target."""

    def setUp(self):
        self.target = np.zeros((CANVAS, CANVAS, 3))
        self.target[:, : CANVAS // 2] = [200, 30, 30]   # izquierda roja
        self.target[:, CANVAS // 2 :] = [30, 30, 200]   # derecha azul

    def test_los_genes_quedan_en_el_dominio(self):
        ind = grid_individual(9, np.random.default_rng(0), self.target)
        self.assertEqual(len(ind.genes), 9 * GENES_PER_TRIANGLE)
        self.assertTrue(np.all((ind.genes >= 0) & (ind.genes <= 1)))

    def test_muestrea_el_color_del_target_en_cada_celda(self):
        ind = grid_individual(4, np.random.default_rng(0), self.target)
        blocks = ind.genes.reshape(-1, GENES_PER_TRIANGLE)
        # grilla 2x2: los de la columna izquierda rojos, los de la derecha azules
        izquierda, derecha = blocks[[0, 2]], blocks[[1, 3]]
        self.assertTrue(np.all(izquierda[:, 6] > izquierda[:, 8]))
        self.assertTrue(np.all(derecha[:, 8] > derecha[:, 6]))

    def test_dos_individuos_no_salen_iguales(self):
        """Sin diversidad inicial la población no puede evolucionar."""
        rng = np.random.default_rng(0)
        a, b = grid_individual(9, rng, self.target), grid_individual(9, rng, self.target)
        self.assertFalse(np.array_equal(a.genes, b.genes))

    def test_arranca_mejor_que_la_inicializacion_al_azar(self):
        rng = np.random.default_rng(0)
        evaluator = FitnessEvaluator(self.target)
        grid = max(evaluator(grid_individual(9, rng, self.target)) for _ in range(10))
        azar = max(evaluator(random_individual(9, rng)) for _ in range(10))
        self.assertGreater(grid, azar)

    def test_grid_necesita_el_target(self):
        with self.assertRaises(ValueError):
            grid_individual(4, np.random.default_rng(0), None)


class TestArtifact(unittest.TestCase):
    """El archivo de triángulos tiene que alcanzar por sí solo para reconstruir
    la imagen: si no, no es una compresión sino un volcado de genes."""

    def setUp(self):
        self.individual = random_individual(6, np.random.default_rng(11))
        self.document = artifact.build(self.individual, CANVAS, CANVAS, (255, 255, 255))

    def test_el_documento_declara_todo_lo_necesario_para_reconstruir(self):
        self.assertEqual(self.document["format_version"], artifact.FORMAT_VERSION)
        self.assertEqual(self.document["canvas"]["width"], CANVAS)
        self.assertEqual(self.document["canvas"]["background"], [255, 255, 255])
        self.assertEqual(self.document["compositing"], "rgba-source-over")
        self.assertIn("paint_order", self.document)
        self.assertEqual(len(self.document["triangles"]), 6)

    def test_ida_y_vuelta_por_disco_reproduce_la_imagen_exacta(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "triangles.json"
            artifact.save(self.document, path)
            recovered = artifact.load(path)
        self.assertEqual(recovered, self.document)
        np.testing.assert_array_equal(
            np.asarray(artifact.render(recovered)), np.asarray(artifact.render(self.document))
        )

    def test_el_documento_y_el_render_del_individuo_dan_la_misma_imagen(self):
        np.testing.assert_array_equal(
            np.asarray(artifact.render(self.document)), np.asarray(render(self.individual, CANVAS))
        )

    def test_el_orden_de_pintado_es_parte_del_formato(self):
        """Invertir el array da otra imagen: por eso el orden está declarado."""
        flipped = {**self.document, "triangles": list(reversed(self.document["triangles"]))}
        self.assertFalse(
            np.array_equal(np.asarray(artifact.render(flipped)), np.asarray(artifact.render(self.document)))
        )

    def test_rechaza_documentos_que_no_puede_interpretar(self):
        for broken in ({"format_version": 99}, {"compositing": "multiply"}):
            with self.subTest(broken):
                with self.assertRaises(ValueError):
                    artifact.render({**self.document, **broken})


class TestFitness(unittest.TestCase):
    def test_fitness_del_target_contra_si_mismo_es_1(self):
        target = solid_target(255.0)
        evaluator = FitnessEvaluator(target)
        self.assertAlmostEqual(evaluator(Individual(np.zeros(0))), 1.0)

    def test_fitness_esta_acotado_y_es_peor_lejos_del_target(self):
        evaluator = FitnessEvaluator(solid_target(0.0))
        blanco = evaluator(Individual(np.zeros(0)))  # canvas blanco vs. target negro
        self.assertAlmostEqual(blanco, 0.0)

    def test_el_fitness_se_cachea(self):
        evaluator = FitnessEvaluator(solid_target(255.0))
        ind = random_individual(3, np.random.default_rng(0))
        evaluator(ind)
        evaluator(ind)
        self.assertEqual(evaluator.evaluations, 1)


class TestOperators(unittest.TestCase):
    def test_elite_devuelve_exactamente_k(self):
        pop = [Individual(np.zeros(1), fitness=f) for f in (0.1, 0.9, 0.5)]
        self.assertEqual([i.fitness for i in elite(pop, 2, make_ctx())], [0.9, 0.5])
        self.assertEqual(len(elite(pop, 7, make_ctx())), 7)

    def test_cruza_produce_genotipos_validos(self):
        rng = np.random.default_rng(1)
        a, b = random_individual(5, rng), random_individual(5, rng)
        for child in one_point(a, b, make_ctx()):
            self.assertEqual(len(child.genes), 5 * GENES_PER_TRIANGLE)
            self.assertTrue(np.all((child.genes >= 0) & (child.genes <= 1)))
            self.assertIsNone(child.fitness)

    def test_cruza_por_triangulo_no_parte_bloques(self):
        rng = np.random.default_rng(2)
        a, b = random_individual(4, rng), random_individual(4, rng)
        ctx = make_ctx(crossover_granularity="triangle")
        for _ in range(20):
            child, _ = one_point(a, b, ctx)
            blocks = child.genes.reshape(-1, GENES_PER_TRIANGLE)
            for block in blocks:
                from_a = np.any(np.all(a.genes.reshape(-1, GENES_PER_TRIANGLE) == block, axis=1))
                from_b = np.any(np.all(b.genes.reshape(-1, GENES_PER_TRIANGLE) == block, axis=1))
                self.assertTrue(from_a or from_b)

    def test_mutacion_respeta_el_dominio_e_invalida_el_cache(self):
        ind = random_individual(3, np.random.default_rng(3))
        ind.fitness = 0.5
        ctx = make_ctx(mutation_rate=1.0, mutation_sigma=5.0)
        gene(ind, ctx)
        self.assertTrue(np.all((ind.genes >= 0) & (ind.genes <= 1)))
        self.assertIsNone(ind.fitness)


class TestSelectionMethods(unittest.TestCase):
    """Los 6 métodos del enunciado (Step 5)."""

    def setUp(self):
        self.pop = [Individual(np.zeros(1), fitness=f) for f in np.linspace(0.1, 0.9, 10)]
        # temperatura baja y fija: así boltzmann también ejerce presión de selección
        self.ctx = Context(
            rng=np.random.default_rng(0),
            params={"boltzmann": {"t0": 0.2, "tmin": 0.2, "k": 0.0}},
            generation=1,
            max_generations=100,
        )

    def test_todos_devuelven_exactamente_k(self):
        for name, method in selection_mod.METHODS.items():
            with self.subTest(name):
                self.assertEqual(len(method(self.pop, 17, self.ctx)), 17)

    def test_todos_tienen_presion_de_seleccion(self):
        """El fitness medio de los elegidos supera al de la población.

        Se pide k < N: con k >= N elite devuelve a todos y no hay presión posible.
        Se promedian muchas selecciones para que no dependa del azar.
        """
        media_poblacion = np.mean([i.fitness for i in self.pop])
        for name, method in selection_mod.METHODS.items():
            with self.subTest(name):
                elegidos = [ind for _ in range(200) for ind in method(self.pop, 5, self.ctx)]
                self.assertGreater(np.mean([i.fitness for i in elegidos]), media_poblacion)

    def test_torneo_deterministico_con_m_igual_a_n_es_elite(self):
        ctx = Context(rng=np.random.default_rng(0), params={"tournament": {"m": len(self.pop)}})
        elegidos = selection_mod.tournament_det(self.pop, 5, ctx)
        self.assertTrue(all(i.fitness == 0.9 for i in elegidos))

    def test_boltzmann_pasa_de_exploracion_a_explotacion(self):
        """T alta: casi uniforme (explora). T baja: casi elite (explota)."""
        def fitness_medio(temperature):
            ctx = Context(
                rng=np.random.default_rng(0),
                params={"boltzmann": {"t0": temperature, "tmin": temperature, "k": 0.0}},
            )
            return np.mean([i.fitness for i in selection_mod.boltzmann(self.pop, 2000, ctx)])

        media_poblacion = np.mean([i.fitness for i in self.pop])
        self.assertAlmostEqual(fitness_medio(1000.0), media_poblacion, places=1)
        self.assertGreater(fitness_medio(0.05), 0.8)

    def test_boltzmann_enfria_con_las_generaciones(self):
        params = {"boltzmann": {"t0": 100.0, "tmin": 1.0, "k": 0.05}}
        temps = [
            selection_mod._temperature(Context(rng=np.random.default_rng(0), params=params, generation=g))
            for g in (0, 10, 100)
        ]
        self.assertEqual(temps, sorted(temps, reverse=True))
        self.assertAlmostEqual(temps[0], 100.0)  # T(0) = T0
        self.assertLess(temps[-1], 2.0)  # tiende a Tmin = 1

    def test_seleccion_combinada_respeta_el_ratio(self):
        selector = selection_mod.build({"method_a": "elite", "method_b": "elite", "a_ratio": 0.25})
        self.assertEqual(len(selector(self.pop, 20, self.ctx)), 20)
        with self.assertRaises(ValueError):
            selection_mod.build({"method_a": "elite", "method_b": "elite", "a_ratio": 2})

    def test_metodo_inexistente_falla_con_mensaje_util(self):
        with self.assertRaises(ValueError):
            selection_mod.get("no_existe")


class TestCrossoverMethods(unittest.TestCase):
    """Los 4 métodos del enunciado (Step 6)."""

    def test_todos_producen_genotipos_validos_con_genes_de_los_padres(self):
        rng = np.random.default_rng(1)
        a, b = random_individual(6, rng), random_individual(6, rng)
        pool = np.concatenate([a.genes, b.genes])
        for granularity in ("gene", "triangle"):
            ctx = make_ctx(crossover_granularity=granularity)
            for name, method in crossover_mod.METHODS.items():
                with self.subTest(name=name, granularity=granularity):
                    for child in method(a, b, ctx):
                        self.assertEqual(len(child.genes), 6 * GENES_PER_TRIANGLE)
                        self.assertTrue(np.all(np.isin(child.genes, pool)))
                        self.assertIsNone(child.fitness)

    def test_los_hijos_son_complementarios(self):
        rng = np.random.default_rng(2)
        a, b = random_individual(4, rng), random_individual(4, rng)
        for name, method in crossover_mod.METHODS.items():
            with self.subTest(name):
                c1, c2 = method(a, b, make_ctx())
                # cada posición viene de un padre distinto en cada hijo
                de_a = c1.genes == a.genes
                np.testing.assert_array_equal(de_a, c2.genes == b.genes)

    def test_la_cruza_espacial_parte_por_geometria_y_no_por_indice(self):
        """Dos triángulos en la misma región van juntos, estén donde estén en la lista."""
        rng = np.random.default_rng(9)
        a, b = random_individual(8, rng), random_individual(8, rng)
        # todos los triángulos de A al mismo lado -> el corte no puede separarlos
        blocks = a.genes.reshape(-1, GENES_PER_TRIANGLE)
        blocks[:, [0, 2, 4]] = 0.05
        blocks[:, [1, 3, 5]] = 0.05
        ctx = make_ctx()
        for _ in range(20):
            child, _ = crossover_mod.spatial(a, b, ctx)
            child_blocks = child.genes.reshape(-1, GENES_PER_TRIANGLE)
            from_a = np.all(child_blocks == blocks, axis=1)
            # o vienen todos de A (el corte los dejó afuera) o ninguno
            self.assertIn(int(from_a.sum()), (0, 8))

    def test_la_cruza_espacial_ignora_la_granularidad_configurada(self):
        """Partir un triángulo al medio no tiene sentido geométrico."""
        rng = np.random.default_rng(10)
        a, b = random_individual(5, rng), random_individual(5, rng)
        ctx = make_ctx(crossover_granularity="gene")
        for _ in range(20):
            child, _ = crossover_mod.spatial(a, b, ctx)
            for block in child.genes.reshape(-1, GENES_PER_TRIANGLE):
                in_a = np.any(np.all(a.genes.reshape(-1, GENES_PER_TRIANGLE) == block, axis=1))
                in_b = np.any(np.all(b.genes.reshape(-1, GENES_PER_TRIANGLE) == block, axis=1))
                self.assertTrue(in_a or in_b)

    def test_la_cruza_no_muta_a_los_padres(self):
        rng = np.random.default_rng(3)
        a, b = random_individual(4, rng), random_individual(4, rng)
        original = a.genes.copy()
        for method in crossover_mod.METHODS.values():
            method(a, b, make_ctx())
        np.testing.assert_array_equal(a.genes, original)


class TestMutationMethods(unittest.TestCase):
    """Los 4 métodos del enunciado (Step 7)."""

    def test_todos_respetan_el_dominio_e_invalidan_el_cache(self):
        ctx = make_ctx(mutation_rate=1.0, mutation_sigma=5.0)
        for name, method in mutation_mod.METHODS.items():
            with self.subTest(name):
                ind = random_individual(4, np.random.default_rng(4))
                ind.fitness = 0.5
                method(ind, ctx)
                self.assertTrue(np.all((ind.genes >= 0) & (ind.genes <= 1)))
                self.assertIsNone(ind.fitness)

    def test_gen_muta_a_lo_sumo_un_gen(self):
        ctx = make_ctx(mutation_rate=1.0, mutation_sigma=0.3)
        ind = random_individual(4, np.random.default_rng(5))
        before = ind.genes.copy()
        mutation_mod.gene(ind, ctx)
        self.assertLessEqual(int((ind.genes != before).sum()), 1)

    def test_uniforme_muta_mas_genes_que_gen(self):
        ctx = make_ctx(mutation_rate=0.5, mutation_sigma=0.3)
        ind = random_individual(10, np.random.default_rng(6))
        before = ind.genes.copy()
        mutation_mod.uniform(ind, ctx)
        self.assertGreater(int((ind.genes != before).sum()), 1)

    def test_no_uniforme_muta_menos_al_final_de_la_corrida(self):
        def cambiados(generation):
            ctx = Context(
                rng=np.random.default_rng(7),
                params={"mutation_rate": 1.0, "mutation_sigma": 0.3, "mutation_decay_floor": 0.0},
                generation=generation,
                max_generations=100,
            )
            ind = random_individual(20, np.random.default_rng(8))
            before = ind.genes.copy()
            mutation_mod.non_uniform(ind, ctx)
            return int((ind.genes != before).sum())

        self.assertGreater(cambiados(0), cambiados(90))


class TestZOrderMutation(unittest.TestCase):
    """Mutación de z-order: el método `zorder` y la perilla `mutation_zorder_rate`.

    El locus del cromosoma es el z-order, y ninguna de las cuatro mutaciones que
    perturban valores puede cambiarlo. Este operador es el único que se mueve en
    esa dimensión.
    """

    def _individuo(self, n=4):
        # cada triángulo lleva un valor distinto y reconocible
        genes = np.concatenate([np.full(GENES_PER_TRIANGLE, i / 10) for i in range(n)])
        return Individual(genes)

    # --- el método registrado en METHODS -------------------------------------

    def test_registrado_en_methods(self):
        self.assertIn("zorder", mutation_mod.METHODS)
        self.assertIs(mutation_mod.METHODS["zorder"], mutation_mod.zorder)

    def test_metodo_reordena_sin_tocar_valores(self):
        antes = self._individuo()
        despues = mutation_mod.get("zorder")(antes.copy(), make_ctx())

        bloques_antes = antes.genes.reshape(-1, GENES_PER_TRIANGLE)
        bloques_despues = despues.genes.reshape(-1, GENES_PER_TRIANGLE)
        # mismo multiconjunto de triángulos: sólo cambió el orden de pintado
        self.assertCountEqual([tuple(b) for b in bloques_antes],
                              [tuple(b) for b in bloques_despues])
        # ningún triángulo quedó partido
        for bloque in bloques_despues:
            self.assertEqual(len(set(bloque)), 1)
        # exactamente dos posiciones cambiaron
        distintas = sum(1 for a, b in zip(bloques_antes, bloques_despues)
                        if not np.array_equal(a, b))
        self.assertEqual(distintas, 2)

    def test_metodo_no_se_envuelve_con_la_perilla(self):
        """`get('zorder')` no puede aplicar el swap dos veces: lo desharía."""
        self.assertIs(mutation_mod.get("zorder"), mutation_mod.zorder)

    def test_metodo_con_un_solo_triangulo_no_hace_nada(self):
        ind = self._individuo(n=1)
        antes = ind.genes.copy()
        mutation_mod.get("zorder")(ind, make_ctx())
        np.testing.assert_array_equal(ind.genes, antes)

    # --- la perilla ortogonal ------------------------------------------------

    def test_apagada_por_defecto_no_cambia_nada(self):
        """Sin la perilla, `get` tiene que devolver exactamente el método de siempre."""
        antes = self._individuo()
        ctx = make_ctx(mutation_rate=0.0)          # sin z-order y sin perturbación
        despues = mutation_mod.get("uniform")(antes.copy(), ctx)
        np.testing.assert_array_equal(despues.genes, antes.genes)

    def test_intercambia_dos_triangulos_enteros(self):
        antes = self._individuo()
        ctx = make_ctx(mutation_rate=0.0, mutation_zorder_rate=1.0)
        despues = mutation_mod.get("uniform")(antes.copy(), ctx)

        bloques_antes = antes.genes.reshape(-1, GENES_PER_TRIANGLE)
        bloques_despues = despues.genes.reshape(-1, GENES_PER_TRIANGLE)
        # el multiconjunto de triángulos es el mismo: sólo cambió el orden
        self.assertCountEqual([tuple(b) for b in bloques_antes],
                              [tuple(b) for b in bloques_despues])
        # y ningún triángulo quedó partido: cada bloque sigue siendo homogéneo
        for bloque in bloques_despues:
            self.assertEqual(len(set(bloque)), 1)
        # exactamente dos posiciones cambiaron
        distintas = sum(1 for a, b in zip(bloques_antes, bloques_despues)
                        if not np.array_equal(a, b))
        self.assertEqual(distintas, 2)

    def test_invalida_el_cache_porque_cambia_la_imagen(self):
        ind = self._individuo()
        ind.fitness = 0.5
        mutation_mod.get("uniform")(ind, make_ctx(mutation_rate=0.0, mutation_zorder_rate=1.0))
        self.assertIsNone(ind.fitness)

    def test_se_aplica_sobre_cualquier_metodo(self):
        for nombre in mutation_mod.METHODS:
            with self.subTest(nombre):
                ind = self._individuo()
                ctx = make_ctx(mutation_rate=0.0, mutation_zorder_rate=1.0,
                               mutation_genes=1, max_generations=10)
                antes = ind.genes.copy()
                mutation_mod.get(nombre)(ind, ctx)
                self.assertFalse(np.array_equal(ind.genes, antes))

    def test_con_un_solo_triangulo_no_hace_nada(self):
        ind = self._individuo(n=1)
        antes = ind.genes.copy()
        mutation_mod.get("uniform")(ind, make_ctx(mutation_rate=0.0, mutation_zorder_rate=1.0))
        np.testing.assert_array_equal(ind.genes, antes)


class TestValidaciones(unittest.TestCase):
    """Rangos que la cátedra acota explícitamente y antes no chequeábamos."""

    def test_torneo_probabilistico_rechaza_threshold_fuera_de_rango(self):
        poblacion = [Individual(np.zeros(10), fitness=f) for f in (0.1, 0.9)]
        for malo in (0.3, 0.49, 1.01):
            with self.subTest(malo):
                ctx = make_ctx(tournament={"threshold": malo})
                with self.assertRaises(ValueError):
                    selection_mod.tournament_prob(poblacion, 2, ctx)

    def test_torneo_probabilistico_acepta_el_rango_valido(self):
        poblacion = [Individual(np.zeros(10), fitness=f) for f in (0.1, 0.9)]
        for bueno in (0.5, 0.75, 1.0):
            with self.subTest(bueno):
                ctx = make_ctx(tournament={"threshold": bueno})
                self.assertEqual(len(selection_mod.tournament_prob(poblacion, 3, ctx)), 3)

    def test_cruza_anular_usa_el_largo_maximo_de_la_catedra(self):
        """L en [0, ceil(S/2)]. Con S impar, n//2 se quedaba uno corto."""
        import math
        for n in (11, 12, 25):
            with self.subTest(n=n):
                vistos = set()
                a = Individual(np.zeros(n)); b = Individual(np.ones(n))
                for semilla in range(400):
                    ctx = Context(rng=np.random.default_rng(semilla),
                                  params={"crossover_granularity": "gene"})
                    hijo, _ = crossover_mod.annular(a, b, ctx)
                    vistos.add(int(hijo.genes.sum()))   # cuántas unidades vinieron de b
                self.assertEqual(max(vistos), math.ceil(n / 2))


class TestReplacement(unittest.TestCase):
    """Ambas estrategias de supervivencia (Step 8)."""

    def setUp(self):
        self.ctx = make_ctx()
        self.ctx.survivor_selector = elite
        self.parents = [Individual(np.zeros(1), fitness=f) for f in (0.9, 0.8, 0.7, 0.6)]
        self.children = [Individual(np.zeros(1), fitness=f) for f in (0.5, 0.4)]

    def test_aditiva_hace_competir_a_padres_e_hijos(self):
        survivors = additive(self.parents, self.children, 4, self.ctx)
        self.assertEqual([i.fitness for i in survivors], [0.9, 0.8, 0.7, 0.6])

    def test_exclusiva_con_k_menor_que_n_mete_a_todos_los_hijos(self):
        survivors = exclusive(self.parents, self.children, 4, self.ctx)
        self.assertEqual([i.fitness for i in survivors], [0.5, 0.4, 0.9, 0.8])

    def test_exclusiva_con_k_mayor_o_igual_que_n_selecciona_entre_los_hijos(self):
        children = [Individual(np.zeros(1), fitness=f) for f in (0.5, 0.4, 0.3, 0.2, 0.1)]
        survivors = exclusive(self.parents, children, 4, self.ctx)
        self.assertEqual([i.fitness for i in survivors], [0.5, 0.4, 0.3, 0.2])


class TestStopping(unittest.TestCase):
    """Criterios de corte (Step 9)."""

    def _state(self, **kwargs):
        base = dict(
            generation=1, best_fitness=0.5, best_global_fitness=0.5, mean_fitness=0.5,
            std_fitness=0.0, diversity=0.0, stalled=0, structure_stable=0,
            share_unchanged=0.0, evaluations=0, elapsed=0.0,
        )
        return engine.GenerationRecord(**{**base, **kwargs})

    def test_los_criterios_apagados_no_cortan(self):
        cfg = stopping.resolve({"max_generations": 100})
        self.assertIsNone(stopping.check(cfg, self._state(generation=50, stalled=999)))

    def test_cada_criterio_corta_cuando_se_configura(self):
        casos = [
            ({"max_generations": 10}, {"generation": 10}, "max_generations"),
            ({"max_seconds": 5}, {"elapsed": 5.1}, "max_seconds"),
            ({"target_fitness": 0.9}, {"best_fitness": 0.95}, "target_fitness"),
            ({"stall_generations": 20}, {"stalled": 20}, "content"),
            ({"structure_generations": 30}, {"structure_stable": 30}, "structure"),
        ]
        for stop_cfg, state_kwargs, expected in casos:
            with self.subTest(expected):
                cfg = stopping.resolve({"max_generations": 10_000, **stop_cfg})
                self.assertEqual(stopping.check(cfg, self._state(**state_kwargs)), expected)


class TestEngine(unittest.TestCase):
    def test_el_mejor_fitness_es_monotono_y_mejora(self):
        target = solid_target(0.0)  # target negro: hay que taparlo con triángulos
        config = {
            "triangles": 5,
            "canvas_size": CANVAS,
            "population_size": 12,
            "offspring_size": 12,
            "seed": 7,
            "stop": {"max_generations": 30},
        }
        result = engine.run(config, target)
        best = [r.best_fitness for r in result.history]
        self.assertEqual(result.generations, 30)
        self.assertEqual(result.stop_reason, "max_generations")
        self.assertEqual(best, sorted(best))  # elitismo: nunca baja
        self.assertGreater(best[-1], best[0])

    def test_el_mejor_global_es_monotono_aunque_la_poblacion_pierda_al_mejor(self):
        """Con supervivencia exclusiva los hijos desplazan a los padres, así que el
        mejor de la población puede bajar. `best_global_fitness` es el acumulado y
        no puede bajar nunca: es el que hay que graficar como curva de convergencia."""
        config = {
            "triangles": 5,
            "canvas_size": CANVAS,
            "population_size": 10,
            "offspring_size": 10,
            "replacement": "exclusive",
            "selection_survivors": "elite",
            "seed": 3,
            "stop": {"max_generations": 40},
        }
        result = engine.run(config, solid_target(0.0))
        poblacion = [r.best_fitness for r in result.history]
        global_ = [r.best_global_fitness for r in result.history]

        self.assertEqual(global_, sorted(global_))
        self.assertAlmostEqual(global_[-1], result.best.fitness)
        # el mejor global domina siempre al de la población
        for actual, acumulado in zip(poblacion, global_):
            self.assertLessEqual(actual, acumulado + 1e-12)

    def test_corta_por_estructura_cuando_la_poblacion_se_congela(self):
        """Sin cruza ni mutación la población no cambia: corta por estructura."""
        config = {
            "triangles": 3,
            "canvas_size": CANVAS,
            "population_size": 8,
            "offspring_size": 8,
            "crossover_rate": 0.0,
            "mutation_rate": 0.0,
            "seed": 2,
            "stop": {"max_generations": 200, "structure_generations": 5},
        }
        result = engine.run(config, solid_target(0.0))
        self.assertEqual(result.stop_reason, "structure")
        self.assertLess(result.generations, 200)

    def test_corre_con_todas_las_combinaciones_de_metodos(self):
        """Smoke test: cualquier combinación configurable arranca y produce hijos."""
        import itertools

        from ga import crossover as cx, mutation as mu, replacement as rp

        combos = itertools.zip_longest(
            selection_mod.METHODS, cx.METHODS, mu.METHODS, rp.METHODS, fillvalue=None
        )
        for sel, cross, mut, rep in combos:
            config = {
                "triangles": 3,
                "canvas_size": CANVAS,
                "population_size": 8,
                "offspring_size": 8,
                "selection_parents": sel or "elite",
                "selection_survivors": sel or "elite",
                "crossover": cross or "one_point",
                "mutation": mut or "gene",
                "replacement": rep or "additive",
                "seed": 3,
                "stop": {"max_generations": 5},
            }
            with self.subTest(selection=sel, crossover=cross, mutation=mut, replacement=rep):
                result = engine.run(config, solid_target(0.0))
                self.assertEqual(result.generations, 5)
                self.assertEqual(len(result.history), 5)

    def test_corte_por_fitness_objetivo(self):
        config = {
            "triangles": 3,
            "canvas_size": CANVAS,
            "population_size": 8,
            "offspring_size": 8,
            "seed": 1,
            "stop": {"max_generations": 200, "target_fitness": 0.5},
        }
        result = engine.run(config, solid_target(255.0))
        self.assertEqual(result.stop_reason, "target_fitness")


if __name__ == "__main__":
    unittest.main()
