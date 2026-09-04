"""Sanity checks del motor (los 'Definition of Done' de los steps 1-4)."""

from __future__ import annotations

import unittest

import numpy as np

from ga import engine
from ga.context import Context
from ga.crossover import one_point
from ga.fitness import FitnessEvaluator
from ga.individual import GENES_PER_TRIANGLE, Individual, random_individual
from ga.mutation import gene
from ga.render import render_array
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
