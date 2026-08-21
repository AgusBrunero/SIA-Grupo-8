from __future__ import annotations

import unittest

from sokoban_solver import (
    SearchResult,
    WEIGHTED_HUNGARIAN_FACTOR,
    compute_dead_squares,
    manhattan_hungarian,
    manhattan_hungarian_weighted,
    manhattan_simple,
    parse_board,
    playable_floor,
    solve_sokoban,
)

LEVEL_1 = [
    "######",
    "#  . #",
    "# #$ #",
    "# @  #",
    "######",
]

CORNER_MAP = [
    "#####",
    "#   #",
    "# $ #",
    "#@ .#",
    "#####",
]


class SolveSokobanMetricsTests(unittest.TestCase):
    def test_bfs_returns_complete_result(self) -> None:
        result = solve_sokoban(LEVEL_1, method="bfs")
        self.assertIsInstance(result, SearchResult)
        self.assertTrue(result.success)
        self.assertFalse(result.timeout)
        self.assertEqual(result.cost, 2)
        self.assertEqual(len(result.path or []), 2)
        self.assertGreaterEqual(result.expanded_nodes, 1)
        self.assertGreaterEqual(result.frontier_nodes_max, result.frontier_nodes_final)
        self.assertGreater(result.elapsed_time, 0)
        self.assertEqual(result.algorithm, "bfs")
        self.assertIsNone(result.heuristic)

    def test_uninformed_ignores_heuristic_name(self) -> None:
        result = solve_sokoban(LEVEL_1, method="dfs", heuristic="hungarian")
        self.assertTrue(result.success)
        self.assertIsNone(result.heuristic)

    def test_timeout_is_reported(self) -> None:
        result = solve_sokoban(
            LEVEL_1,
            method="bfs",
            timeout_seconds=0.0,
        )
        self.assertFalse(result.success)
        self.assertTrue(result.timeout)
        self.assertIsNone(result.cost)
        self.assertIsNone(result.path)

    def test_memory_measurement_is_optional(self) -> None:
        without_mem = solve_sokoban(LEVEL_1, method="bfs", measure_memory=False)
        with_mem = solve_sokoban(LEVEL_1, method="bfs", measure_memory=True)
        self.assertIsNone(without_mem.peak_memory_kb)
        self.assertIsNotNone(with_mem.peak_memory_kb)
        self.assertGreater(with_mem.peak_memory_kb or 0, 0)

    def test_informedness_is_between_zero_and_one_on_level1(self) -> None:
        _, targets, boxes, _ = parse_board(LEVEL_1)
        h = manhattan_simple(boxes, targets)
        h_star = 2
        informedness = h / h_star
        self.assertGreaterEqual(informedness, 0)
        self.assertLessEqual(informedness, 1)
        self.assertEqual(h, 1)

    def test_inner_corner_is_a_dead_square(self) -> None:
        walls, targets, _boxes, player = parse_board(CORNER_MAP)
        floor = playable_floor(walls, player, len(CORNER_MAP), len(CORNER_MAP[0]))
        dead = compute_dead_squares(walls, targets, floor)
        self.assertIn((1, 1), dead)
        self.assertNotIn((3, 2), dead)

    def test_dead_square_pruning_still_solves_level1(self) -> None:
        with_prune = solve_sokoban(LEVEL_1, method="bfs", dead_square_pruning=True)
        without = solve_sokoban(LEVEL_1, method="bfs", dead_square_pruning=False)
        self.assertTrue(with_prune.success)
        self.assertTrue(without.success)
        self.assertEqual(with_prune.cost, without.cost)
        self.assertLessEqual(with_prune.expanded_nodes, without.expanded_nodes)

    def test_weighted_hungarian_is_scaled_and_can_exceed_optimum(self) -> None:
        _, targets, boxes, _ = parse_board(LEVEL_1)
        h_admissible = manhattan_hungarian(boxes, targets)
        h_weighted = manhattan_hungarian_weighted(boxes, targets)
        self.assertEqual(h_weighted, WEIGHTED_HUNGARIAN_FACTOR * h_admissible)
        self.assertGreater(h_weighted, 2)

    def test_astar_weighted_returns_a_path(self) -> None:
        result = solve_sokoban(LEVEL_1, method="astar", heuristic="weighted")
        self.assertTrue(result.success)
        self.assertEqual(result.heuristic, "weighted")
        self.assertIsNotNone(result.path)

    def test_optimal_path_reaches_all_targets(self) -> None:
        result = solve_sokoban(LEVEL_1, method="astar", heuristic="hungarian")
        _, targets, boxes, player = parse_board(LEVEL_1)
        boxes_set = set(boxes)
        for dx, dy in result.path or []:
            nxt = (player[0] + dy, player[1] + dx)
            if nxt in boxes_set:
                boxes_set.remove(nxt)
                boxes_set.add((nxt[0] + dy, nxt[1] + dx))
            player = nxt
        self.assertEqual(frozenset(boxes_set), targets)


if __name__ == "__main__":
    unittest.main()
