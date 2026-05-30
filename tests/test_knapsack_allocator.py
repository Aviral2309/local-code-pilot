"""
tests/test_knapsack_allocator.py
--------------------------------
Tests for the greedy knapsack context allocator.

Key invariants to verify:
1. Total tokens of selected chunks never exceeds token_cap
2. Highest density chunks are selected first
3. Scoring formula produces expected relative ordering
4. Empty input handled gracefully
"""

import time
import pytest
from src.ast_indexer import CodeChunk, make_chunk_id
from src.knapsack_allocator import (
    score_chunk,
    greedy_knapsack,
    ScoredChunk,
    INFINITY_DISTANCE,
)


def make_chunk(name: str, source: str = None, line: int = 0) -> CodeChunk:
    src = source or f"def {name}():\n    pass\n"
    return CodeChunk(
        chunk_id=make_chunk_id("/test/file.py", name, line),
        file_path="/test/file.py",
        func_name=name,
        source=src,
        start_line=line,
        end_line=line + 2,
        last_modified=time.time(),
    )


def make_scored(
    name: str,
    semantic: float,
    graph_dist: int,
    source: str = None,
) -> ScoredChunk:
    chunk = make_chunk(name, source)
    return score_chunk(
        chunk=chunk,
        semantic_score=semantic,
        graph_distance=graph_dist,
    )


class TestScoreChunk:

    def test_high_semantic_score_wins(self):
        """Higher semantic similarity produces higher total score."""
        s1 = make_scored("func_a", semantic=0.9, graph_dist=INFINITY_DISTANCE)
        s2 = make_scored("func_b", semantic=0.3, graph_dist=INFINITY_DISTANCE)
        assert s1.total_score > s2.total_score

    def test_close_graph_distance_wins(self):
        """Closer graph distance produces higher total score."""
        s1 = make_scored("func_a", semantic=0.5, graph_dist=1)
        s2 = make_scored("func_b", semantic=0.5, graph_dist=2)
        assert s1.total_score > s2.total_score

    def test_infinity_distance_lower_than_close(self):
        """Unrelated function (infinity distance) scores lower than related."""
        s_related = make_scored("func_a", semantic=0.5, graph_dist=1)
        s_unrelated = make_scored("func_b", semantic=0.5, graph_dist=INFINITY_DISTANCE)
        assert s_related.total_score > s_unrelated.total_score

    def test_score_always_positive(self):
        """Score is always positive for valid inputs."""
        s = make_scored("func", semantic=0.0, graph_dist=INFINITY_DISTANCE)
        assert s.total_score >= 0

    def test_density_is_score_over_tokens(self):
        """Density = total_score / token_count."""
        s = make_scored("func", semantic=0.8, graph_dist=1)
        expected_density = s.total_score / s.token_count
        assert abs(s.density - expected_density) < 1e-9

    def test_token_count_positive(self):
        """Token count is always at least 1."""
        s = make_scored("func", semantic=0.5, graph_dist=1)
        assert s.token_count >= 1


class TestGreedyKnapsack:

    def test_token_cap_never_exceeded(self):
        """Total tokens of selected chunks never exceeds cap."""
        chunks = [
            make_scored("f1", 0.9, 1, source="def f1():\n" + "    x = 1\n" * 50),
            make_scored("f2", 0.8, 1, source="def f2():\n" + "    y = 2\n" * 50),
            make_scored("f3", 0.7, 2, source="def f3():\n    pass\n"),
            make_scored("f4", 0.6, 2, source="def f4():\n    pass\n"),
        ]
        cap = 100
        selected = greedy_knapsack(chunks, token_cap=cap)
        total_tokens = sum(s.token_count for s in selected)
        assert total_tokens <= cap

    def test_empty_input_returns_empty(self):
        """Empty candidate list returns empty result."""
        assert greedy_knapsack([], token_cap=1800) == []

    def test_single_chunk_fits(self):
        """A single chunk that fits is always selected."""
        chunk = make_scored("func", 0.9, 1)
        selected = greedy_knapsack([chunk], token_cap=1800)
        assert len(selected) == 1

    def test_single_chunk_too_large_excluded(self):
        """A chunk larger than the token cap is excluded."""
        large_source = "def big():\n" + "    x = 1\n" * 1000
        chunk = make_scored("big_func", 0.9, 1, source=large_source)
        selected = greedy_knapsack([chunk], token_cap=10)
        assert len(selected) == 0

    def test_higher_density_selected_first(self):
        """The highest density chunk is always selected if it fits."""
        # high_density: high score, small size
        high_density = make_scored("small_high", 0.95, 1,
                                   source="def small_high():\n    pass\n")
        # low_density: same score but much bigger
        low_density = make_scored("big_low", 0.95, 1,
                                  source="def big_low():\n" + "    x = 1\n" * 100)

        # With a small cap, only high_density fits
        selected = greedy_knapsack(
            [low_density, high_density],
            token_cap=high_density.token_count + 1,
        )
        selected_names = {s.chunk.func_name for s in selected}
        assert "small_high" in selected_names

    def test_result_sorted_by_score(self):
        """Result is sorted by total_score descending."""
        chunks = [
            make_scored("low", 0.3, INFINITY_DISTANCE),
            make_scored("high", 0.9, 1),
            make_scored("mid", 0.6, 2),
        ]
        selected = greedy_knapsack(chunks, token_cap=1800)
        scores = [s.total_score for s in selected]
        assert scores == sorted(scores, reverse=True)

    def test_all_fit_all_selected(self):
        """When all chunks fit within cap, all are selected."""
        chunks = [make_scored(f"f{i}", 0.5, 1) for i in range(5)]
        selected = greedy_knapsack(chunks, token_cap=1800)
        assert len(selected) == 5

    def test_zero_token_cap_returns_empty(self):
        """Zero token cap selects nothing."""
        chunks = [make_scored("func", 0.9, 1)]
        selected = greedy_knapsack(chunks, token_cap=0)
        assert len(selected) == 0

    def test_skips_oversized_keeps_rest(self):
        """Skips chunks that don't fit but continues checking smaller ones."""
        small1 = make_scored("small1", 0.5, 1, source="def small1():\n    pass\n")
        large = make_scored("large", 0.95, 1, source="def large():\n" + "    x=1\n" * 500)
        small2 = make_scored("small2", 0.4, 1, source="def small2():\n    pass\n")

        # Cap that fits small1 and small2 but not large
        cap = small1.token_count + small2.token_count + 5
        selected = greedy_knapsack([large, small1, small2], token_cap=cap)
        selected_names = {s.chunk.func_name for s in selected}

        assert "large" not in selected_names
        assert "small1" in selected_names or "small2" in selected_names
