"""
tests/test_call_graph.py
------------------------
Tests for the directed call graph and BFS traversal.

Critical behaviors:
1. BFS finds all reachable functions within max_depth
2. Cycle detection prevents infinite loops
3. Graph distance is correctly reported
4. Unknown starting functions return empty results
"""

import pytest
from src.ast_indexer import CodeChunk, make_chunk_id
from src.call_graph import CallGraph, TREE_SITTER_AVAILABLE
import time

pytestmark = pytest.mark.skipif(
    not TREE_SITTER_AVAILABLE,
    reason="tree-sitter not installed"
)


def make_chunk(name: str, source: str, file_path: str = "/test/file.py") -> CodeChunk:
    return CodeChunk(
        chunk_id=make_chunk_id(file_path, name, 0),
        file_path=file_path,
        func_name=name,
        source=source,
        start_line=0,
        end_line=5,
        last_modified=time.time(),
    )


class TestCallGraph:

    def test_build_from_chunks_registers_nodes(self):
        """All function names are registered as graph nodes."""
        chunks = [
            make_chunk("func_a", "def func_a():\n    return func_b()"),
            make_chunk("func_b", "def func_b():\n    return 1"),
        ]
        graph = CallGraph()
        graph.build_from_chunks(chunks)
        assert graph.node_count == 2

    def test_unknown_function_returns_empty(self):
        """BFS on a non-existent function returns empty list."""
        graph = CallGraph()
        result = graph.get_related_functions("nonexistent_func")
        assert result == []

    def test_direct_call_found_at_depth_1(self):
        """A directly called function is found at graph distance 1."""
        chunks = [
            make_chunk("func_a", "def func_a():\n    return func_b()"),
            make_chunk("func_b", "def func_b():\n    return 1"),
        ]
        graph = CallGraph()
        graph.build_from_chunks(chunks)

        related = graph.get_related_functions("func_a", max_depth=2)
        related_dict = dict(related)

        assert "func_b" in related_dict
        assert related_dict["func_b"] == 1

    def test_max_depth_respected(self):
        """BFS does not traverse beyond max_depth."""
        chunks = [
            make_chunk("func_a", "def func_a():\n    return func_b()"),
            make_chunk("func_b", "def func_b():\n    return func_c()"),
            make_chunk("func_c", "def func_c():\n    return 42"),
        ]
        graph = CallGraph()
        graph.build_from_chunks(chunks)

        # With max_depth=1, func_c should NOT appear
        related = graph.get_related_functions("func_a", max_depth=1)
        related_dict = dict(related)

        assert "func_b" in related_dict
        assert "func_c" not in related_dict

    def test_depth_2_traversal(self):
        """BFS at depth 2 finds functions 2 hops away."""
        chunks = [
            make_chunk("func_a", "def func_a():\n    return func_b()"),
            make_chunk("func_b", "def func_b():\n    return func_c()"),
            make_chunk("func_c", "def func_c():\n    return 42"),
        ]
        graph = CallGraph()
        graph.build_from_chunks(chunks)

        related = graph.get_related_functions("func_a", max_depth=2)
        related_dict = dict(related)

        assert "func_b" in related_dict
        assert related_dict["func_b"] == 1
        assert "func_c" in related_dict
        assert related_dict["func_c"] == 2

    def test_cycle_detection_prevents_infinite_loop(self):
        """Circular call chains don't cause infinite BFS loops."""
        # func_a → func_b → func_a (circular)
        chunks = [
            make_chunk("func_a", "def func_a():\n    return func_b()"),
            make_chunk("func_b", "def func_b():\n    return func_a()"),
        ]
        graph = CallGraph()
        graph.build_from_chunks(chunks)

        # This must terminate — no infinite loop
        related = graph.get_related_functions("func_a", max_depth=10)
        assert isinstance(related, list)
        # Should find func_b but not loop forever
        related_dict = dict(related)
        assert "func_b" in related_dict

    def test_isolated_function_returns_empty(self):
        """A function that calls nothing returns no related functions."""
        chunks = [
            make_chunk("isolated", "def isolated():\n    return 42"),
            make_chunk("other", "def other():\n    pass"),
        ]
        graph = CallGraph()
        graph.build_from_chunks(chunks)

        related = graph.get_related_functions("isolated", max_depth=2)
        assert related == []

    def test_update_node_adds_new_function(self):
        """update_node adds a previously unknown function to the graph."""
        graph = CallGraph()
        new_chunk = make_chunk("new_func", "def new_func():\n    return 1")
        graph.update_node(new_chunk)
        assert graph.node_count == 1

    def test_remove_file_cleans_nodes(self):
        """remove_file removes all functions from that file."""
        chunks = [
            make_chunk("func_a", "def func_a(): pass", "/file_a.py"),
            make_chunk("func_b", "def func_b(): pass", "/file_b.py"),
        ]
        graph = CallGraph()
        graph.build_from_chunks(chunks)

        graph.remove_file("/file_a.py")
        assert graph.node_count == 1

    def test_self_not_in_related(self):
        """The starting function itself is not included in results."""
        chunks = [
            make_chunk("func_a", "def func_a():\n    return func_b()"),
            make_chunk("func_b", "def func_b():\n    return 1"),
        ]
        graph = CallGraph()
        graph.build_from_chunks(chunks)

        related = graph.get_related_functions("func_a")
        related_names = [name for name, _ in related]
        assert "func_a" not in related_names
