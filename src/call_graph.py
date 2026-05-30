"""
call_graph.py
-------------
In-memory directed call graph of the workspace codebase.

WHAT IT IS:
    A directed graph where:
    - Nodes = function names (strings)
    - Edges = "function A calls function B" or "imports module C"

    When you are inside function_A and request a completion,
    the call graph tells us: "function_A depends on function_B
    and imports module_C — those are relevant context."

WHY THIS MATTERS FOR COMPLETIONS:
    Semantic similarity finds functions with similar vocabulary.
    Call graph finds functions that are STRUCTURALLY related —
    even if they use completely different words.

    Example: calculate_sharpe_ratio calls rolling_std and portfolio_value.
    If you're completing inside calculate_sharpe_ratio, rolling_std is
    highly relevant — but it might not score high on semantic similarity
    because "rolling" and "std" don't appear in the prefix text yet.
    The call graph catches this.

CYCLE DETECTION:
    Python allows circular imports (with some restrictions).
    Our BFS uses a visited set to prevent infinite traversal.
    This is standard cycle detection — O(V + E) with a visited set.

IMPLEMENTATION:
    We use Python dicts (adjacency list representation).
    No external graph library — this demonstrates you understand
    graph data structures from first principles.

    Adjacency list: O(V + E) space, O(degree) edge lookup
    vs adjacency matrix: O(V²) space — wasteful for sparse graphs

Interview talking point: You chose adjacency list over adjacency matrix
because function call graphs are sparse — most functions call a small
fraction of other functions. You can explain the space complexity tradeoff.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

from src.ast_indexer import CodeChunk

logger = logging.getLogger(__name__)

try:
    from tree_sitter import Language, Parser
    import tree_sitter_python as tspython
    try:
        PY_LANGUAGE = Language(tspython.language(), "python")
    except TypeError:
        PY_LANGUAGE = Language(tspython.language())
    TREE_SITTER_AVAILABLE = True
except ImportError:
    TREE_SITTER_AVAILABLE = False


@dataclass
class GraphNode:
    """
    A node in the call graph.

    Attributes:
        func_name:  Name of the function.
        file_path:  File where this function is defined.
        chunk_id:   Corresponding CodeChunk ID.
        calls:      Set of function names this function calls.
        imports:    Set of module/name imports in this function's scope.
    """
    func_name: str
    file_path: str
    chunk_id: str
    calls: set = field(default_factory=set)
    imports: set = field(default_factory=set)


class CallGraph:
    """
    In-memory directed call graph.

    Storage:
        _nodes: {func_name -> GraphNode}
        _file_to_funcs: {file_path -> [func_name]} for fast file lookup

    Graph traversal:
        BFS from a starting function, up to max_depth hops.
        Returns all function names reachable within max_depth steps.
        Visited set prevents infinite loops on circular dependencies.
    """

    def __init__(self) -> None:
        self._nodes: dict[str, GraphNode] = {}
        self._file_to_funcs: dict[str, list[str]] = {}

    def build_from_chunks(self, chunks: list[CodeChunk]) -> None:
        """
        Build the call graph from a list of CodeChunks.

        For each chunk, parse its source to find:
        - call expressions (function calls)
        - import statements

        Args:
            chunks: All indexed CodeChunks from the workspace.
        """
        if not TREE_SITTER_AVAILABLE:
            logger.warning("Tree-Sitter not available, call graph disabled")
            return

        t_start = time.perf_counter()

        # First pass: register all nodes
        for chunk in chunks:
            node = GraphNode(
                func_name=chunk.func_name,
                file_path=chunk.file_path,
                chunk_id=chunk.chunk_id,
            )
            self._nodes[chunk.func_name] = node

            if chunk.file_path not in self._file_to_funcs:
                self._file_to_funcs[chunk.file_path] = []
            self._file_to_funcs[chunk.file_path].append(chunk.func_name)

        # Second pass: extract calls and imports for each chunk
        for chunk in chunks:
            if chunk.func_name in self._nodes:
                calls, imports = self._extract_calls_and_imports(chunk.source)
                self._nodes[chunk.func_name].calls = calls
                self._nodes[chunk.func_name].imports = imports

        elapsed = (time.perf_counter() - t_start) * 1000
        logger.info(
            f"Call graph built: {len(self._nodes)} nodes in {elapsed:.0f}ms"
        )

    def _extract_calls_and_imports(
        self, source: str
    ) -> tuple[set[str], set[str]]:
        """
        Parse function source to extract call expressions and imports.

        Uses Tree-Sitter to find:
            call nodes: function_call → identifier → name
            import nodes: import_statement, import_from_statement

        Args:
            source: Source text of a single function.

        Returns:
            (calls, imports) — sets of string names.
        """
        calls = set()
        imports = set()

        if not TREE_SITTER_AVAILABLE:
            return calls, imports

        try:
            parser = Parser()
            parser.set_language(PY_LANGUAGE)
            tree = parser.parse(source.encode("utf-8"))
            self._walk_for_calls(tree.root_node, source.encode(), calls, imports)
        except Exception as e:
            logger.debug(f"Call extraction error: {e}")

        return calls, imports

    def _walk_for_calls(
        self,
        node,
        source_bytes: bytes,
        calls: set,
        imports: set,
    ) -> None:
        """Recursively walk AST to collect call and import nodes."""

        if node.type == "call":
            # call → function → identifier OR attribute
            func_node = next(
                (c for c in node.children if c.type in ("identifier", "attribute")),
                None
            )
            if func_node:
                if func_node.type == "identifier":
                    name = source_bytes[
                        func_node.start_byte:func_node.end_byte
                    ].decode("utf-8", errors="replace")
                    calls.add(name)
                elif func_node.type == "attribute":
                    # e.g. np.mean → take "mean" (the method name)
                    attr = next(
                        (c for c in func_node.children if c.type == "identifier"),
                        None
                    )
                    if attr:
                        name = source_bytes[
                            attr.start_byte:attr.end_byte
                        ].decode("utf-8", errors="replace")
                        calls.add(name)

        elif node.type in ("import_statement", "import_from_statement"):
            for child in node.children:
                if child.type in ("dotted_name", "identifier"):
                    name = source_bytes[
                        child.start_byte:child.end_byte
                    ].decode("utf-8", errors="replace")
                    imports.add(name.split(".")[0])  # Take top-level module

        for child in node.children:
            self._walk_for_calls(child, source_bytes, calls, imports)

    def get_related_functions(
        self,
        func_name: str,
        max_depth: int = 2,
    ) -> list[tuple[str, int]]:
        """
        BFS traversal from func_name up to max_depth hops.

        Returns all function names reachable from func_name within
        max_depth steps in the call graph. Includes graph distance
        for use in the knapsack scorer.

        Cycle detection: visited set prevents revisiting nodes,
        which would cause infinite loops on circular call chains.

        Args:
            func_name: Starting function name.
            max_depth: Maximum graph distance to traverse.

        Returns:
            List of (func_name, graph_distance) tuples.
            Sorted by distance ascending (closest first).
        """
        if func_name not in self._nodes:
            return []

        visited = {func_name}
        queue = [(func_name, 0)]  # (name, distance)
        results = []

        while queue:
            current_name, depth = queue.pop(0)  # BFS: FIFO

            if depth > 0:  # Don't include the starting function itself
                results.append((current_name, depth))

            if depth >= max_depth:
                continue

            current_node = self._nodes.get(current_name)
            if not current_node:
                continue

            # Traverse call edges
            for called_name in current_node.calls:
                if called_name in self._nodes and called_name not in visited:
                    visited.add(called_name)
                    queue.append((called_name, depth + 1))

        return results

    def update_node(self, chunk: CodeChunk) -> None:
        """
        Update a single node when its function is modified.

        Called by dirty watcher after a function changes.

        Args:
            chunk: The modified CodeChunk.
        """
        calls, imports = self._extract_calls_and_imports(chunk.source)

        if chunk.func_name in self._nodes:
            self._nodes[chunk.func_name].calls = calls
            self._nodes[chunk.func_name].imports = imports
        else:
            node = GraphNode(
                func_name=chunk.func_name,
                file_path=chunk.file_path,
                chunk_id=chunk.chunk_id,
                calls=calls,
                imports=imports,
            )
            self._nodes[chunk.func_name] = node

            if chunk.file_path not in self._file_to_funcs:
                self._file_to_funcs[chunk.file_path] = []
            if chunk.func_name not in self._file_to_funcs[chunk.file_path]:
                self._file_to_funcs[chunk.file_path].append(chunk.func_name)

        logger.debug(f"Call graph updated for {chunk.func_name}")

    def remove_file(self, file_path: str) -> None:
        """Remove all nodes from a deleted file."""
        funcs = self._file_to_funcs.pop(file_path, [])
        for func_name in funcs:
            self._nodes.pop(func_name, None)

    @property
    def node_count(self) -> int:
        return len(self._nodes)
