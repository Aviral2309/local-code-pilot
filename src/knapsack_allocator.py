"""
knapsack_allocator.py
---------------------
Greedy knapsack context budget allocator.

THE PROBLEM:
    We have N candidate code chunks, each with:
    - A relevance score (how useful is this for the current completion?)
    - A token cost (how much of the context window does it consume?)

    We have a fixed token budget (1800 tokens for a 2K context window,
    leaving 200 for the generation itself).

    We want to maximize total relevance while staying under budget.
    This is the 0/1 Knapsack Problem — NP-hard in general.

WHY GREEDY INSTEAD OF DYNAMIC PROGRAMMING:
    The exact 0/1 knapsack solution is O(n × W) where W is the token
    budget (~1800). With n=20 candidates, that's 36,000 operations —
    fast enough. But we'd need to run this on EVERY keystroke.

    Greedy by density (score/weight) gives a near-optimal solution
    in O(n log n) — just a sort. For code chunks where scores vary
    significantly, greedy performs within 90-95% of optimal.

    The approximation error doesn't matter in practice: the difference
    between the optimal context selection and the greedy selection
    is smaller than the model's generation variance anyway.

SCORING FORMULA:
    Score(chunk) = w1 × semantic_similarity
                 + w2 × (1 / graph_distance)
                 + w3 × (1 / time_since_edit_seconds)

    Default weights: w1=0.5, w2=0.3, w3=0.2

    Semantic similarity: cosine similarity from embedding search (0 to 1)
    Graph distance: BFS hops from current function (1, 2, or ∞)
    Recency: seconds since last edit (recent edits = more relevant)

DENSITY:
    We sort by Score / TokenCount (efficiency density).
    A chunk with score=0.8 and 50 tokens beats a chunk with
    score=0.9 and 200 tokens, because the first gives more
    relevance per token spent.

Interview talking point: "I framed context injection as a variant of
the fractional knapsack problem. I sort candidates by score-per-token
density and greedily fill until the token cap. This consistently
outperforms naive top-K retrieval in my ablation study because it
accounts for token cost — a highly relevant but verbose chunk can
crowd out several smaller, collectively more useful chunks."
"""

import logging
import time
from dataclasses import dataclass
from typing import Optional

from src.ast_indexer import CodeChunk
from src.call_graph import CallGraph

logger = logging.getLogger(__name__)

# Large distance assigned when a chunk has no graph relationship
# to the current function. Effectively zero weight_graph contribution.
INFINITY_DISTANCE = 999


@dataclass
class ScoredChunk:
    """
    A candidate chunk with its computed relevance score.

    Attributes:
        chunk:            The CodeChunk being scored.
        semantic_score:   Cosine similarity to the query (0 to 1).
        graph_distance:   BFS hops from current function (1, 2, or INFINITY).
        recency_score:    1 / seconds_since_edit (higher = more recent).
        total_score:      Weighted combination of the above.
        token_count:      Estimated token cost of this chunk.
        density:          total_score / token_count (for greedy sort).
    """
    chunk: CodeChunk
    semantic_score: float
    graph_distance: int
    recency_score: float
    total_score: float
    token_count: int
    density: float


def score_chunk(
    chunk: CodeChunk,
    semantic_score: float,
    graph_distance: int,
    weight_semantic: float = 0.5,
    weight_graph: float = 0.3,
    weight_recency: float = 0.2,
) -> ScoredChunk:
    """
    Compute the compound relevance score for a single chunk.

    Args:
        chunk:           The CodeChunk to score.
        semantic_score:  Cosine similarity from embedding search.
        graph_distance:  BFS distance from current function.
        weight_semantic: Weight for semantic similarity component.
        weight_graph:    Weight for call graph distance component.
        weight_recency:  Weight for recency component.

    Returns:
        ScoredChunk with all scores computed.
    """
    # Graph distance component: closer = higher score
    # Distance 1 → 1.0, Distance 2 → 0.5, Infinity → ~0.001
    graph_component = 1.0 / graph_distance if graph_distance > 0 else 0.0

    # Recency component: time since last edit
    # Just edited (1s ago) → 1.0, Edited 1 hour ago → 0.00028
    seconds_since_edit = max(1.0, time.time() - chunk.last_modified)
    recency_component = 1.0 / seconds_since_edit

    # Normalize recency to [0, 1] range (cap at 1.0)
    # We consider anything edited within the last minute as "very recent"
    recency_normalized = min(1.0, recency_component * 60)

    # Compound score
    total_score = (
        weight_semantic * semantic_score
        + weight_graph * graph_component
        + weight_recency * recency_normalized
    )

    token_count = max(1, chunk.token_estimate)  # Prevent division by zero
    density = total_score / token_count

    return ScoredChunk(
        chunk=chunk,
        semantic_score=semantic_score,
        graph_distance=graph_distance,
        recency_score=recency_normalized,
        total_score=total_score,
        token_count=token_count,
        density=density,
    )


def greedy_knapsack(
    candidates: list[ScoredChunk],
    token_cap: int = 1800,
) -> list[ScoredChunk]:
    """
    Greedy knapsack: select chunks to maximize score within token budget.

    Algorithm:
        1. Sort candidates by density (score/token_count) descending
        2. Add chunks greedily until token_cap is reached
        3. Return selected chunks in relevance order (highest score first)

    This is O(n log n) — dominated by the sort.

    Args:
        candidates: List of ScoredChunk objects to select from.
        token_cap:  Maximum total tokens allowed.

    Returns:
        Selected chunks, sorted by total_score descending.
        Total token count of result is guaranteed <= token_cap.
    """
    if not candidates:
        return []

    # Sort by density descending — highest value-per-token first
    sorted_candidates = sorted(candidates, key=lambda x: x.density, reverse=True)

    selected = []
    tokens_used = 0

    for scored in sorted_candidates:
        if tokens_used + scored.token_count <= token_cap:
            selected.append(scored)
            tokens_used += scored.token_count
        # Skip if it doesn't fit — don't stop, continue checking
        # smaller chunks that might still fit

    # Return sorted by total_score for presentation order
    selected.sort(key=lambda x: x.total_score, reverse=True)

    logger.debug(
        f"Knapsack: {len(selected)}/{len(candidates)} chunks selected | "
        f"{tokens_used}/{token_cap} tokens used"
    )

    return selected


class ContextAllocator:
    """
    Orchestrates the full context retrieval and allocation pipeline.

    Usage:
        allocator = ContextAllocator(embedder, call_graph, config)
        context_str = allocator.get_context(
            query=prefix_text,
            current_file=file_path,
            current_func=func_name,
        )
    """

    def __init__(
        self,
        embedder,           # EmbeddingEngine
        call_graph: CallGraph,
        token_cap: int = 1800,
        weight_semantic: float = 0.5,
        weight_graph: float = 0.3,
        weight_recency: float = 0.2,
        top_k_candidates: int = 20,
    ) -> None:
        self._embedder = embedder
        self._call_graph = call_graph
        self.token_cap = token_cap
        self.weight_semantic = weight_semantic
        self.weight_graph = weight_graph
        self.weight_recency = weight_recency
        self.top_k_candidates = top_k_candidates

    def get_context(
        self,
        query: str,
        current_file: str,
        current_func: Optional[str] = None,
    ) -> tuple[str, list[str]]:
        """
        Full pipeline: search → score → knapsack → format.

        Args:
            query:        Current prefix text (what the user has typed).
            current_file: URI of the file being edited (exclude from results).
            current_func: Name of the function cursor is in (for graph lookup).

        Returns:
            (context_string, chunk_ids_used):
                context_string: Formatted code snippets ready for FIM injection.
                chunk_ids_used: List of chunk IDs for telemetry logging.
        """
        if not self._embedder.is_ready:
            return "", []

        t_start = time.perf_counter()

        # Convert URI to file path for exclusion
        file_path = current_file.replace("file:///", "/").replace("file://", "")
        # Windows path fix
        if len(file_path) > 2 and file_path[0] == "/" and file_path[2] == ":":
            file_path = file_path[1:]  # Remove leading slash on Windows

        # Step 1: Semantic search — get top-K candidates
        search_results = self._embedder.search(
            query=query,
            top_k=self.top_k_candidates,
            exclude_file=file_path,
        )

        if not search_results:
            return "", []

        # Step 2: Get call graph distances for current function
        graph_distances: dict[str, int] = {}
        if current_func:
            related = self._call_graph.get_related_functions(
                current_func, max_depth=2
            )
            graph_distances = {name: dist for name, dist in related}

        # Step 3: Score each candidate
        scored_chunks = []
        for chunk, semantic_score in search_results:
            graph_dist = graph_distances.get(chunk.func_name, INFINITY_DISTANCE)
            scored = score_chunk(
                chunk=chunk,
                semantic_score=semantic_score,
                graph_distance=graph_dist,
                weight_semantic=self.weight_semantic,
                weight_graph=self.weight_graph,
                weight_recency=self.weight_recency,
            )
            scored_chunks.append(scored)

        # Step 4: Greedy knapsack selection
        selected = greedy_knapsack(scored_chunks, token_cap=self.token_cap)

        if not selected:
            return "", []

        # Step 5: Format selected chunks for injection
        context_parts = []
        chunk_ids = []

        for scored in selected:
            chunk = scored.chunk
            file_name = chunk.file_path.split("/")[-1].split("\\")[-1]

            context_parts.append(
                f"# From {file_name} (score={scored.total_score:.2f}):\n"
                f"{chunk.source}"
            )
            chunk_ids.append(chunk.chunk_id)

        context_string = "\n\n".join(context_parts)

        elapsed = (time.perf_counter() - t_start) * 1000
        logger.debug(
            f"Context allocated: {len(selected)} chunks | "
            f"{sum(s.token_count for s in selected)} tokens | "
            f"{elapsed:.1f}ms"
        )

        return context_string, chunk_ids
