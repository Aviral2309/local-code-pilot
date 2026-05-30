"""
embedder.py
-----------
Semantic embedding engine for code chunks.

WHAT IT DOES:
    Converts function source code into 384-dimensional dense vectors
    using all-MiniLM-L6-v2. Stores all vectors in a NumPy matrix.
    Answers queries like "which functions are semantically similar
    to this prefix?" via cosine similarity.

WHY ALL-MINILM-L6-V2:
    - 90MB RAM — fits easily in our 8GB budget
    - ~20ms per batch on CPU — fast enough for real-time indexing
    - 384 dimensions — compact but expressive
    - Trained on 1B+ sentence pairs — generalizes well to code
    - No GPU required

WHY NUMPY INSTEAD OF CHROMADB/FAISS:
    ChromaDB and FAISS are excellent but they're separate processes
    that consume significant RAM. On 8GB with Ollama already using
    1.1GB, we can't afford another 500MB process.
    NumPy cosine similarity on 384-dim vectors across 500 functions
    takes <1ms. We don't need a vector database for this scale.

COSINE SIMILARITY:
    cos(A, B) = (A · B) / (||A|| × ||B||)
    Range: [-1, 1], higher = more similar
    We normalize all vectors at insert time so dot product = cosine sim.
    This is O(n) at query time — fast for up to ~10,000 chunks.

Interview talking point: You chose the right tool for the scale.
A vector DB would be premature optimization for <1000 functions.
NumPy is faster, simpler, and uses 10x less RAM at this scale.
"""

import logging
import time
from typing import Optional

import numpy as np

from src.ast_indexer import CodeChunk

logger = logging.getLogger(__name__)

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logger.warning(
        "sentence-transformers not available. "
        "Install: pip install sentence-transformers"
    )


def _make_embedding_text(chunk: CodeChunk) -> str:
    """
    Build the text string to embed for a code chunk.

    We embed more than just the function name — we include the docstring
    and the first few lines of the body. This gives the embedding model
    enough context to understand what the function does semantically,
    not just what it's called.

    Structure:
        Function: {func_name}
        {docstring if present}
        {first 20 lines of source}

    Args:
        chunk: CodeChunk to build text for.

    Returns:
        Text string ready for embedding.
    """
    parts = [f"Function: {chunk.func_name}"]

    if chunk.docstring:
        parts.append(chunk.docstring[:200])  # Cap docstring length

    # First 20 lines of source (captures signature + early body)
    source_lines = chunk.source.splitlines()[:20]
    parts.append("\n".join(source_lines))

    return "\n".join(parts)


class EmbeddingEngine:
    """
    Manages the embedding model and in-memory vector index.

    State:
        _model:      SentenceTransformer model (loaded once at startup).
        _chunks:     List of CodeChunk objects (parallel to _matrix rows).
        _matrix:     NumPy matrix of shape (N, 384) — normalized vectors.
        _id_to_idx:  Dict mapping chunk_id → row index in _matrix.

    Thread safety:
        Embedding is CPU-bound and blocking. All public methods that
        call _model.encode() should be run in a ThreadPoolExecutor
        from the asyncio event loop. See server.py for the pattern.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self._model: Optional["SentenceTransformer"] = None
        self._chunks: list[CodeChunk] = []
        self._matrix: Optional[np.ndarray] = None  # Shape: (N, 384)
        self._id_to_idx: dict[str, int] = {}

    def load_model(self) -> None:
        """
        Load the embedding model into memory.

        This is a blocking operation (~2-3 seconds on first call).
        Call this once at server startup inside a ThreadPoolExecutor.
        Subsequent calls are instant (model is already loaded).
        """
        if self._model is not None:
            return  # Already loaded

        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            logger.error("sentence-transformers not installed")
            return

        t_start = time.perf_counter()
        logger.info(f"Loading embedding model: {self.model_name}")

        self._model = SentenceTransformer(
            self.model_name,
            device="cpu",  # Force CPU — no GPU on Iris Xe
        )

        elapsed = (time.perf_counter() - t_start) * 1000
        logger.info(f"Embedding model loaded in {elapsed:.0f}ms")

    def _embed_texts(self, texts: list[str]) -> np.ndarray:
        """
        Embed a list of text strings into normalized vectors.

        Normalization: divide each vector by its L2 norm so that
        dot product = cosine similarity. This lets us use fast
        matrix multiplication instead of computing norms at query time.

        Args:
            texts: List of strings to embed.

        Returns:
            NumPy array of shape (len(texts), 384), L2-normalized.
        """
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        vectors = self._model.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,  # L2 normalize — dot product = cosine sim
        )
        return vectors.astype(np.float32)

    def index_chunks(self, chunks: list[CodeChunk]) -> None:
        """
        Build the initial vector index from a list of chunks.

        Call this once at startup after parsing the workspace.
        For incremental updates, use update_chunk() and remove_chunk().

        Args:
            chunks: All CodeChunks from the workspace.
        """
        if not chunks:
            logger.info("No chunks to index")
            return

        if self._model is None:
            logger.warning("Model not loaded, skipping indexing")
            return

        t_start = time.perf_counter()

        texts = [_make_embedding_text(c) for c in chunks]
        vectors = self._embed_texts(texts)

        self._chunks = list(chunks)
        self._matrix = vectors
        self._id_to_idx = {c.chunk_id: i for i, c in enumerate(self._chunks)}

        # Mark all as clean (just embedded)
        for chunk in self._chunks:
            chunk.is_dirty = False

        elapsed = (time.perf_counter() - t_start) * 1000
        logger.info(
            f"Indexed {len(chunks)} chunks in {elapsed:.0f}ms "
            f"| matrix shape: {self._matrix.shape}"
        )

    def update_chunk(self, chunk: CodeChunk) -> None:
        """
        Re-embed a single chunk (dirty-bit update).

        Called by the dirty watcher when a specific function changes.
        Instead of re-embedding the entire workspace, we:
        1. Embed just this one function (~20ms)
        2. Replace its row in the matrix
        3. Update the index mapping

        This is the O(1) update that makes dirty-bit indexing fast.

        Args:
            chunk: The modified CodeChunk to re-embed.
        """
        if self._model is None:
            return

        t_start = time.perf_counter()

        text = _make_embedding_text(chunk)
        vector = self._embed_texts([text])[0]  # Shape: (384,)

        if chunk.chunk_id in self._id_to_idx:
            # Replace existing row in matrix
            idx = self._id_to_idx[chunk.chunk_id]
            self._matrix[idx] = vector
            self._chunks[idx] = chunk
        else:
            # New chunk — append to matrix
            if self._matrix is None:
                self._matrix = vector.reshape(1, -1)
                self._chunks = [chunk]
            else:
                self._matrix = np.vstack([self._matrix, vector.reshape(1, -1)])
                self._chunks.append(chunk)
            self._id_to_idx[chunk.chunk_id] = len(self._chunks) - 1

        chunk.is_dirty = False

        elapsed = (time.perf_counter() - t_start) * 1000
        logger.debug(f"Re-embedded {chunk.display_name} in {elapsed:.0f}ms")

    def remove_chunk(self, chunk_id: str) -> None:
        """
        Remove a deleted function from the index.

        Uses row deletion from NumPy matrix via np.delete().
        Rebuilds the id_to_idx mapping after deletion.

        Args:
            chunk_id: chunk_id of the chunk to remove.
        """
        if chunk_id not in self._id_to_idx:
            return

        idx = self._id_to_idx[chunk_id]

        # Remove from matrix and chunk list
        self._matrix = np.delete(self._matrix, idx, axis=0)
        self._chunks.pop(idx)

        # Rebuild index mapping (indices shift after deletion)
        self._id_to_idx = {
            c.chunk_id: i for i, c in enumerate(self._chunks)
        }

        logger.debug(f"Removed chunk {chunk_id} from index")

    def search(
        self,
        query: str,
        top_k: int = 10,
        exclude_file: Optional[str] = None,
    ) -> list[tuple[CodeChunk, float]]:
        """
        Find the top-K most semantically similar chunks to a query string.

        Algorithm:
            1. Embed the query string into a 384-dim vector
            2. Compute dot product of query vector with all stored vectors
               (equivalent to cosine similarity since all are L2-normalized)
            3. Return top-K by score

        The dot product with normalized vectors IS cosine similarity:
            sim(q, d) = q·d / (||q|| × ||d||) = q·d  (if ||d||=1)

        Args:
            query:        Text to search for (typically the code prefix).
            top_k:        Number of results to return.
            exclude_file: Skip chunks from this file path (avoid self-retrieval).

        Returns:
            List of (CodeChunk, similarity_score) tuples, sorted by score desc.
            Empty list if index is empty or model not loaded.
        """
        if self._model is None or self._matrix is None or len(self._chunks) == 0:
            return []

        t_start = time.perf_counter()

        # Embed query
        query_vector = self._embed_texts([query])[0]  # Shape: (384,)

        # Cosine similarity via matrix multiply
        # scores[i] = cosine_sim(query, chunk_i)
        # Shape: (N,)
        scores = self._matrix @ query_vector

        # Build result list, filtering excluded file
        results = []
        for idx, score in enumerate(scores):
            chunk = self._chunks[idx]
            if exclude_file and chunk.file_path == exclude_file:
                continue
            results.append((chunk, float(score)))

        # Sort by score descending
        results.sort(key=lambda x: x[1], reverse=True)
        top_results = results[:top_k]

        elapsed = (time.perf_counter() - t_start) * 1000
        logger.debug(
            f"Search completed in {elapsed:.1f}ms | "
            f"top score: {top_results[0][1]:.3f}" if top_results else
            f"Search completed in {elapsed:.1f}ms | no results"
        )

        return top_results

    @property
    def index_size(self) -> int:
        """Number of chunks currently in the index."""
        return len(self._chunks)

    @property
    def is_ready(self) -> bool:
        """True if model is loaded and index has at least one chunk."""
        return self._model is not None and self._matrix is not None and len(self._chunks) > 0
