"""
dirty_watcher.py
----------------
Dirty-bit cache invalidation engine for incremental re-indexing.

THE PROBLEM IT SOLVES:
    Naive approach: when any file changes, re-parse and re-embed
    the entire workspace. On a 50-file project with 200 functions,
    that's 200 embedding operations × ~20ms each = 4 seconds of
    indexing on every save. The editor freezes. Unusable.

THE DIRTY-BIT SOLUTION:
    When file F is saved:
    1. Parse only file F into new chunks (fast, <20ms)
    2. Diff new chunks against cached chunks for file F
    3. Find exactly which functions changed (the "dirty" set)
    4. Re-embed only dirty functions (~20ms each)
    5. Update only affected call graph edges
    6. Everything else stays cached — untouched

    If you change one line in one function, exactly ONE embedding
    operation runs. The rest of the index stays valid.

THE NAME "DIRTY BIT":
    This term comes directly from operating systems / computer architecture.
    A "dirty bit" (or "modified bit") is a flag on a cache page or
    TLB entry that indicates the data has been modified and needs
    to be written back. We apply the same concept at function granularity.

    In OS: dirty bit on memory page → write to disk on eviction
    Here:  dirty bit on function chunk → re-embed on next index pass

Interview talking point: "I implemented dirty-bit cache invalidation
at function granularity using Tree-Sitter AST diffs. When a file is
saved, I parse only that file, diff the function list against the
cached snapshot, and re-embed only the changed functions. This keeps
re-indexing latency under 50ms regardless of workspace size."
"""

import asyncio
import logging
import time
from pathlib import Path
from typing import Optional

from src.ast_indexer import CodeChunk, diff_chunks, parse_file
from src.call_graph import CallGraph
from src.embedder import EmbeddingEngine

logger = logging.getLogger(__name__)


class DirtyWatcher:
    """
    Manages the per-file chunk cache and orchestrates incremental updates.

    State:
        _file_chunks: {file_path -> [CodeChunk]} — last known state per file
        _embedder:    Reference to the shared EmbeddingEngine
        _call_graph:  Reference to the shared CallGraph
        _dirty_queue: asyncio.Queue of file paths waiting to be re-indexed

    Flow:
        1. LSP server calls on_file_saved(file_path) on every didSave event
        2. DirtyWatcher adds file_path to _dirty_queue
        3. Background worker processes the queue:
           a. Parse the file
           b. Diff against cached chunks
           c. Re-embed dirty chunks
           d. Update call graph edges
           e. Remove deleted chunks from index
    """

    def __init__(
        self,
        embedder: EmbeddingEngine,
        call_graph: CallGraph,
    ) -> None:
        self._embedder = embedder
        self._call_graph = call_graph
        self._file_chunks: dict[str, list[CodeChunk]] = {}
        self._dirty_queue: asyncio.Queue = asyncio.Queue()
        self._worker_task: Optional[asyncio.Task] = None
        self._running = False

    def initialize_from_chunks(self, all_chunks: list[CodeChunk]) -> None:
        """
        Populate the file→chunks cache from initial workspace indexing.

        Call this after parse_workspace() and index_chunks() complete.
        This gives the dirty watcher a baseline to diff against.

        Args:
            all_chunks: All chunks from the initial workspace parse.
        """
        self._file_chunks.clear()
        for chunk in all_chunks:
            if chunk.file_path not in self._file_chunks:
                self._file_chunks[chunk.file_path] = []
            self._file_chunks[chunk.file_path].append(chunk)

        logger.info(
            f"DirtyWatcher initialized with "
            f"{len(self._file_chunks)} files, "
            f"{sum(len(v) for v in self._file_chunks.values())} chunks"
        )

    async def start(self) -> None:
        """Start the background worker that processes the dirty queue."""
        self._running = True
        self._worker_task = asyncio.create_task(self._process_queue())
        logger.info("DirtyWatcher background worker started")

    async def stop(self) -> None:
        """Stop the background worker gracefully."""
        self._running = False
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
        logger.info("DirtyWatcher stopped")

    def on_file_saved(self, file_path: str) -> None:
        """
        Called by the LSP server when a file is saved (textDocument/didSave).

        Adds the file to the dirty queue for background re-indexing.
        Non-blocking — returns immediately. The actual work happens
        in the background worker.

        Args:
            file_path: Absolute path to the saved file.
        """
        if not file_path.endswith(".py"):
            return

        try:
            self._dirty_queue.put_nowait(file_path)
            logger.debug(f"Queued for re-index: {Path(file_path).name}")
        except asyncio.QueueFull:
            logger.warning(f"Dirty queue full, skipping: {file_path}")

    async def _process_queue(self) -> None:
        """
        Background worker: process files from the dirty queue.

        Runs indefinitely until stop() is called.
        Uses asyncio.sleep(0) to yield control between items,
        keeping the event loop responsive.
        """
        while self._running:
            try:
                # Wait for a file to appear in the queue (timeout = 1s)
                file_path = await asyncio.wait_for(
                    self._dirty_queue.get(),
                    timeout=1.0,
                )
                await self._reindex_file(file_path)
                self._dirty_queue.task_done()

                # Yield control to event loop between items
                await asyncio.sleep(0)

            except asyncio.TimeoutError:
                # No files in queue — loop back and wait
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"DirtyWatcher worker error: {e}", exc_info=True)

    async def _reindex_file(self, file_path: str) -> None:
        """
        Core dirty-bit re-indexing logic for a single file.

        Steps:
            1. Parse the file into new chunks
            2. Diff against cached chunks to find dirty set
            3. Re-embed dirty chunks (in ThreadPoolExecutor)
            4. Update call graph for dirty chunks
            5. Remove deleted chunks from embedding index
            6. Update file cache

        Args:
            file_path: Absolute path to the modified Python file.
        """
        t_start = time.perf_counter()
        file_name = Path(file_path).name

        # Step 1: Parse the modified file
        new_chunks = parse_file(file_path)

        # Step 2: Diff against cache
        old_chunks = self._file_chunks.get(file_path, [])
        dirty_chunks, deleted_ids = diff_chunks(old_chunks, new_chunks)

        if not dirty_chunks and not deleted_ids:
            logger.debug(f"No changes detected in {file_name}")
            return

        # Step 3: Re-embed dirty chunks
        # Run in thread pool — SentenceTransformer.encode() is blocking
        loop = asyncio.get_event_loop()
        for chunk in dirty_chunks:
            await loop.run_in_executor(
                None,  # Default executor
                self._embedder.update_chunk,
                chunk,
            )

        # Step 4: Update call graph for dirty chunks
        for chunk in dirty_chunks:
            self._call_graph.update_node(chunk)

        # Step 5: Remove deleted chunks
        for chunk_id in deleted_ids:
            self._embedder.remove_chunk(chunk_id)

        # Step 6: Update file cache
        self._file_chunks[file_path] = new_chunks

        elapsed = (time.perf_counter() - t_start) * 1000
        logger.info(
            f"Re-indexed {file_name}: "
            f"{len(dirty_chunks)} dirty, "
            f"{len(deleted_ids)} deleted | "
            f"{elapsed:.0f}ms"
        )

    def get_stats(self) -> dict:
        """Return watcher statistics for the /metrics endpoint."""
        return {
            "tracked_files": len(self._file_chunks),
            "total_chunks": sum(
                len(v) for v in self._file_chunks.values()
            ),
            "queue_size": self._dirty_queue.qsize(),
            "worker_running": self._running,
        }
