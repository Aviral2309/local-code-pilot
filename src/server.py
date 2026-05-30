"""
server.py
---------
AuraLSP Language Server — main entry point.

This is the core of the entire project. It implements the Language Server
Protocol (LSP) using the pygls library. The LSP is a JSON-RPC protocol
over stdin/stdout that any editor (VS Code, Neovim, Emacs) can connect to.

LSP Lifecycle:
    1. Editor spawns this process via the command in .vscode/settings.json
    2. Editor sends `initialize` request → we respond with our capabilities
    3. Editor sends `initialized` notification → we do our startup work
    4. As user types, editor sends `textDocument/didChange` notifications
    5. When user pauses, editor sends `textDocument/completion` requests
    6. We return CompletionItem list → editor shows ghost text
    7. On shutdown: editor sends `shutdown` + `exit`

Concurrency architecture:
    pygls runs on asyncio. All LSP handlers are async coroutines.
    PROBLEM: sentence-transformers and httpx are partially blocking.
    SOLUTION: Run blocking operations in a ThreadPoolExecutor via
    loop.run_in_executor(). This keeps the LSP event loop responsive
    — the editor never freezes waiting for an embedding or Ollama call.

Interview talking point: The asyncio + ThreadPoolExecutor pattern is
exactly how production async servers handle CPU-bound or I/O-blocking
work without blocking the event loop. Same pattern used in FastAPI,
aiohttp, and any serious async Python server.
"""

import asyncio
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

from lsprotocol import types
from pygls.server import LanguageServer

from src.ast_indexer import parse_workspace
from src.call_graph import CallGraph
from src.config import Config, load_config
from src.debounce import AsyncDebouncer
from src.dirty_watcher import DirtyWatcher
from src.embedder import EmbeddingEngine
from src.fim_builder import build_fim_prompt
from src.knapsack_allocator import ContextAllocator
from src.ollama_client import OllamaClient, OllamaConnectionError, OllamaTimeoutError
from src.telemetry import CompletionEvent, TelemetryLogger

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def setup_logging(config: Config) -> None:
    """Configure logging to file + stderr. LSP uses stdout for JSON-RPC."""
    level = getattr(logging, config.server.log_level.upper(), logging.INFO)

    handlers = [
        logging.StreamHandler(sys.stderr),  # NEVER sys.stdout — that's for JSON-RPC
    ]

    if config.server.log_file:
        handlers.append(logging.FileHandler(config.server.log_file, encoding="utf-8"))

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=handlers,
    )


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Document state: track the content of open files
# ---------------------------------------------------------------------------

class DocumentState:
    """
    In-memory store for currently open document contents.

    pygls provides its own workspace.get_text_document() but we maintain
    our own copy for direct access during completion without going through
    the pygls workspace abstraction layer.
    """

    def __init__(self) -> None:
        # uri -> full text content
        self._documents: dict[str, str] = {}

    def update(self, uri: str, text: str) -> None:
        self._documents[uri] = text

    def get(self, uri: str) -> Optional[str]:
        return self._documents.get(uri)

    def remove(self, uri: str) -> None:
        self._documents.pop(uri, None)

    def get_prefix_suffix(
        self,
        uri: str,
        position: types.Position,
    ) -> tuple[str, str]:
        """
        Split document at cursor position into prefix (above) and suffix (below).

        Args:
            uri:      Document URI.
            position: LSP Position (line + character, 0-indexed).

        Returns:
            (prefix, suffix) tuple. Both are empty strings if document unknown.
        """
        text = self._documents.get(uri, "")
        if not text:
            return "", ""

        lines = text.splitlines(keepends=True)
        line_idx = position.line
        char_idx = position.character

        # Validate bounds
        if line_idx >= len(lines):
            return text, ""

        # Everything before the cursor
        prefix_lines = lines[:line_idx]
        current_line_prefix = lines[line_idx][:char_idx]
        prefix = "".join(prefix_lines) + current_line_prefix

        # Everything after the cursor
        current_line_suffix = lines[line_idx][char_idx:]
        suffix_lines = lines[line_idx + 1:] if line_idx + 1 < len(lines) else []
        suffix = current_line_suffix + "".join(suffix_lines)

        return prefix, suffix


# ---------------------------------------------------------------------------
# The AuraLSP Server
# ---------------------------------------------------------------------------

class AuraLSPServer:
    """
    Wraps the pygls LanguageServer and wires all LSP handlers.

    Separation of concerns:
        - AuraLSPServer owns business logic and state
        - pygls LanguageServer handles JSON-RPC protocol mechanics
        - Handlers are registered on the pygls server but delegate here
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self.lsp = LanguageServer(
            name="auralsp",
            version="1.0.0",
        )
        self.docs = DocumentState()
        self.ollama = OllamaClient(config.ollama)
        self.telemetry = TelemetryLogger(config.metrics.db_path)
        self.debouncer = AsyncDebouncer(delay_ms=config.completion.debounce_ms)

        # ThreadPoolExecutor for blocking calls (embeddings, heavy I/O)
        # max_workers=2: one for embeddings, one for misc blocking ops
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="aura")

        # Phase 2: Semantic retrieval engine
        self._embedder = EmbeddingEngine(
            model_name=config.embedding.model_name
        )
        self._call_graph = CallGraph()
        self._dirty_watcher = DirtyWatcher(self._embedder, self._call_graph)
        self._allocator = ContextAllocator(
            embedder=self._embedder,
            call_graph=self._call_graph,
            token_cap=config.knapsack.token_cap,
            weight_semantic=config.knapsack.weight_semantic,
            weight_graph=config.knapsack.weight_graph_distance,
            weight_recency=config.knapsack.weight_recency,
            top_k_candidates=config.knapsack.top_k_candidates,
        )
        self._workspace_root: Optional[str] = None
        self._index_ready = False

        # Track the last completion for accept/reject telemetry
        self._last_completion_row_id: Optional[int] = None
        self._last_completion_text: Optional[str] = None

        # Register all LSP handlers
        self._register_handlers()
        logger.info("AuraLSPServer initialized")

    def _register_handlers(self) -> None:
        """Wire LSP lifecycle and document handlers to this server instance."""

        lsp = self.lsp

        # ------------------------------------------------------------------
        # Lifecycle: initialize
        # ------------------------------------------------------------------
        @lsp.feature(types.INITIALIZE)
        async def on_initialize(params: types.InitializeParams) -> None:
            """
            Editor sends this first. We declare our capabilities here.

            Key capability: completionProvider with triggerCharacters.
            This tells the editor "call me for completions when the user
            types any of these characters or explicitly requests completion."
            """
            logger.info(
                f"Initialize from '{params.client_info.name if params.client_info else 'unknown'}' "
                f"| workspace: {params.root_uri}"
            )

            # Store workspace root for indexing
            if params.root_uri:
                self._workspace_root = (
                    params.root_uri
                    .replace("file:///", "")
                    .replace("file://", "")
                )
                # Windows path fix
                if (len(self._workspace_root) > 2
                        and self._workspace_root[0] == "/"
                        and self._workspace_root[2] == ":"):
                    self._workspace_root = self._workspace_root[1:]

            # Run startup tasks in background — don't block initialize
            asyncio.create_task(self._startup_health_check())
            asyncio.create_task(self._startup_indexing())

        # ------------------------------------------------------------------
        # Lifecycle: initialized (server is ready)
        # ------------------------------------------------------------------
        @lsp.feature(types.INITIALIZED)
        async def on_initialized(params: types.InitializedParams) -> None:
            logger.info("LSP initialized — AuraLSP ready.")

        # ------------------------------------------------------------------
        # Document sync: opened
        # ------------------------------------------------------------------
        @lsp.feature(types.TEXT_DOCUMENT_DID_OPEN)
        async def on_open(params: types.DidOpenTextDocumentParams) -> None:
            """Track document content when user opens a file."""
            doc = params.text_document
            self.docs.update(doc.uri, doc.text)
            logger.debug(f"Opened: {doc.uri} ({doc.language_id})")

        # ------------------------------------------------------------------
        # Document sync: changed (every keystroke if incremental sync)
        # ------------------------------------------------------------------
        @lsp.feature(types.TEXT_DOCUMENT_DID_CHANGE)
        async def on_change(params: types.DidChangeTextDocumentParams) -> None:
            """
            Update our document mirror on every change.

            Note: We request full-document sync (TextDocumentSyncKind.Full)
            in our server capabilities. This means each change event contains
            the COMPLETE current text, not a delta. Simpler to implement
            correctly, slightly more bandwidth — fine for local IPC.
            """
            if not params.content_changes:
                return

            # With full sync, there's exactly one change containing full text
            full_text = params.content_changes[-1].text
            self.docs.update(params.text_document.uri, full_text)

        # ------------------------------------------------------------------
        # Document sync: saved
        # ------------------------------------------------------------------
        @lsp.feature(types.TEXT_DOCUMENT_DID_SAVE)
        async def on_save(params: types.DidSaveTextDocumentParams) -> None:
            """
            Triggered when user saves. Notifies dirty watcher to
            re-index only the changed functions in this file.
            """
            uri = params.text_document.uri
            file_path = uri.replace("file:///", "").replace("file://", "")
            if len(file_path) > 2 and file_path[0] == "/" and file_path[2] == ":":
                file_path = file_path[1:]

            logger.debug(f"Saved: {Path(file_path).name}")
            self._dirty_watcher.on_file_saved(file_path)

        # ------------------------------------------------------------------
        # Document sync: closed
        # ------------------------------------------------------------------
        @lsp.feature(types.TEXT_DOCUMENT_DID_CLOSE)
        async def on_close(params: types.DidCloseTextDocumentParams) -> None:
            self.docs.remove(params.text_document.uri)
            self.debouncer.cancel()
            logger.debug(f"Closed: {params.text_document.uri}")

        # ------------------------------------------------------------------
        # CORE: Completion request
        # ------------------------------------------------------------------
        @lsp.feature(
            types.TEXT_DOCUMENT_COMPLETION,
            types.CompletionOptions(
                trigger_characters=self.config.completion.trigger_characters,
                resolve_provider=False,
            ),
        )
        async def on_completion(
            params: types.CompletionParams,
        ) -> Optional[types.CompletionList]:
            """
            THE MAIN HANDLER: called when the editor wants a completion.

            Flow:
                1. Extract prefix and suffix from document at cursor position
                2. Build FIM prompt (Phase 1: no context injection yet)
                3. Stream completion from Ollama
                4. Return as CompletionItem with insertText
                5. Log telemetry

            Phase 2 will insert the knapsack-allocated context between
            steps 2 and 3.
            """
            uri = params.text_document.uri
            position = params.position

            # Get language ID from pygls workspace
            try:
                lang_id = self.lsp.workspace.get_text_document(uri).language_id
            except Exception:
                lang_id = "python"  # Safe fallback

            prefix, suffix = self.docs.get_prefix_suffix(uri, position)

            if not prefix.strip():
                # Don't fire on completely empty files
                return None

            t_request_start = time.perf_counter()

            # Phase 2: get workspace context via knapsack allocator
            injected_context = ""
            chunk_ids_used = []
            if self._index_ready:
                loop = asyncio.get_event_loop()
                injected_context, chunk_ids_used = await loop.run_in_executor(
                    self._executor,
                    self._allocator.get_context,
                    prefix,
                    uri,
                    None,  # current_func detection coming in Phase 3
                )

            payload = build_fim_prompt(
                prefix=prefix,
                suffix=suffix,
                language_id=lang_id,
                injected_context=injected_context,
                max_prefix_lines=self.config.completion.max_prefix_lines,
                max_suffix_lines=self.config.completion.max_suffix_lines,
            )

            logger.debug(
                f"Completion triggered | {uri.split('/')[-1]} "
                f"L{position.line}:{position.character}"
            )

            # Stream completion from Ollama
            completion_text = ""
            ttft_ms = 0.0
            first_token = True

            try:
                async for token in self.ollama.stream_completion(payload):
                    if first_token:
                        ttft_ms = (time.perf_counter() - t_request_start) * 1000
                        first_token = False
                    completion_text += token

            except OllamaTimeoutError as e:
                logger.warning(f"Ollama timeout: {e}")
                return None
            except OllamaConnectionError as e:
                logger.error(f"Ollama connection error: {e}")
                return None
            except Exception as e:
                logger.error(f"Unexpected error during completion: {e}", exc_info=True)
                return None

            if not completion_text.strip():
                return None

            total_ms = (time.perf_counter() - t_request_start) * 1000
            logger.info(
                f"Completion done | "
                f"ttft={ttft_ms:.0f}ms | "
                f"total={total_ms:.0f}ms | "
                f"len={len(completion_text)}"
            )

            # Log telemetry (non-blocking)
            event = CompletionEvent(
                file_path=uri,
                language_id=lang_id,
                context_used=chunk_ids_used,
                ttft_ms=ttft_ms,
                total_ms=total_ms,
                completion_text=completion_text,
                completion_length=len(completion_text),
            )

            try:
                row_id = self.telemetry.log_completion(event)
                self._last_completion_row_id = row_id
                self._last_completion_text = completion_text
            except Exception as e:
                logger.warning(f"Telemetry log failed: {e}")

            # Return as LSP CompletionList
            # insertText is what gets inserted when user accepts (Tab)
            completion_item = types.CompletionItem(
                label=completion_text[:40] + "..." if len(completion_text) > 40 else completion_text,
                insert_text=completion_text,
                kind=types.CompletionItemKind.Text,
                detail=f"AuraLSP [{lang_id}] | {total_ms:.0f}ms",
                documentation=types.MarkupContent(
                    kind=types.MarkupKind.Markdown,
                    value=(
                        f"**AuraLSP completion**\n\n"
                        f"- Model: `{self.config.ollama.model}`\n"
                        f"- TTFT: `{ttft_ms:.0f}ms`\n"
                        f"- Total: `{total_ms:.0f}ms`\n"
                        f"- Context: `{len(injected_context)} chars injected`"
                    ),
                ),
                insert_text_format=types.InsertTextFormat.PlainText,
            )

            return types.CompletionList(
                is_incomplete=False,
                items=[completion_item],
            )

        # ------------------------------------------------------------------
        # Shutdown
        # ------------------------------------------------------------------
        @lsp.feature(types.SHUTDOWN)
        async def on_shutdown(params: None) -> None:
            logger.info("Shutdown requested. Cleaning up.")
            await self._dirty_watcher.stop()
            await self.ollama.close()
            self._executor.shutdown(wait=False)

    async def _startup_health_check(self) -> None:
        """Background task: verify Ollama is running after initialization."""
        try:
            await self.ollama.health_check()
        except OllamaConnectionError as e:
            logger.error(f"Ollama not available: {e}")
            self.lsp.show_message(
                f"⚠ AuraLSP: Ollama not found. Run `ollama serve` then "
                f"`ollama pull {self.config.ollama.model}`",
                types.MessageType.Warning,
            )

    async def _startup_indexing(self) -> None:
        """
        Background task: load embedding model and index workspace.

        Runs after initialize — does NOT block the editor from opening.
        Completions work without context until indexing completes,
        then automatically switch to context-aware mode.
        """
        if not self._workspace_root:
            logger.info("No workspace root — skipping indexing")
            return

        loop = asyncio.get_event_loop()

        # Step 1: Load embedding model (blocking — run in executor)
        logger.info("Loading embedding model...")
        await loop.run_in_executor(self._executor, self._embedder.load_model)

        if not self._embedder._model:
            logger.warning("Embedding model failed to load — context injection disabled")
            return

        # Step 2: Parse workspace
        logger.info(f"Indexing workspace: {self._workspace_root}")
        all_chunks = await loop.run_in_executor(
            self._executor,
            parse_workspace,
            self._workspace_root,
        )

        if not all_chunks:
            logger.info("No Python files found in workspace")
            return

        # Step 3: Build embeddings
        await loop.run_in_executor(
            self._executor,
            self._embedder.index_chunks,
            all_chunks,
        )

        # Step 4: Build call graph
        await loop.run_in_executor(
            self._executor,
            self._call_graph.build_from_chunks,
            all_chunks,
        )

        # Step 5: Initialize dirty watcher
        self._dirty_watcher.initialize_from_chunks(all_chunks)
        await self._dirty_watcher.start()

        self._index_ready = True
        logger.info(
            f"Indexing complete: {self._embedder.index_size} chunks | "
            f"{self._call_graph.node_count} graph nodes | "
            f"Context injection ACTIVE"
        )
        self.lsp.show_message(
            f"AuraLSP: indexed {self._embedder.index_size} functions — context injection active",
            types.MessageType.Info,
        )

    def get_server_capabilities(self) -> types.ServerCapabilities:
        """
        Declare what this LSP server can do.
        The editor uses this to know when to call us.
        """
        return types.ServerCapabilities(
            # Full sync: send entire file on every change
            text_document_sync=types.TextDocumentSyncOptions(
                open_close=True,
                change=types.TextDocumentSyncKind.Full,
                save=types.SaveOptions(include_text=False),
            ),
            # We provide completions
            completion_provider=types.CompletionOptions(
                trigger_characters=self.config.completion.trigger_characters,
                resolve_provider=False,
            ),
        )

    def start(self) -> None:
        """Start the LSP server. Blocks until shutdown."""
        logger.info(
            f"Starting AuraLSP server | model={self.config.ollama.model} | "
            f"debounce={self.config.completion.debounce_ms}ms"
        )
        # stdio transport: JSON-RPC over stdin/stdout
        # This is the standard LSP transport for editor-spawned servers
        self.lsp.start_io()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Main entry point. Called by `auralsp` CLI command (see pyproject.toml).

    Usage:
        python -m src.server
        # OR after pip install -e .:
        auralsp
    """
    config = load_config()
    setup_logging(config)

    logger.info("=" * 60)
    logger.info("AuraLSP v1.0.0 — Local Workspace-Aware Code Intelligence")
    logger.info("=" * 60)

    server = AuraLSPServer(config)
    try:
        server.start()
    except KeyboardInterrupt:
        logger.info("Interrupted. Shutting down.")
    except Exception as e:
        logger.critical(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
