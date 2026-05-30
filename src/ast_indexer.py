"""
ast_indexer.py
--------------
Parses Python source files into function-level chunks using Tree-Sitter.

WHY TREE-SITTER INSTEAD OF PYTHON'S BUILT-IN ast MODULE:
    Python's ast module only parses valid Python. Tree-Sitter parses
    INCOMPLETE and INVALID code too — critical for an editor where the
    user is mid-way through writing a function. Tree-Sitter is also
    used by GitHub, Neovim, and Helix for exactly this reason.

WHY FUNCTION-LEVEL CHUNKING INSTEAD OF LINE-COUNT SPLITTING:
    Splitting by 20 lines is arbitrary. A function boundary is a
    semantic boundary — everything inside one function definition is
    a coherent unit of meaning. Injecting half a function into the
    model context is worse than injecting nothing.

WHAT A CHUNK IS:
    A CodeChunk represents one function_definition node from the AST.
    It contains the full source text of that function, its name,
    its file path, and its line range. The embedder will convert
    the source text into a 384-dim dense vector.

Interview talking point: This is compiler fundamentals applied to ML.
You are doing lexical analysis and AST construction — the first two
stages of a compiler pipeline — to produce structured input for a
retrieval system.
"""

import hashlib
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Tree-Sitter imports — wrapped in try/except for graceful degradation
# if tree-sitter isn't installed yet
try:
    import tree_sitter_python as tspython
    from tree_sitter import Language, Parser, Node
    # tree-sitter 0.21.x API: Language(path_or_ptr, name)
    # tree-sitter 0.22+ API: Language(ptr)
    try:
        PY_LANGUAGE = Language(tspython.language(), "python")
    except TypeError:
        PY_LANGUAGE = Language(tspython.language())
    TREE_SITTER_AVAILABLE = True
except ImportError:
    TREE_SITTER_AVAILABLE = False
    logger.warning(
        "tree-sitter not available. Install with: "
        "pip install tree-sitter tree-sitter-python"
    )


@dataclass
class CodeChunk:
    """
    A single function extracted from a source file.

    This is the atomic unit of the retrieval system.
    One CodeChunk = one function definition = one embedding vector.

    Attributes:
        chunk_id:     Unique identifier (SHA256 of file_path + func_name).
        file_path:    Absolute path to the source file.
        func_name:    Name of the function (e.g. "calculate_sharpe_ratio").
        source:       Full source text of the function body.
        start_line:   Line number where the function starts (0-indexed).
        end_line:     Line number where the function ends (0-indexed).
        is_dirty:     True if this chunk needs re-embedding.
        last_modified: Timestamp of last modification (for recency scoring).
        docstring:    First docstring of the function if present.
    """
    chunk_id: str
    file_path: str
    func_name: str
    source: str
    start_line: int
    end_line: int
    is_dirty: bool = True
    last_modified: float = field(default_factory=time.time)
    docstring: Optional[str] = None

    @property
    def display_name(self) -> str:
        """Human-readable identifier for logs and telemetry."""
        file_name = Path(self.file_path).name
        return f"{file_name}::{self.func_name}"

    @property
    def token_estimate(self) -> int:
        """Rough token count (~4 chars per token)."""
        return len(self.source) // 4


def make_chunk_id(file_path: str, func_name: str, start_line: int) -> str:
    """
    Generate a stable unique ID for a code chunk.

    Uses SHA256 of path + name + line to handle:
    - Same function name in different files
    - Renamed functions (old ID disappears, new ID appears)
    - Moved functions (line change = new ID)

    Args:
        file_path:  Absolute path to source file.
        func_name:  Function name.
        start_line: Start line number.

    Returns:
        8-character hex string (first 8 chars of SHA256).
    """
    key = f"{file_path}::{func_name}::{start_line}"
    return hashlib.sha256(key.encode()).hexdigest()[:8]


def _extract_docstring(node: "Node", source_bytes: bytes) -> Optional[str]:
    """
    Extract the docstring from a function_definition node if present.

    Tree-Sitter structure for a function with docstring:
        function_definition
            body: block
                expression_statement
                    string  ← this is the docstring

    Args:
        node:         Tree-Sitter function_definition node.
        source_bytes: Raw bytes of the source file.

    Returns:
        Docstring text, or None if not present.
    """
    try:
        body = next(
            (c for c in node.children if c.type == "block"), None
        )
        if not body:
            return None

        first_stmt = next(
            (c for c in body.children if c.type == "expression_statement"), None
        )
        if not first_stmt:
            return None

        string_node = next(
            (c for c in first_stmt.children if c.type == "string"), None
        )
        if not string_node:
            return None

        raw = source_bytes[string_node.start_byte:string_node.end_byte].decode(
            "utf-8", errors="replace"
        )
        # Strip quotes
        return raw.strip('"""').strip("'''").strip('"').strip("'").strip()

    except Exception:
        return None


def parse_file(file_path: str) -> list[CodeChunk]:
    """
    Parse a Python source file and extract all function-level chunks.

    Handles:
    - Top-level functions
    - Class methods (extracted as individual chunks)
    - Nested functions (extracted separately — each is its own chunk)
    - Files with syntax errors (Tree-Sitter is error-tolerant)
    - Empty files (returns empty list)

    Args:
        file_path: Absolute path to a Python source file.

    Returns:
        List of CodeChunk objects, one per function_definition node.
        Empty list if file can't be read or has no functions.
    """
    if not TREE_SITTER_AVAILABLE:
        logger.debug("Tree-Sitter not available, skipping parse")
        return []

    path = Path(file_path)
    if not path.exists() or not path.is_file():
        logger.warning(f"File not found: {file_path}")
        return []

    if path.suffix not in (".py",):
        return []

    try:
        source_bytes = path.read_bytes()
    except (PermissionError, OSError) as e:
        logger.warning(f"Cannot read {file_path}: {e}")
        return []

    if not source_bytes.strip():
        return []

    try:
        parser = Parser()
        parser.set_language(PY_LANGUAGE)
        tree = parser.parse(source_bytes)
    except Exception as e:
        logger.warning(f"Tree-Sitter parse error for {file_path}: {e}")
        return []

    chunks = []
    _walk_for_functions(
        node=tree.root_node,
        source_bytes=source_bytes,
        file_path=file_path,
        chunks=chunks,
    )

    logger.debug(f"Parsed {file_path}: found {len(chunks)} functions")
    return chunks


def _walk_for_functions(
    node: "Node",
    source_bytes: bytes,
    file_path: str,
    chunks: list,
) -> None:
    """
    Recursively walk the AST to find all function_definition nodes.

    We use depth-first traversal. Every function_definition we find
    becomes a CodeChunk — including nested functions and methods.

    Args:
        node:         Current AST node being visited.
        source_bytes: Raw source bytes (for text extraction).
        file_path:    Source file path.
        chunks:       Accumulator list — append chunks here.
    """
    if node.type == "function_definition":
        # Extract function name
        name_node = next(
            (c for c in node.children if c.type == "identifier"), None
        )
        func_name = (
            source_bytes[name_node.start_byte:name_node.end_byte].decode(
                "utf-8", errors="replace"
            )
            if name_node else "unknown"
        )

        # Extract full source text of this function
        source = source_bytes[node.start_byte:node.end_byte].decode(
            "utf-8", errors="replace"
        )

        # Line numbers (0-indexed from Tree-Sitter)
        start_line = node.start_point[0]
        end_line = node.end_point[0]

        # Extract docstring for richer embedding
        docstring = _extract_docstring(node, source_bytes)

        chunk_id = make_chunk_id(file_path, func_name, start_line)

        chunks.append(CodeChunk(
            chunk_id=chunk_id,
            file_path=file_path,
            func_name=func_name,
            source=source,
            start_line=start_line,
            end_line=end_line,
            is_dirty=True,
            last_modified=time.time(),
            docstring=docstring,
        ))

    # Recurse into all children
    for child in node.children:
        _walk_for_functions(child, source_bytes, file_path, chunks)


def parse_workspace(
    root_path: str,
    exclude_dirs: Optional[list[str]] = None,
) -> list[CodeChunk]:
    """
    Parse all Python files in a workspace directory.

    Skips:
    - Hidden directories (starting with .)
    - __pycache__ directories
    - venv / .venv directories
    - Any directories in exclude_dirs

    Args:
        root_path:    Root directory of the workspace.
        exclude_dirs: Additional directory names to skip.

    Returns:
        All CodeChunks from all Python files in the workspace.
    """
    exclude = {
        "__pycache__", "venv", ".venv", "env", ".env",
        "node_modules", ".git", "dist", "build", ".mypy_cache",
        ".pytest_cache", "htmlcov",
    }
    if exclude_dirs:
        exclude.update(exclude_dirs)

    root = Path(root_path)
    if not root.exists():
        logger.warning(f"Workspace root not found: {root_path}")
        return []

    all_chunks = []
    py_files = [
        p for p in root.rglob("*.py")
        if not any(part in exclude for part in p.parts)
        and not any(part.startswith(".") for part in p.parts[len(root.parts):])
    ]

    logger.info(f"Indexing {len(py_files)} Python files in {root_path}")
    t_start = time.perf_counter()

    for py_file in py_files:
        chunks = parse_file(str(py_file))
        all_chunks.extend(chunks)

    elapsed = (time.perf_counter() - t_start) * 1000
    logger.info(
        f"Workspace indexed: {len(all_chunks)} functions "
        f"from {len(py_files)} files in {elapsed:.0f}ms"
    )

    return all_chunks


def diff_chunks(
    old_chunks: list[CodeChunk],
    new_chunks: list[CodeChunk],
) -> tuple[list[CodeChunk], list[str]]:
    """
    Compare old and new chunk lists to find what changed.

    This is the core of the dirty-bit system. Instead of re-embedding
    everything when a file changes, we find exactly which functions
    changed and mark only those as dirty.

    Algorithm:
        Build a dict of {chunk_id: source} for old chunks.
        For each new chunk:
            - If chunk_id not in old → new function (dirty)
            - If source changed → modified function (dirty)
            - If source unchanged → clean (not dirty)
        Collect chunk_ids that no longer exist → deleted

    Args:
        old_chunks: Previously indexed chunks for a file.
        new_chunks: Freshly parsed chunks for the same file.

    Returns:
        (dirty_chunks, deleted_ids):
            dirty_chunks: Chunks that need re-embedding.
            deleted_ids:  chunk_ids that were removed.
    """
    old_map = {c.chunk_id: c.source for c in old_chunks}
    new_map = {c.chunk_id: c for c in new_chunks}

    dirty = []
    for chunk_id, chunk in new_map.items():
        if chunk_id not in old_map:
            # New function — never been embedded
            chunk.is_dirty = True
            dirty.append(chunk)
        elif old_map[chunk_id] != chunk.source:
            # Function body changed — needs re-embedding
            chunk.is_dirty = True
            chunk.last_modified = time.time()
            dirty.append(chunk)
        else:
            # Unchanged — keep existing embedding
            chunk.is_dirty = False

    deleted_ids = [cid for cid in old_map if cid not in new_map]

    if dirty or deleted_ids:
        logger.debug(
            f"Diff: {len(dirty)} dirty, {len(deleted_ids)} deleted"
        )

    return dirty, deleted_ids
