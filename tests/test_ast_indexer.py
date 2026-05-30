"""
tests/test_ast_indexer.py
-------------------------
Tests for Tree-Sitter AST parsing and function-level chunking.

These tests use actual Python source strings — no mock files needed.
Tree-Sitter is error-tolerant so even malformed code is testable.
"""

import tempfile
import os
import pytest
from pathlib import Path

from src.ast_indexer import (
    parse_file,
    parse_workspace,
    diff_chunks,
    make_chunk_id,
    CodeChunk,
    TREE_SITTER_AVAILABLE,
)

# Skip all tests if Tree-Sitter isn't installed
pytestmark = pytest.mark.skipif(
    not TREE_SITTER_AVAILABLE,
    reason="tree-sitter not installed"
)


def write_temp_py(content: str, suffix: str = ".py") -> str:
    """Write content to a temp file and return its path."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=suffix, delete=False, encoding="utf-8"
    ) as f:
        f.write(content)
        return f.name


class TestParseFile:

    def test_single_function(self):
        """A file with one function produces one chunk."""
        source = """
def hello_world():
    return "hello"
"""
        path = write_temp_py(source)
        try:
            chunks = parse_file(path)
            assert len(chunks) == 1
            assert chunks[0].func_name == "hello_world"
        finally:
            os.unlink(path)

    def test_multiple_functions(self):
        """A file with three functions produces three chunks."""
        source = """
def func_a():
    pass

def func_b():
    return 1

def func_c(x, y):
    return x + y
"""
        path = write_temp_py(source)
        try:
            chunks = parse_file(path)
            assert len(chunks) == 3
            names = {c.func_name for c in chunks}
            assert names == {"func_a", "func_b", "func_c"}
        finally:
            os.unlink(path)

    def test_class_methods_extracted(self):
        """Methods inside a class are extracted as individual chunks."""
        source = """
class MyClass:
    def method_a(self):
        return 1

    def method_b(self, x):
        return x * 2
"""
        path = write_temp_py(source)
        try:
            chunks = parse_file(path)
            names = {c.func_name for c in chunks}
            assert "method_a" in names
            assert "method_b" in names
        finally:
            os.unlink(path)

    def test_chunk_contains_full_source(self):
        """Chunk source contains the complete function text."""
        source = """def add(a, b):
    result = a + b
    return result
"""
        path = write_temp_py(source)
        try:
            chunks = parse_file(path)
            assert len(chunks) == 1
            assert "result = a + b" in chunks[0].source
            assert "return result" in chunks[0].source
        finally:
            os.unlink(path)

    def test_chunk_has_correct_line_numbers(self):
        """Chunk start_line matches actual position in file."""
        source = """# comment
# another comment

def my_function():
    pass
"""
        path = write_temp_py(source)
        try:
            chunks = parse_file(path)
            assert len(chunks) == 1
            assert chunks[0].start_line == 3  # 0-indexed, line 4
        finally:
            os.unlink(path)

    def test_chunk_id_is_unique(self):
        """Each chunk gets a unique chunk_id."""
        source = """
def func_a():
    pass

def func_b():
    pass
"""
        path = write_temp_py(source)
        try:
            chunks = parse_file(path)
            ids = [c.chunk_id for c in chunks]
            assert len(ids) == len(set(ids)), "Chunk IDs must be unique"
        finally:
            os.unlink(path)

    def test_docstring_extracted(self):
        """Docstrings are captured in the chunk."""
        source = '''
def calculate_mean(data):
    """Calculate the arithmetic mean of a list."""
    return sum(data) / len(data)
'''
        path = write_temp_py(source)
        try:
            chunks = parse_file(path)
            assert len(chunks) == 1
            assert chunks[0].docstring is not None
            assert "arithmetic mean" in chunks[0].docstring
        finally:
            os.unlink(path)

    def test_empty_file_returns_empty(self):
        """Empty file returns no chunks."""
        path = write_temp_py("")
        try:
            chunks = parse_file(path)
            assert chunks == []
        finally:
            os.unlink(path)

    def test_nonexistent_file_returns_empty(self):
        """Non-existent file path returns empty list without crashing."""
        chunks = parse_file("/nonexistent/path/to/file.py")
        assert chunks == []

    def test_non_python_file_returns_empty(self):
        """Non-.py files are skipped."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".js", delete=False
        ) as f:
            f.write("function hello() { return 1; }")
            path = f.name
        try:
            chunks = parse_file(path)
            assert chunks == []
        finally:
            os.unlink(path)

    def test_file_path_stored_in_chunk(self):
        """Each chunk stores the absolute file path."""
        source = "def foo(): pass\n"
        path = write_temp_py(source)
        try:
            chunks = parse_file(path)
            assert chunks[0].file_path == path
        finally:
            os.unlink(path)

    def test_token_estimate_positive(self):
        """Token estimate is positive for non-empty functions."""
        source = """
def long_function(data, window=20):
    result = []
    for i in range(len(data)):
        window_data = data[max(0, i-window):i+1]
        result.append(sum(window_data) / len(window_data))
    return result
"""
        path = write_temp_py(source)
        try:
            chunks = parse_file(path)
            assert chunks[0].token_estimate > 0
        finally:
            os.unlink(path)


class TestMakeChunkId:

    def test_same_inputs_same_id(self):
        """Same inputs always produce same ID (deterministic)."""
        id1 = make_chunk_id("/path/file.py", "my_func", 10)
        id2 = make_chunk_id("/path/file.py", "my_func", 10)
        assert id1 == id2

    def test_different_file_different_id(self):
        """Same function name in different files gets different IDs."""
        id1 = make_chunk_id("/path/file_a.py", "my_func", 10)
        id2 = make_chunk_id("/path/file_b.py", "my_func", 10)
        assert id1 != id2

    def test_different_line_different_id(self):
        """Same name at different line gets different ID."""
        id1 = make_chunk_id("/path/file.py", "my_func", 10)
        id2 = make_chunk_id("/path/file.py", "my_func", 20)
        assert id1 != id2

    def test_id_is_8_chars(self):
        """Chunk ID is exactly 8 hex characters."""
        chunk_id = make_chunk_id("/path/file.py", "func", 0)
        assert len(chunk_id) == 8
        assert all(c in "0123456789abcdef" for c in chunk_id)


class TestDiffChunks:

    def _make_chunk(self, name: str, source: str, line: int = 0) -> CodeChunk:
        return CodeChunk(
            chunk_id=make_chunk_id("/test/file.py", name, line),
            file_path="/test/file.py",
            func_name=name,
            source=source,
            start_line=line,
            end_line=line + 3,
        )

    def test_no_changes_returns_empty(self):
        """Identical old and new chunks produce no dirty chunks."""
        chunk = self._make_chunk("func_a", "def func_a():\n    pass", 0)
        dirty, deleted = diff_chunks([chunk], [chunk])
        assert dirty == []
        assert deleted == []

    def test_new_function_is_dirty(self):
        """A function present in new but not old is marked dirty."""
        old = [self._make_chunk("func_a", "def func_a(): pass", 0)]
        new_chunk = self._make_chunk("func_b", "def func_b(): return 1", 5)
        new = [old[0], new_chunk]

        dirty, deleted = diff_chunks(old, new)
        dirty_names = {c.func_name for c in dirty}
        assert "func_b" in dirty_names

    def test_deleted_function_returned(self):
        """A function removed from the file appears in deleted_ids."""
        chunk_a = self._make_chunk("func_a", "def func_a(): pass", 0)
        chunk_b = self._make_chunk("func_b", "def func_b(): pass", 5)

        dirty, deleted = diff_chunks([chunk_a, chunk_b], [chunk_a])
        assert chunk_b.chunk_id in deleted

    def test_modified_function_is_dirty(self):
        """A function whose source changed is marked dirty."""
        old_chunk = self._make_chunk("func_a", "def func_a():\n    return 1", 0)
        new_chunk = self._make_chunk("func_a", "def func_a():\n    return 2", 0)

        dirty, deleted = diff_chunks([old_chunk], [new_chunk])
        assert len(dirty) == 1
        assert dirty[0].func_name == "func_a"

    def test_unchanged_function_not_dirty(self):
        """An unchanged function is not in the dirty list."""
        chunk = self._make_chunk("func_a", "def func_a():\n    return 1", 0)
        dirty, deleted = diff_chunks([chunk], [chunk])
        assert len(dirty) == 0

    def test_empty_old_all_new_dirty(self):
        """When old is empty, all new chunks are dirty."""
        new = [
            self._make_chunk("func_a", "def func_a(): pass", 0),
            self._make_chunk("func_b", "def func_b(): pass", 5),
        ]
        dirty, deleted = diff_chunks([], new)
        assert len(dirty) == 2
        assert deleted == []


class TestParseWorkspace:

    def test_finds_python_files(self, tmp_path):
        """parse_workspace finds all .py files recursively."""
        (tmp_path / "main.py").write_text("def main(): pass\n")
        (tmp_path / "sub").mkdir()
        (tmp_path / "sub" / "utils.py").write_text("def helper(): pass\n")

        chunks = parse_workspace(str(tmp_path))
        names = {c.func_name for c in chunks}
        assert "main" in names
        assert "helper" in names

    def test_skips_venv(self, tmp_path):
        """venv directory is excluded from indexing."""
        (tmp_path / "main.py").write_text("def real_func(): pass\n")
        venv = tmp_path / "venv"
        venv.mkdir()
        (venv / "lib.py").write_text("def venv_func(): pass\n")

        chunks = parse_workspace(str(tmp_path))
        names = {c.func_name for c in chunks}
        assert "real_func" in names
        assert "venv_func" not in names

    def test_nonexistent_root_returns_empty(self):
        """Non-existent workspace root returns empty list."""
        chunks = parse_workspace("/nonexistent/workspace")
        assert chunks == []
