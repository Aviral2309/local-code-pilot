"""
tests/test_fim_builder.py
-------------------------
Unit tests for the FIM prompt builder.

These tests verify:
1. Correct FIM token structure (exact token order matters for the model)
2. Prefix/suffix trimming logic
3. Context injection format
4. Stop token cleanup
5. Edge cases (empty input, very long files, no-newline files)

Run with: pytest tests/test_fim_builder.py -v
"""

import pytest
from src.fim_builder import (
    build_fim_prompt,
    extract_clean_completion,
    trim_prefix,
    trim_suffix,
    FIM_PREFIX_TOKEN,
    FIM_SUFFIX_TOKEN,
    FIM_MIDDLE_TOKEN,
    END_OF_TEXT_TOKEN,
)


# ---------------------------------------------------------------------------
# trim_prefix tests
# ---------------------------------------------------------------------------

class TestTrimPrefix:

    def test_short_prefix_unchanged(self):
        """Prefix with fewer lines than max_lines is returned as-is."""
        prefix = "line1\nline2\nline3"
        result = trim_prefix(prefix, max_lines=10)
        assert result == prefix

    def test_long_prefix_trimmed_from_top(self):
        """Only the LAST N lines are kept — most recent code is most relevant."""
        lines = [f"line{i}" for i in range(100)]
        prefix = "\n".join(lines)
        result = trim_prefix(prefix, max_lines=10)
        result_lines = result.split("\n")
        assert len(result_lines) == 10
        # Should keep the LAST 10 lines (line90..line99)
        assert result_lines[0] == "line90"
        assert result_lines[-1] == "line99"

    def test_exact_limit_unchanged(self):
        """Prefix exactly at the limit is unchanged."""
        lines = [f"line{i}" for i in range(50)]
        prefix = "\n".join(lines)
        result = trim_prefix(prefix, max_lines=50)
        assert result == prefix

    def test_empty_prefix(self):
        """Empty prefix stays empty."""
        assert trim_prefix("", max_lines=50) == ""

    def test_single_line_prefix(self):
        """Single-line prefix is never trimmed."""
        prefix = "def hello_world():"
        assert trim_prefix(prefix, max_lines=1) == prefix
        assert trim_prefix(prefix, max_lines=50) == prefix


# ---------------------------------------------------------------------------
# trim_suffix tests
# ---------------------------------------------------------------------------

class TestTrimSuffix:

    def test_short_suffix_unchanged(self):
        suffix = "    return x\n\ndef other():\n    pass"
        result = trim_suffix(suffix, max_lines=10)
        assert result == suffix

    def test_long_suffix_trimmed_from_bottom(self):
        """Only the FIRST N lines are kept — immediate context is most useful."""
        lines = [f"line{i}" for i in range(100)]
        suffix = "\n".join(lines)
        result = trim_suffix(suffix, max_lines=10)
        result_lines = result.split("\n")
        assert len(result_lines) == 10
        assert result_lines[0] == "line0"
        assert result_lines[-1] == "line9"

    def test_empty_suffix(self):
        assert trim_suffix("", max_lines=20) == ""


# ---------------------------------------------------------------------------
# build_fim_prompt tests
# ---------------------------------------------------------------------------

class TestBuildFIMPrompt:

    def test_correct_token_order(self):
        """
        The FIM token order is CRITICAL for Qwen2.5-Coder to work correctly.
        Wrong order = garbage output. Verify it every time.

        Correct: <|fim_prefix|>PREFIX<|fim_suffix|>SUFFIX<|fim_middle|>
        """
        payload = build_fim_prompt(
            prefix="def hello():\n    ",
            suffix="\n\nhello()",
            language_id="python",
        )
        prompt = payload.prompt

        prefix_pos = prompt.index(FIM_PREFIX_TOKEN)
        suffix_pos = prompt.index(FIM_SUFFIX_TOKEN)
        middle_pos = prompt.index(FIM_MIDDLE_TOKEN)

        assert prefix_pos < suffix_pos < middle_pos, (
            f"Token order wrong! prefix={prefix_pos}, "
            f"suffix={suffix_pos}, middle={middle_pos}"
        )

    def test_prefix_in_prompt(self):
        """The actual prefix content appears in the prompt."""
        prefix = "def calculate_returns(portfolio):\n    "
        payload = build_fim_prompt(prefix=prefix, suffix="")
        assert "calculate_returns" in payload.prompt

    def test_suffix_in_prompt(self):
        """The suffix content appears after the suffix token."""
        suffix = "\n    return result"
        payload = build_fim_prompt(prefix="def foo():\n    x = 1", suffix=suffix)
        suffix_section = payload.prompt.split(FIM_SUFFIX_TOKEN)[1]
        assert "return result" in suffix_section

    def test_ends_with_fim_middle(self):
        """The prompt must end with <|fim_middle|> — nothing after it."""
        payload = build_fim_prompt(prefix="x = ", suffix=" + 1")
        assert payload.prompt.endswith(FIM_MIDDLE_TOKEN)

    def test_context_injection_appears_before_prefix(self):
        """
        Injected context appears BEFORE the user's prefix in the effective prefix.
        The model reads top-to-bottom, so context definitions come first.
        """
        context = "def helper(x): return x * 2"
        payload = build_fim_prompt(
            prefix="result = helper(",
            suffix=")",
            injected_context=context,
        )
        # Context should appear before the user's code in the prompt
        context_pos = payload.prompt.index("helper(x)")
        user_code_pos = payload.prompt.index("result = helper(")
        assert context_pos < user_code_pos

    def test_context_injection_has_comment_markers(self):
        """Context block is wrapped in comments so the model understands it."""
        context = "def foo(): pass"
        payload = build_fim_prompt(prefix="foo(", suffix=")", injected_context=context)
        assert "WORKSPACE CONTEXT" in payload.prompt

    def test_no_context_no_comment_block(self):
        """With no injected context, no WORKSPACE CONTEXT comment appears."""
        payload = build_fim_prompt(prefix="x = 1\n", suffix="", injected_context="")
        assert "WORKSPACE CONTEXT" not in payload.prompt

    def test_empty_prefix_handled(self):
        """Empty prefix doesn't cause an exception."""
        payload = build_fim_prompt(prefix="", suffix="return 42")
        assert FIM_PREFIX_TOKEN in payload.prompt

    def test_empty_suffix_handled(self):
        """Empty suffix doesn't cause an exception."""
        payload = build_fim_prompt(prefix="def foo():\n    ", suffix="")
        assert FIM_SUFFIX_TOKEN in payload.prompt

    def test_payload_metadata(self):
        """FIMPayload contains correct metadata fields."""
        payload = build_fim_prompt(
            prefix="x = 1",
            suffix="y = 2",
            language_id="python",
            injected_context="# context",
        )
        assert payload.language_id == "python"
        assert payload.prefix == "x = 1"
        assert payload.suffix == "y = 2"
        assert payload.token_estimate > 0

    def test_token_estimate_is_positive(self):
        """Token estimate should always be > 0 for non-empty prompts."""
        payload = build_fim_prompt(prefix="import os\n", suffix="")
        assert payload.token_estimate > 0

    def test_prefix_trimming_applied(self):
        """Very long prefix is trimmed to max_prefix_lines."""
        long_prefix = "\n".join([f"# line {i}" for i in range(200)])
        payload = build_fim_prompt(
            prefix=long_prefix,
            suffix="",
            max_prefix_lines=10,
        )
        # The effective prefix inside the prompt should have ~10 lines
        # (plus possible context block lines)
        fim_prefix_content = payload.prompt.split(FIM_PREFIX_TOKEN)[1].split(FIM_SUFFIX_TOKEN)[0]
        # Count non-empty lines that are actual code (not context comments)
        code_lines = [
            l for l in fim_prefix_content.split("\n")
            if l.strip() and "CONTEXT" not in l and "END" not in l
        ]
        assert len(code_lines) <= 15  # Some tolerance for multi-line context header

    def test_real_python_completion_scenario(self):
        """Simulate a realistic Python completion scenario."""
        prefix = """import numpy as np
from typing import List

def calculate_sharpe_ratio(returns: List[float], risk_free_rate: float = 0.02) -> float:
    \"\"\"Calculate the Sharpe ratio for a portfolio.\"\"\"
    returns_array = np.array(returns)
    excess_returns = returns_array - risk_free_rate / 252
    """
        suffix = """
    return sharpe_ratio

def annualize_returns(daily_returns: List[float]) -> float:
    return np.mean(daily_returns) * 252
"""
        payload = build_fim_prompt(prefix=prefix, suffix=suffix, language_id="python")

        assert FIM_PREFIX_TOKEN in payload.prompt
        assert "calculate_sharpe_ratio" in payload.prompt
        assert FIM_SUFFIX_TOKEN in payload.prompt
        assert "annualize_returns" in payload.prompt
        assert payload.prompt.endswith(FIM_MIDDLE_TOKEN)


# ---------------------------------------------------------------------------
# extract_clean_completion tests
# ---------------------------------------------------------------------------

class TestExtractCleanCompletion:

    def test_strips_end_of_text_token(self):
        raw = f"    return x * 2{END_OF_TEXT_TOKEN}"
        assert extract_clean_completion(raw) == "    return x * 2"

    def test_strips_fim_prefix_token(self):
        raw = f"    result = 42{FIM_PREFIX_TOKEN}some garbage"
        result = extract_clean_completion(raw)
        assert FIM_PREFIX_TOKEN not in result
        assert "result = 42" in result

    def test_strips_fim_suffix_token(self):
        raw = f"    return value{FIM_SUFFIX_TOKEN}"
        result = extract_clean_completion(raw)
        assert FIM_SUFFIX_TOKEN not in result

    def test_clean_completion_unchanged(self):
        """A completion with no stop tokens is returned as-is (minus trailing space)."""
        clean = "    mean = np.mean(data)\n    std = np.std(data)"
        assert extract_clean_completion(clean) == clean

    def test_trailing_whitespace_stripped(self):
        raw = "    return x    \n\n\n"
        result = extract_clean_completion(raw)
        assert not result.endswith(" ")
        assert not result.endswith("\n")

    def test_empty_input(self):
        assert extract_clean_completion("") == ""

    def test_only_stop_token(self):
        """A response that is only a stop token returns empty string."""
        result = extract_clean_completion(END_OF_TEXT_TOKEN)
        assert result == ""

    def test_multiple_stop_tokens_first_wins(self):
        """Splits at the FIRST occurrence of any stop token."""
        raw = f"valid_code(){END_OF_TEXT_TOKEN}more_code{FIM_PREFIX_TOKEN}garbage"
        result = extract_clean_completion(raw)
        assert "valid_code()" in result
        assert "more_code" not in result
