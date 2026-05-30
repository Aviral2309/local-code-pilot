"""
tests/test_eval_metrics.py
--------------------------
Unit tests for evaluation metric functions.

These test the metrics themselves — not the model output.
We want to verify that BLEU, syntax check, and exact match
behave correctly before trusting the eval harness numbers.
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from eval.eval_harness import (
    compute_bleu,
    check_syntax_valid,
    check_exact_match,
    mask_solution,
)


class TestComputeBLEU:

    def test_identical_strings_score_one(self):
        """Identical reference and hypothesis = perfect score."""
        ref = "return sum(x for x in data)"
        assert compute_bleu(ref, ref) == 1.0

    def test_empty_hypothesis_score_zero(self):
        """Empty completion scores zero."""
        assert compute_bleu("return x + 1", "") == 0.0

    def test_empty_reference_empty_hyp_score_one(self):
        """Both empty = trivially perfect."""
        assert compute_bleu("", "") == 1.0

    def test_partial_overlap_between_zero_and_one(self):
        """Partial token match gives intermediate score."""
        ref = "return sum(numbers)"
        hyp = "return sum(data)"
        score = compute_bleu(ref, hyp)
        assert 0.0 < score < 1.0

    def test_completely_different_score_low(self):
        """Completely different tokens give low score."""
        ref = "return True"
        hyp = "x = 42"
        score = compute_bleu(ref, hyp)
        assert score < 0.3

    def test_score_is_between_zero_and_one(self):
        """Score is always in valid range."""
        pairs = [
            ("return x", "return y"),
            ("a b c d", "a b"),
            ("hello world", "world hello"),
        ]
        for ref, hyp in pairs:
            score = compute_bleu(ref, hyp)
            assert 0.0 <= score <= 1.0, f"Score {score} out of range for {ref!r} vs {hyp!r}"


class TestCheckSyntaxValid:

    def test_valid_complete_function(self):
        """A complete valid function passes."""
        prefix = "def add(a, b):\n"
        completion = "    return a + b\n"
        suffix = ""
        assert check_syntax_valid(prefix, completion, suffix) is True

    def test_invalid_syntax_fails(self):
        """Broken syntax returns False."""
        prefix = "def foo():\n"
        completion = "    return (\n"  # unclosed paren
        suffix = ""
        assert check_syntax_valid(prefix, completion, suffix) is False

    def test_empty_completion_with_valid_frame(self):
        """Empty completion with valid surrounding code."""
        prefix = "def foo():\n    pass\n"
        completion = ""
        suffix = ""
        assert check_syntax_valid(prefix, completion, suffix) is True

    def test_completion_fixes_incomplete_prefix(self):
        """Completion that completes an incomplete function."""
        prefix = "def multiply(a, b):\n"
        completion = "    return a * b\n"
        suffix = "\nresult = multiply(3, 4)\n"
        assert check_syntax_valid(prefix, completion, suffix) is True

    def test_realistic_humaneval_scenario(self):
        """Realistic HumanEval-style completion."""
        prefix = "def sum_list(numbers):\n    total = 0\n    for n in numbers:\n"
        completion = "        total += n\n    return total\n"
        suffix = ""
        assert check_syntax_valid(prefix, completion, suffix) is True


class TestCheckExactMatch:

    def test_identical_match(self):
        """Identical strings match."""
        assert check_exact_match("    return x\n", "    return x\n") is True

    def test_whitespace_stripped_match(self):
        """Whitespace differences are ignored."""
        assert check_exact_match("    return x\n", "    return x") is True

    def test_different_content_no_match(self):
        """Different content does not match."""
        assert check_exact_match("return x", "return y") is False

    def test_empty_both_match(self):
        """Both empty strings match."""
        assert check_exact_match("", "") is True

    def test_one_empty_no_match(self):
        """One empty, one not = no match."""
        assert check_exact_match("return x", "") is False


class TestMaskSolution:

    def test_returns_three_parts(self):
        """mask_solution returns exactly three strings."""
        prompt = "def foo():\n    pass\n"
        canonical = "    return 42\n"
        result = mask_solution(prompt, canonical)
        assert len(result) == 3

    def test_prefix_contains_prompt(self):
        """Prefix always starts with the prompt."""
        prompt = "def foo():\n"
        canonical = "    x = 1\n    return x\n"
        prefix, target, suffix = mask_solution(prompt, canonical)
        assert prompt in prefix

    def test_target_is_nonempty_for_multi_line(self):
        """Multi-line solutions have a nonempty target to complete."""
        prompt = "def foo():\n"
        canonical = "    x = 1\n    y = 2\n    return x + y\n"
        prefix, target, suffix = mask_solution(prompt, canonical)
        assert len(target) > 0

    def test_prefix_plus_target_equals_full_solution(self):
        """prefix_addition + target reconstructs the canonical solution."""
        prompt = "def foo():\n"
        canonical = "    line1\n    line2\n    line3\n"
        prefix, target, suffix = mask_solution(prompt, canonical)

        # Extract what we added to prefix (everything after prompt)
        prefix_addition = prefix[len(prompt):]
        reconstructed = prefix_addition + target
        assert reconstructed == canonical
