"""
fim_builder.py
--------------
Constructs Fill-in-the-Middle (FIM) prompts for Qwen2.5-Coder.

FIM is a training objective where the model learns to predict a missing
middle segment given the code before it (prefix) and after it (suffix).
This is fundamentally better than next-token completion because the model
can see BOTH sides of what it's generating.

Qwen2.5-Coder FIM token format (from official documentation):
    <|fim_prefix|>{prefix}<|fim_suffix|>{suffix}<|fim_middle|>

The model then generates the middle segment and stops at <|endoftext|>.

Interview talking point: Most student projects only use "complete after
cursor." FIM requires capturing suffix context — a non-trivial change to
the LSP textDocument/completion handler — and produces measurably better
completions for partial expressions and multi-line constructs.
"""

import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

# FIM special tokens for Qwen2.5-Coder
# These are baked into the model's tokenizer and training data.
FIM_PREFIX_TOKEN = "<|fim_prefix|>"
FIM_SUFFIX_TOKEN = "<|fim_suffix|>"
FIM_MIDDLE_TOKEN = "<|fim_middle|>"
END_OF_TEXT_TOKEN = "<|endoftext|>"

# Stop sequences: tell Ollama to halt generation at these tokens
FIM_STOP_SEQUENCES = [END_OF_TEXT_TOKEN, FIM_PREFIX_TOKEN, FIM_SUFFIX_TOKEN]


@dataclass
class FIMPayload:
    """
    The complete structured input to the local model.

    Attributes:
        prompt:           The raw FIM-formatted string sent to Ollama.
        prefix:           Code above the cursor (original, untrimmed).
        suffix:           Code below the cursor (original, untrimmed).
        injected_context: Code snippets injected from workspace retrieval.
        language_id:      VS Code language identifier (e.g. "python").
        token_estimate:   Rough token count of the full prompt.
    """
    prompt: str
    prefix: str
    suffix: str
    injected_context: str
    language_id: str
    token_estimate: int


def trim_prefix(prefix: str, max_lines: int) -> str:
    """
    Keep only the last `max_lines` lines of the prefix.
    The most recent code is most relevant — distant lines add noise.

    Args:
        prefix:    Raw prefix string (everything above cursor).
        max_lines: Maximum number of lines to retain.

    Returns:
        Trimmed prefix string.
    """
    lines = prefix.splitlines()
    if len(lines) <= max_lines:
        return prefix
    trimmed = lines[-max_lines:]
    return "\n".join(trimmed)


def trim_suffix(suffix: str, max_lines: int) -> str:
    """
    Keep only the first `max_lines` lines of the suffix.
    The immediate context after the cursor is most useful.

    Args:
        suffix:    Raw suffix string (everything below cursor).
        max_lines: Maximum number of lines to retain.

    Returns:
        Trimmed suffix string.
    """
    lines = suffix.splitlines()
    if len(lines) <= max_lines:
        return suffix
    trimmed = lines[:max_lines]
    return "\n".join(trimmed)


def build_fim_prompt(
    prefix: str,
    suffix: str,
    language_id: str = "python",
    injected_context: str = "",
    max_prefix_lines: int = 50,
    max_suffix_lines: int = 20,
) -> FIMPayload:
    """
    Build a complete FIM prompt ready for Ollama inference.

    Context injection strategy:
        Injected workspace context is prepended to the prefix, separated
        by a clear comment block. This tells the model about relevant
        functions/classes defined elsewhere in the project.

        Structure:
            [WORKSPACE CONTEXT]
            {injected_context}
            [END CONTEXT]

            {trimmed_prefix}

        The suffix is placed after FIM_SUFFIX_TOKEN unchanged — the model
        uses it to understand what comes after the generation point.

    Args:
        prefix:            Text above the cursor in the current file.
        suffix:            Text below the cursor in the current file.
        language_id:       Language identifier from VS Code.
        injected_context:  Relevant code snippets from other files.
        max_prefix_lines:  Hard limit on prefix lines to avoid token overflow.
        max_suffix_lines:  Hard limit on suffix lines.

    Returns:
        FIMPayload with the formatted prompt and metadata.
    """
    trimmed_prefix = trim_prefix(prefix, max_prefix_lines)
    trimmed_suffix = trim_suffix(suffix, max_suffix_lines)

    # Build the effective prefix: context block + current file prefix
    if injected_context.strip():
        context_block = (
            f"# [WORKSPACE CONTEXT — relevant definitions from this project]\n"
            f"{injected_context}\n"
            f"# [END CONTEXT]\n\n"
        )
        effective_prefix = context_block + trimmed_prefix
    else:
        effective_prefix = trimmed_prefix

    # Assemble the FIM prompt
    # NOTE: No spaces around the tokens — the model is sensitive to whitespace
    # adjacent to FIM special tokens. Test this if completions look wrong.
    prompt = (
        f"{FIM_PREFIX_TOKEN}"
        f"{effective_prefix}"
        f"{FIM_SUFFIX_TOKEN}"
        f"{trimmed_suffix}"
        f"{FIM_MIDDLE_TOKEN}"
    )

    # Rough token estimate: ~4 chars per token (good enough for budget checks)
    token_estimate = len(prompt) // 4

    logger.debug(
        f"FIM prompt built | lang={language_id} | "
        f"prefix_lines={len(trimmed_prefix.splitlines())} | "
        f"suffix_lines={len(trimmed_suffix.splitlines())} | "
        f"context_len={len(injected_context)} | "
        f"token_estimate={token_estimate}"
    )

    return FIMPayload(
        prompt=prompt,
        prefix=prefix,
        suffix=suffix,
        injected_context=injected_context,
        language_id=language_id,
        token_estimate=token_estimate,
    )


def extract_clean_completion(raw: str) -> str:
    """
    Strip FIM stop tokens and trailing whitespace from model output.

    The model sometimes echoes stop tokens in its output — especially
    with smaller quantized models. We clean these before returning to
    the editor.

    Args:
        raw: Raw string from Ollama.

    Returns:
        Clean completion text.
    """
    result = raw
    for stop_token in FIM_STOP_SEQUENCES:
        result = result.split(stop_token)[0]

    # Strip trailing whitespace but preserve internal structure
    result = result.rstrip()

    return result
