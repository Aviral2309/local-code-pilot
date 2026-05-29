"""
tests/test_config.py
--------------------
Tests for configuration loading and defaults.
"""

import json
import os
import tempfile
from pathlib import Path

import pytest
from src.config import load_config, Config


class TestLoadConfig:

    def test_loads_default_config(self):
        """Default config file loads without error."""
        config = load_config()
        assert isinstance(config, Config)

    def test_defaults_applied_when_file_missing(self):
        """Non-existent config path returns default Config."""
        config = load_config(Path("/nonexistent/path/config.json"))
        assert config.ollama.model == "qwen2.5-coder:1.5b"
        assert config.completion.debounce_ms == 200
        assert config.knapsack.token_cap == 1800

    def test_custom_config_overrides_defaults(self):
        """Values in JSON file override dataclass defaults."""
        custom = {
            "ollama": {"model": "llama3.2:1b", "temperature": 0.5},
            "completion": {"debounce_ms": 500},
        }
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(custom, f)
            tmp_path = Path(f.name)

        try:
            config = load_config(tmp_path)
            assert config.ollama.model == "llama3.2:1b"
            assert config.ollama.temperature == 0.5
            assert config.completion.debounce_ms == 500
            # Non-overridden values stay at defaults
            assert config.knapsack.token_cap == 1800
        finally:
            tmp_path.unlink()

    def test_env_var_overrides_path(self, tmp_path):
        """AURALSP_CONFIG environment variable takes highest priority."""
        custom = {"ollama": {"model": "deepseek-coder:1.3b"}}
        config_file = tmp_path / "test_config.json"
        config_file.write_text(json.dumps(custom))

        os.environ["AURALSP_CONFIG"] = str(config_file)
        try:
            config = load_config()
            assert config.ollama.model == "deepseek-coder:1.3b"
        finally:
            del os.environ["AURALSP_CONFIG"]

    def test_ollama_config_defaults(self):
        """OllamaConfig defaults are sensible for local inference."""
        config = load_config(Path("/nonexistent"))
        assert config.ollama.base_url == "http://localhost:11434"
        assert 0 <= config.ollama.temperature <= 1.0
        assert config.ollama.timeout_seconds > 0

    def test_knapsack_weights_sum_to_one(self):
        """Knapsack weights should approximately sum to 1.0."""
        config = load_config()
        total = (
            config.knapsack.weight_semantic
            + config.knapsack.weight_graph_distance
            + config.knapsack.weight_recency
        )
        assert abs(total - 1.0) < 0.01, f"Weights sum to {total}, expected ~1.0"

    def test_token_cap_is_positive(self):
        config = load_config()
        assert config.knapsack.token_cap > 0

    def test_trigger_characters_is_list(self):
        config = load_config()
        assert isinstance(config.completion.trigger_characters, list)
        assert len(config.completion.trigger_characters) > 0
