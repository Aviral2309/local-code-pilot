"""
config.py
---------
Loads and validates AuraLSP configuration from JSON.
Provides a single shared Config object across all modules.

Design: One config object, loaded once at startup, accessed everywhere.
No global mutable state — pass config explicitly where needed.
"""

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Default config path — can be overridden via AURALSP_CONFIG env var
DEFAULT_CONFIG_PATH = Path(__file__).parent.parent / "config" / "default.json"


@dataclass
class ServerConfig:
    host: str = "127.0.0.1"
    port: int = 2087
    log_level: str = "INFO"
    log_file: str = "auralsp.log"


@dataclass
class OllamaConfig:
    base_url: str = "http://localhost:11434"
    model: str = "qwen2.5-coder:1.5b"
    temperature: float = 0.1
    top_p: float = 0.9
    repeat_penalty: float = 1.05
    stream: bool = True
    timeout_seconds: int = 30


@dataclass
class CompletionConfig:
    debounce_ms: int = 200
    max_prefix_lines: int = 50
    max_suffix_lines: int = 20
    max_completion_tokens: int = 128
    trigger_characters: list = field(default_factory=lambda: [".", "(", " "])


@dataclass
class KnapsackConfig:
    token_cap: int = 1800
    weight_semantic: float = 0.5
    weight_graph_distance: float = 0.3
    weight_recency: float = 0.2
    top_k_candidates: int = 20


@dataclass
class EmbeddingConfig:
    model_name: str = "all-MiniLM-L6-v2"
    batch_size: int = 32
    vector_dim: int = 384


@dataclass
class MetricsConfig:
    enabled: bool = True
    db_path: str = "auralsp_telemetry.db"
    metrics_port: int = 8765


@dataclass
class Config:
    server: ServerConfig = field(default_factory=ServerConfig)
    ollama: OllamaConfig = field(default_factory=OllamaConfig)
    completion: CompletionConfig = field(default_factory=CompletionConfig)
    knapsack: KnapsackConfig = field(default_factory=KnapsackConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)


def load_config(path: Optional[Path] = None) -> Config:
    """
    Load config from JSON file. Falls back to defaults for missing keys.
    Environment variable AURALSP_CONFIG overrides the path argument.

    Args:
        path: Path to config JSON. Uses DEFAULT_CONFIG_PATH if None.

    Returns:
        Populated Config dataclass.

    Raises:
        FileNotFoundError: If the specified config file does not exist.
        json.JSONDecodeError: If the config file is malformed JSON.
    """
    # Environment variable takes highest priority
    env_path = os.environ.get("AURALSP_CONFIG")
    if env_path:
        path = Path(env_path)

    config_path = path or DEFAULT_CONFIG_PATH

    if not config_path.exists():
        logger.warning(
            f"Config file not found at {config_path}. Using all defaults."
        )
        return Config()

    with open(config_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    logger.info(f"Loaded config from {config_path}")

    return Config(
        server=ServerConfig(**raw.get("server", {})),
        ollama=OllamaConfig(**raw.get("ollama", {})),
        completion=CompletionConfig(**raw.get("completion", {})),
        knapsack=KnapsackConfig(**raw.get("knapsack", {})),
        embedding=EmbeddingConfig(**raw.get("embedding", {})),
        metrics=MetricsConfig(**raw.get("metrics", {})),
    )
