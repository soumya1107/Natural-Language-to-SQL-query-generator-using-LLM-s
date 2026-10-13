"""
config/settings.py
──────────────────
Central configuration loaded from environment variables (.env file).
All tuneable knobs live here — import this module wherever needed.
"""

import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()


# ── Database ──────────────────────────────────────────────────────────────────

@dataclass
class DBConfig:
    host: str = field(default_factory=lambda: os.getenv("DB_HOST", "localhost"))
    port: int = field(default_factory=lambda: int(os.getenv("DB_PORT", "3306")))
    user: str = field(default_factory=lambda: os.getenv("DB_USER", "root"))
    password: str = field(default_factory=lambda: os.getenv("DB_PASSWORD", ""))
    database: str = field(default_factory=lambda: os.getenv("DB_NAME", ""))
    connect_timeout: int = 10


# ── LLM ───────────────────────────────────────────────────────────────────────

@dataclass
class LLMConfig:
    # API keys
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    anthropic_api_key: str = field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", ""))
    google_api_key: str = field(default_factory=lambda: os.getenv("GOOGLE_API_KEY", ""))
    ollama_base_url: str = field(default_factory=lambda: os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
    ollama_default_model: str = field(default_factory=lambda: os.getenv("OLLAMA_DEFAULT_MODEL", "llama3"))

    # Model options per provider
    openai_models: list = field(default_factory=lambda: ["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"])
    anthropic_models: list = field(default_factory=lambda: ["claude-opus-4-6", "claude-sonnet-4-6"])
    gemini_models: list = field(
        default_factory=lambda: [
            "gemini-3.5-flash",
            "gemini-2.5-flash",
            "gemini-2.5-pro",
        ]
    )

    # Generation params
    temperature: float = 0.0          # Low temp = deterministic SQL
    max_tokens: int = 1024


# ── App behaviour ─────────────────────────────────────────────────────────────

@dataclass
class AppConfig:
    max_result_rows: int = field(default_factory=lambda: int(os.getenv("MAX_RESULT_ROWS", "500")))
    read_only_guard: bool = field(default_factory=lambda: os.getenv("READ_ONLY_GUARD", "true").lower() == "true")
    schema_chunk_size: int = field(default_factory=lambda: int(os.getenv("SCHEMA_CHUNK_SIZE", "3000")))  # tokens
    history_max: int = 50             # Max queries stored in session history


# ── Singleton accessors ───────────────────────────────────────────────────────

_db_config: DBConfig | None = None
_llm_config: LLMConfig | None = None
_app_config: AppConfig | None = None


def get_db_config() -> DBConfig:
    global _db_config
    if _db_config is None:
        _db_config = DBConfig()
    return _db_config


def get_llm_config() -> LLMConfig:
    global _llm_config
    if _llm_config is None:
        _llm_config = LLMConfig()
    return _llm_config


def get_app_config() -> AppConfig:
    global _app_config
    if _app_config is None:
        _app_config = AppConfig()
    return _app_config
