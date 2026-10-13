"""
LLM adapter factory and model capability metadata.

Prompt construction uses the capability metadata to keep the schema context
inside a sensible input budget without coupling the prompt builder to a
specific provider SDK.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Any

from config.settings import get_llm_config
from llm.base import BaseLLM


@dataclass(frozen=True)
class ModelCapabilities:
    context_window: int
    recommended_schema_tokens: int
    supports_system_messages: bool = True


_PROVIDER_CLASSES = {
    "openai": ("llm.openai_llm", "OpenAILLM"),
    "anthropic": ("llm.anthropic_llm", "AnthropicLLM"),
    "gemini": ("llm.gemini_llm", "GeminiLLM"),
    "ollama": ("llm.ollama_llm", "OllamaLLM"),
}

_DEFAULT_CAPABILITIES = {
    "openai": ModelCapabilities(128_000, 8_000),
    "anthropic": ModelCapabilities(200_000, 8_000),
    "gemini": ModelCapabilities(1_000_000, 10_000),
    "ollama": ModelCapabilities(8_192, 3_000),
}

_MODEL_CAPABILITIES = {
    ("openai", "gpt-4o"): ModelCapabilities(128_000, 8_000),
    ("openai", "gpt-4o-mini"): ModelCapabilities(128_000, 6_000),
    ("openai", "gpt-3.5-turbo"): ModelCapabilities(16_385, 3_000),
    ("anthropic", "claude-opus-4-6"): ModelCapabilities(200_000, 8_000),
    ("anthropic", "claude-sonnet-4-6"): ModelCapabilities(200_000, 8_000),
    ("gemini", "gemini-3.5-flash"): ModelCapabilities(1_000_000, 10_000),
    ("gemini", "gemini-2.5-pro"): ModelCapabilities(1_000_000, 10_000),
    ("gemini", "gemini-2.5-flash"): ModelCapabilities(1_000_000, 8_000),
}


def get_model_capabilities(provider: str, model: str | None = None) -> ModelCapabilities:
    """Return conservative prompt-budget metadata for a provider/model."""
    provider_key = provider.lower().strip()
    if provider_key not in _PROVIDER_CLASSES:
        raise ValueError(
            f"Unknown provider '{provider}'. "
            f"Choose from: {', '.join(_PROVIDER_CLASSES)}"
        )
    return _MODEL_CAPABILITIES.get(
        (provider_key, model or ""),
        _DEFAULT_CAPABILITIES[provider_key],
    )


def _default_model(provider: str) -> str:
    cfg = get_llm_config()
    defaults = {
        "openai": cfg.openai_models[0],
        "anthropic": cfg.anthropic_models[0],
        "gemini": cfg.gemini_models[0],
        "ollama": cfg.ollama_default_model,
    }
    return defaults[provider]


def get_llm(
    provider: str,
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    **kwargs: Any,
) -> BaseLLM:
    """Build an LLM adapter and attach model-aware prompt capabilities."""
    provider_key = provider.lower().strip()
    if provider_key not in _PROVIDER_CLASSES:
        raise ValueError(
            f"Unknown provider '{provider}'. "
            f"Choose from: {', '.join(_PROVIDER_CLASSES)}"
        )

    cfg = get_llm_config()
    resolved_model = model or _default_model(provider_key)
    constructor_kwargs: dict[str, Any] = {
        "model": resolved_model,
        "temperature": temperature if temperature is not None else cfg.temperature,
        "max_tokens": max_tokens if max_tokens is not None else cfg.max_tokens,
    }

    if provider_key == "openai":
        constructor_kwargs["api_key"] = kwargs.pop("api_key", None) or cfg.openai_api_key
    elif provider_key == "anthropic":
        constructor_kwargs["api_key"] = kwargs.pop("api_key", None) or cfg.anthropic_api_key
    elif provider_key == "gemini":
        constructor_kwargs["api_key"] = kwargs.pop("api_key", None) or cfg.google_api_key
    else:
        constructor_kwargs["base_url"] = (
            kwargs.pop("base_url", None) or cfg.ollama_base_url
        )

    constructor_kwargs.update(kwargs)
    module_name, class_name = _PROVIDER_CLASSES[provider_key]
    adapter_class = getattr(import_module(module_name), class_name)
    llm = adapter_class(**constructor_kwargs)

    capabilities = get_model_capabilities(provider_key, resolved_model)
    llm.provider_key = provider_key
    llm.context_window = capabilities.context_window
    llm.recommended_schema_tokens = capabilities.recommended_schema_tokens
    llm.supports_system_messages = capabilities.supports_system_messages
    return llm


PROVIDER_DISPLAY = {
    "openai": "OpenAI",
    "anthropic": "Anthropic (Claude)",
    "gemini": "Google Gemini",
    "ollama": "Ollama (Local)",
}

PROVIDER_MODELS = {
    "openai": ["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"],
    "anthropic": ["claude-opus-4-6", "claude-sonnet-4-6"],
    "gemini": ["gemini-3.5-flash", "gemini-2.5-flash", "gemini-2.5-pro"],
    "ollama": [],
}
