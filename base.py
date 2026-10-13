"""
llm/base.py
───────────
Abstract base class for all LLM adapters.
Every adapter must implement `generate(messages) -> str`.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class Message:
    role: str    # "system" | "user" | "assistant"
    content: str


class BaseLLM(ABC):
    """
    Minimal interface every LLM adapter must satisfy.
    """

    def __init__(self, model: str, temperature: float = 0.0, max_tokens: int = 1024):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        # The factory replaces these conservative defaults with provider/model
        # metadata. Keeping defaults here supports directly constructed adapters.
        self.provider_key = ""
        self.context_window = 8192
        self.recommended_schema_tokens = 3000
        self.supports_system_messages = True

    @abstractmethod
    def generate(self, messages: list[Message]) -> str:
        """
        Send *messages* to the LLM and return the text response.
        Raises LLMError on failure.
        """
        ...

    @property
    @abstractmethod
    def provider(self) -> str:
        """Human-readable provider name, e.g. 'OpenAI'."""
        ...


class LLMError(Exception):
    """Raised when an LLM call fails after retries."""
