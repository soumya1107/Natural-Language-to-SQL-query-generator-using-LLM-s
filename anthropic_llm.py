"""
llm/anthropic_llm.py
────────────────────
Anthropic adapter (Claude Opus 4, Claude Sonnet 4, etc.)
"""

from tenacity import retry, stop_after_attempt, wait_exponential

from llm.base import BaseLLM, Message, LLMError
from utils.logger import get_logger

logger = get_logger(__name__)


class AnthropicLLM(BaseLLM):

    def __init__(
        self,
        model: str = "claude-sonnet-4-6",
        temperature: float = 0.0,
        max_tokens: int = 1024,
        api_key: str = "",
    ):
        super().__init__(model, temperature, max_tokens)
        try:
            import anthropic
            self._client = anthropic.Anthropic(api_key=api_key or None)
        except ImportError:
            raise ImportError("anthropic package not installed. Run: pip install anthropic")

    @property
    def provider(self) -> str:
        return "Anthropic"

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10))
    def generate(self, messages: list[Message]) -> str:
        try:
            # Anthropic separates system prompt from the messages list
            system_content = ""
            user_messages = []

            for m in messages:
                if m.role == "system":
                    system_content = m.content
                else:
                    user_messages.append({"role": m.role, "content": m.content})

            kwargs = dict(
                model=self.model,
                max_tokens=self.max_tokens,
                messages=user_messages,
                temperature=self.temperature,
            )
            if system_content:
                kwargs["system"] = system_content

            response = self._client.messages.create(**kwargs)
            return response.content[0].text.strip()
        except Exception as e:
            logger.error(f"Anthropic error: {e}")
            raise LLMError(f"Anthropic call failed: {e}") from e
