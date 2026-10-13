"""
llm/openai_llm.py
─────────────────
OpenAI adapter (GPT-4o, GPT-4o-mini, GPT-3.5-turbo).
"""

from tenacity import retry, stop_after_attempt, wait_exponential

from llm.base import BaseLLM, Message, LLMError
from utils.logger import get_logger

logger = get_logger(__name__)


class OpenAILLM(BaseLLM):

    def __init__(self, model: str = "gpt-4o", temperature: float = 0.0, max_tokens: int = 1024, api_key: str = ""):
        super().__init__(model, temperature, max_tokens)
        try:
            from openai import OpenAI
            self._client = OpenAI(api_key=api_key or None)   # falls back to OPENAI_API_KEY env var
        except ImportError:
            raise ImportError("openai package not installed. Run: pip install openai")

    @property
    def provider(self) -> str:
        return "OpenAI"

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10))
    def generate(self, messages: list[Message]) -> str:
        try:
            oai_messages = [{"role": m.role, "content": m.content} for m in messages]
            response = self._client.chat.completions.create(
                model=self.model,
                messages=oai_messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"OpenAI error: {e}")
            raise LLMError(f"OpenAI call failed: {e}") from e
