"""Google Gemini adapter using the supported Google GenAI SDK."""

from __future__ import annotations

from tenacity import retry, stop_after_attempt, wait_exponential

from llm.base import BaseLLM, LLMError, Message
from utils.logger import get_logger

logger = get_logger(__name__)


class GeminiLLM(BaseLLM):
    def __init__(
        self,
        model: str = "gemini-3.5-flash",
        temperature: float = 0.0,
        max_tokens: int = 1024,
        api_key: str = "",
    ):
        super().__init__(model, temperature, max_tokens)
        try:
            from google import genai
            from google.genai import types

            self._types = types
            self._client = genai.Client(api_key=api_key or None)
        except ImportError as exc:
            raise ImportError(
                "google-genai is not installed. Run: pip install google-genai"
            ) from exc

    @property
    def provider(self) -> str:
        return "Google Gemini"

    def _prepare_request(self, messages: list[Message]):
        """
        Convert provider-neutral messages into Gemini contents.

        Gemini supports a dedicated system instruction. Assistant messages use
        the Gemini role name ``model`` so this also works for future multi-turn
        prompts.
        """
        system_parts: list[str] = []
        contents = []
        for message in messages:
            if message.role == "system":
                system_parts.append(message.content)
                continue

            role = "model" if message.role == "assistant" else "user"
            contents.append(
                self._types.Content(
                    role=role,
                    parts=[self._types.Part.from_text(text=message.content)],
                )
            )

        if not contents:
            raise ValueError("Gemini request requires at least one non-system message.")

        config = self._types.GenerateContentConfig(
            system_instruction="\n\n".join(system_parts) or None,
            temperature=self.temperature,
            max_output_tokens=self.max_tokens,
        )
        return contents, config

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10))
    def generate(self, messages: list[Message]) -> str:
        try:
            contents, config = self._prepare_request(messages)
            response = self._client.models.generate_content(
                model=self.model,
                contents=contents,
                config=config,
            )
            text = response.text
            if not text:
                raise LLMError("Gemini returned no text response.")
            return text.strip()
        except LLMError:
            raise
        except Exception as exc:
            logger.error("Gemini error: %s", exc)
            raise LLMError(f"Gemini call failed: {exc}") from exc

    def close(self) -> None:
        """Release the SDK HTTP client when an explicit lifecycle is used."""
        self._client.close()
