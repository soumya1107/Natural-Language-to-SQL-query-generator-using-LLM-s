"""
llm/ollama_llm.py
─────────────────
Ollama adapter for locally-running models (Llama 3, Mistral, CodeLlama, etc.)
Calls the Ollama REST API at http://localhost:11434 by default.
"""

import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from llm.base import BaseLLM, Message, LLMError
from utils.logger import get_logger

logger = get_logger(__name__)


class OllamaLLM(BaseLLM):

    def __init__(
        self,
        model: str = "llama3",
        temperature: float = 0.0,
        max_tokens: int = 1024,
        base_url: str = "http://localhost:11434",
    ):
        super().__init__(model, temperature, max_tokens)
        self.base_url = base_url.rstrip("/")

    @property
    def provider(self) -> str:
        return "Ollama (local)"

    @retry(stop=stop_after_attempt(2), wait=wait_exponential(min=1, max=5))
    def generate(self, messages: list[Message]) -> str:
        try:
            payload = {
                "model": self.model,
                "messages": [{"role": m.role, "content": m.content} for m in messages],
                "stream": False,
                "options": {
                    "temperature": self.temperature,
                    "num_predict": self.max_tokens,
                },
            }
            resp = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=120,
            )
            resp.raise_for_status()
            return resp.json()["message"]["content"].strip()
        except requests.exceptions.ConnectionError:
            raise LLMError(
                f"Cannot connect to Ollama at {self.base_url}. "
                "Is Ollama running? Start it with: ollama serve"
            )
        except Exception as e:
            logger.error(f"Ollama error: {e}")
            raise LLMError(f"Ollama call failed: {e}") from e

    def list_models(self) -> list[str]:
        """Return locally available model names."""
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            resp.raise_for_status()
            return [m["name"] for m in resp.json().get("models", [])]
        except Exception:
            return []
