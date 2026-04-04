import httpx
import json
from core.config import settings
from core.logger import logger
from typing import AsyncGenerator

class OllamaClient:
    def __init__(self):
        self.base_url = settings.OLLAMA_BASE_URL
        self.model = settings.OLLAMA_MODEL

    async def generate_stream(self, prompt: str, system_prompt: str = "") -> AsyncGenerator[str, None]:
        """Stream tokens from Ollama API."""
        full_prompt = f"{system_prompt}\n\nUser: {prompt}\nAssistant:"
        try:
            async with httpx.AsyncClient() as client:
                async with client.stream(
                    "POST",
                    f"{self.base_url}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": full_prompt,
                        "stream": True,
                        "keep_alive": settings.OLLAMA_KEEP_ALIVE
                    },
                    timeout=60.0
                ) as response:
                    async for line in response.aiter_lines():
                        if not line:
                            continue
                        body = json.loads(line)
                        token = body.get("response", "")
                        if token:
                            yield token
                        if body.get("done", False):
                            break
        except Exception as e:
            logger.error(f"Ollama Stream Error: {str(e)}")
            yield "Error: Unable to stream advice from Ollama."

    async def generate(self, prompt: str, system_prompt: str = "") -> str:
        """Standard non-streaming generation."""
        full_prompt = f"{system_prompt}\n\nUser: {prompt}\nAssistant:"
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": full_prompt,
                        "stream": False,
                        "keep_alive": settings.OLLAMA_KEEP_ALIVE
                    },
                    timeout=60.0
                )
                response.raise_for_status()
                return response.json().get("response", "")
        except Exception as e:
            logger.error(f"Ollama Async Error: {str(e)}")
            return "Error: Unable to get advice from Ollama."
