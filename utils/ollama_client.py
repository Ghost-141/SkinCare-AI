import httpx
import json
from core.config import settings
from core.logger import logger
from typing import AsyncGenerator

class OllamaClient:
    def __init__(self):
        self.base_url = settings.OLLAMA_BASE_URL
        self.model = settings.OLLAMA_MODEL
        # Shared httpx client with connection pooling
        timeout = httpx.Timeout(10.0, read=None)
        self.client = httpx.AsyncClient(timeout=timeout)

    async def generate_stream(self, prompt: str, system_prompt: str = "") -> AsyncGenerator[str, None]:
        """Stream tokens from Ollama API using /api/generate."""
        try:
            async with self.client.stream(
                "POST",
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "system": system_prompt,
                    "stream": True,
                    "keep_alive": settings.OLLAMA_KEEP_ALIVE
                }
            ) as response:
                if response.status_code != 200:
                    error_text = await response.aread()
                    logger.error(f"Ollama Error: {response.status_code} - {error_text.decode()}")
                    yield f"Error: Ollama returned status {response.status_code}"
                    return

                async for line in response.aiter_lines():
                    if not line:
                        continue
                    try:
                        body = json.loads(line)
                        token = body.get("response", "")
                        if token:
                            yield token
                        if body.get("done", False):
                            break
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            logger.error(f"Ollama Stream Error: {str(e)}")
            yield f"Error: Unable to stream advice from Ollama."

    async def generate(self, prompt: str, system_prompt: str = "") -> str:
        """Standard non-streaming generation using /api/generate."""
        try:
            response = await self.client.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "system": system_prompt,
                    "stream": False,
                    "keep_alive": settings.OLLAMA_KEEP_ALIVE
                },
                timeout=60.0
            )
            response.raise_for_status()
            return response.json().get("response", "")
        except Exception as e:
            logger.error(f"Ollama Async Error: {str(e)}")
            return f"Error: Unable to get advice from Ollama ({str(e)})."
