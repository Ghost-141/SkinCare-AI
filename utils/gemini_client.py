from google import genai
from core.config import settings
from core.logger import logger
from typing import AsyncGenerator

class GeminiClient:
    def __init__(self):
        # The new SDK uses a Client object
        self.client = genai.Client(api_key=settings.GOOGLE_API_KEY)
        self.model_name = settings.GEMINI_MODEL

    async def generate_stream(self, prompt: str, system_prompt: str = "") -> AsyncGenerator[str, None]:
        """Stream tokens from Gemini API using the modern google-genai SDK."""
        try:
            # The new SDK combines content and config
            config = {"system_instruction": system_prompt} if system_prompt else None
            
            # models.generate_content_stream is the method for streaming
            async for chunk in await self.client.aio.models.generate_content_stream(
                model=self.model_name,
                contents=prompt,
                config=config
            ):
                if chunk.text:
                    yield chunk.text
        except Exception as e:
            logger.error(f"Gemini Stream Error: {str(e)}")
            yield "Error: Unable to stream advice from Gemini."

    async def generate(self, prompt: str, system_prompt: str = "") -> str:
        """Standard non-streaming generation using the modern google-genai SDK."""
        try:
            config = {"system_instruction": system_prompt} if system_prompt else None
            
            response = await self.client.aio.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=config
            )
            return response.text
        except Exception as e:
            logger.error(f"Gemini API Error: {str(e)}")
            return "Error: Unable to get advice from Gemini at this time."
