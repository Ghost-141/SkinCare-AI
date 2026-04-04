from services.interface.chat_interface import IChatAdvisor
from utils.groq_client import GroqClient
from utils.ollama_client import OllamaClient
from core.config import settings
from system_prompts.prompt_v1 import SYSTEM_ADVISOR_PROMPT
from core.logger import logger


class AdvisorService(IChatAdvisor):
    def __init__(self):
        self.provider_name = settings.LLM_PROVIDER
        if self.provider_name == "Groq":
            self.client = GroqClient()
        else:
            self.client = OllamaClient()

    async def get_recommendation(self, disease: str, confidence: float) -> str:
        prompt = f"The model detected '{disease}' with {confidence*100:.1f}% confidence. Please provide advice."
        try:
            if self.provider_name == "Groq":
                # Note: Groq SDK is synchronous by default.
                # For real production, use groq.AsyncGroq()
                return self.client.generate(prompt, SYSTEM_ADVISOR_PROMPT)
            return await self.client.generate(prompt, SYSTEM_ADVISOR_PROMPT)
        except Exception as e:
            logger.error(f"AdvisorService Error: {str(e)}")
            return "Unable to generate recommendation at this time."

    async def get_recommendation_stream(self, disease: str, confidence: float):
        """Async generator for streaming LLM recommendations."""
        prompt = f"The model detected '{disease}' with {confidence*100:.1f}% confidence. Please provide advice."
        try:
            if self.provider_name == "Groq":
                # Fallback for Groq (simplified for demo)
                yield self.client.generate(prompt, SYSTEM_ADVISOR_PROMPT)
            else:
                async for token in self.client.generate_stream(
                    prompt, SYSTEM_ADVISOR_PROMPT
                ):
                    yield token
        except Exception as e:
            logger.error(f"Advisor Stream Error: {str(e)}")
            yield "Unable to stream recommendation."
