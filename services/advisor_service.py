from services.interface.chat_interface import IChatAdvisor
from utils.groq_client import GroqClient
from utils.ollama_client import OllamaClient
from utils.gemini_client import GeminiClient
from core.config import settings
from system_prompts.prompt_v1 import SYSTEM_ADVISOR_PROMPT
from system_prompts.prompt_llama import SYSTEM_LLAMA_PROMPT
from core.logger import logger


class AdvisorService(IChatAdvisor):
    def __init__(self):
        self.provider_name = settings.LLM_PROVIDER.upper()
        self.system_prompt = SYSTEM_ADVISOR_PROMPT

        if self.provider_name == "GROQ":
            self.client = GroqClient()
            # Llama models respond better to a more direct prompt
            self.system_prompt = SYSTEM_LLAMA_PROMPT
        elif self.provider_name == "GEMINI":
            self.client = GeminiClient()
        else:
            self.client = OllamaClient()

    async def get_recommendation(self, disease: str, confidence: float) -> str:
        prompt = f"The model detected '{disease}' with {confidence*100:.1f}% confidence. Please provide advice."
        try:
            return await self.client.generate(prompt, self.system_prompt)
        except Exception as e:
            logger.error(f"AdvisorService Error: {str(e)}")
            return "Unable to generate recommendation at this time."

    async def get_recommendation_stream(self, disease: str, confidence: float):
        """Async generator for streaming LLM recommendations."""
        prompt = f"The model detected '{disease}' with {confidence*100:.1f}% confidence. Please provide advice."
        try:
            async for token in self.client.generate_stream(
                prompt, self.system_prompt
            ):
                yield token
        except Exception as e:
            logger.error(f"Advisor Stream Error: {str(e)}")
            yield "Unable to stream recommendation."

