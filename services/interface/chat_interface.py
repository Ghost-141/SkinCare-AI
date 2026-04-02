from abc import ABC, abstractmethod


class IChatAdvisor(ABC):
    @abstractmethod
    def get_recommendation(self, disease: str, confidence: float) -> str:
        pass
