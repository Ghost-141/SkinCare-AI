from abc import ABC, abstractmethod
import torch

class ISkinAnalysis(ABC):
    @abstractmethod
    def predict(self, image_path: str) -> tuple[str, float, int]:
        pass

    @abstractmethod
    def load_model(self, model_path: str):
        pass
