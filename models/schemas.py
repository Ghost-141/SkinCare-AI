from pydantic import BaseModel, ConfigDict
from datetime import datetime
from typing import Optional


class AnalysisResponse(BaseModel):
    id: int
    user_id: str
    patient_name: Optional[str] = None
    age: int
    image_path: str
    prediction: str
    accuracy: float
    llm_recommendation: str
    llm_provider: str
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)
