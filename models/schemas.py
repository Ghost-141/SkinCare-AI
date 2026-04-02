from pydantic import BaseModel
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

    class Config:
        from_attributes = True

class ChatRequest(BaseModel):
    message: str
    context: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    provider: str
