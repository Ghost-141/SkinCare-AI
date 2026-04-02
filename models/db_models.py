from sqlalchemy import Column, Integer, String, Float, Text, DateTime
from sqlalchemy.orm import declarative_base
from datetime import datetime, timezone

Base = declarative_base()


class SkinAnalysisLog(Base):
    __tablename__ = "skin_analysis_logs"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, index=True)  # Patient ID
    patient_name = Column(String)  # Patient Name
    age = Column(Integer)
    image_path = Column(String)
    prediction = Column(String)
    accuracy = Column(Float)
    llm_recommendation = Column(Text)
    llm_provider = Column(String)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
