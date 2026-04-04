from fastapi import (
    APIRouter,
    UploadFile,
    File,
    Depends,
    HTTPException,
    BackgroundTasks,
    Form,
)
from typing import List
from models.schemas import AnalysisResponse
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from core.db import SessionLocal, commit_to_db
from core.dependency import get_db, get_skin_service, get_advisor_service
from services.skin_service import SkinService
from services.advisor_service import AdvisorService
from models.db_models import SkinAnalysisLog
from core.config import settings
from core.logger import logger
from utils.file_validator import validate_upload
from anyio import to_thread
import shutil
import os
import uuid
import json
from datetime import datetime, timezone

router = APIRouter()


def log_to_db(
    user_id: str,
    patient_name: str,
    age: int,
    image_path: str,
    prediction: str,
    confidence: float,
    recommendation: str,
):
    """Background task to save result to SQLite."""
    db = SessionLocal()
    try:
        db_log = SkinAnalysisLog(
            user_id=user_id,
            patient_name=patient_name,
            age=age,
            image_path=image_path,
            prediction=prediction,
            accuracy=confidence,
            llm_recommendation=recommendation,
            llm_provider=settings.LLM_PROVIDER,
        )
        commit_to_db(db, db_log)
    except Exception as e:
        logger.error(f"Background DB Log Error: {str(e)}")
    finally:
        db.close()


@router.post("/analyze_skin")
async def analyze_skin(
    background_tasks: BackgroundTasks,
    user_id: str = Form(...),  # Patient ID
    patient_name: str = Form(...),  # Patient Name
    age: int = Form(...),
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    skin_service: SkinService = Depends(get_skin_service),
    advisor_service: AdvisorService = Depends(get_advisor_service),
):
    """Single API for image classification and streaming LLM advice."""
    try:

        # Validate File
        content = await validate_upload(file)

        # 1. Save Image (Non-blocking I/O)
        file_extension = os.path.splitext(file.filename)[1]
        file_name = f"{uuid.uuid4()}{file_extension}"
        file_path = os.path.join(settings.UPLOAD_DIR, file_name)

        def save_file():
            with open(file_path, "wb") as buffer:
                buffer.write(content)
                # shutil.copyfileobj(file.file, buffer)

        await to_thread.run_sync(save_file)

        # 2. Run Classification
        prediction, confidence = skin_service.predict(file_path)

        async def combined_generator():

            full_recommendation = []

            # 3. First chunk: Metadata as JSON
            metadata = {
                "user_id": user_id,
                "patient_name": patient_name,
                "age": age,
                "prediction": prediction,
                "accuracy": confidence,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            # We use a separator to help the UI distinguish metadata from tokens
            yield json.dumps(metadata) + "||METADATA_END||"

            # 4. Stream LLM Recommendation
            async for token in advisor_service.get_recommendation_stream(
                prediction, confidence
            ):
                full_recommendation.append(token)
                yield token

            # 5. Log to DB once stream finishes
            final_text = "".join(full_recommendation)
            background_tasks.add_task(
                log_to_db,
                user_id,
                patient_name,
                age,
                file_path,
                prediction,
                confidence,
                final_text,
            )

        return StreamingResponse(combined_generator(), media_type="text/event-stream")

    except HTTPException:
        raise

    except Exception as e:
        logger.error(f"Single API Error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Server Error")


@router.get("/history/{user_id}", response_model=List[AnalysisResponse])
async def get_user_history(user_id: str, db: Session = Depends(get_db)):
    """Fetch all past scans for a specific user ID."""
    logs = (
        db.query(SkinAnalysisLog)
        .filter(SkinAnalysisLog.user_id == user_id)
        .order_by(SkinAnalysisLog.created_at.desc())
        .all()
    )
    return logs
