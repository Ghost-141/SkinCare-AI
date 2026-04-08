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
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from core.db import AsyncSessionLocal, commit_to_db
from core.dependency import get_db, get_skin_service, get_advisor_service
from services.skin_service import SkinService
from services.advisor_service import AdvisorService
from models.db_models import SkinAnalysisLog
from core.config import settings
from core.logger import logger
from utils.file_validator import validate_upload
from utils.visualization import create_and_save_heatmap
from anyio import to_thread
import shutil
import os
import uuid
import json
from datetime import datetime, timezone

router = APIRouter()


async def update_llm_recommendation(log_id: int, recommendation: str):
    """Background task to update the LLM recommendation in the database asynchronously."""
    async with AsyncSessionLocal() as db:
        try:
            result = await db.execute(
                select(SkinAnalysisLog).filter(SkinAnalysisLog.id == log_id)
            )
            log = result.scalars().first()
            if log:
                log.llm_recommendation = recommendation
                await db.commit()
                logger.info(f"Updated LLM recommendation for log ID: {log_id}")
        except Exception as e:
            logger.error(f"Error updating LLM recommendation: {str(e)}")
        finally:
            await db.close()


@router.post("/analyze_skin")
async def analyze_skin(
    background_tasks: BackgroundTasks,
    user_id: str = Form(...),  # Patient ID
    patient_name: str = Form(...),  # Patient Name
    age: int = Form(...),
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
    skin_service: SkinService = Depends(get_skin_service),
    advisor_service: AdvisorService = Depends(get_advisor_service),
):
    """Single API for image classification and streaming LLM advice."""
    try:
        # Strip user_id to prevent whitespace issues
        user_id = user_id.strip()
        patient_name = patient_name.strip()

        # Validate File
        content = await validate_upload(file)

        # 1. Save Image
        file_extension = os.path.splitext(file.filename)[1]
        file_name = f"{uuid.uuid4()}{file_extension}"
        file_path = os.path.join(settings.UPLOAD_DIR, file_name)

        def save_file():
            with open(file_path, "wb") as buffer:
                buffer.write(content)

        await to_thread.run_sync(save_file)

        # 2. Run Classification
        try:
            prediction, confidence, index = await to_thread.run_sync(
                skin_service.predict, file_path
            )
        except ValueError as e:
            logger.error(f"Input validation error: {str(e)}")
            raise HTTPException(status_code=400, detail=str(e))
        except RuntimeError as e:
            logger.error(f"Prediction logic error: {str(e)}")
            raise HTTPException(
                status_code=422, detail="Analysis engine failed to process the image."
            )
        except Exception as e:
            logger.error(f"Unexpected prediction error: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail="An unexpected error occurred during prediction.",
            )

        # 2.1 Generate Heatmap
        heatmap_name = f"heatmap_{os.path.basename(file_path)}"
        heatmap_path = os.path.join(settings.UPLOAD_DIR, heatmap_name)
        model_path = getattr(skin_service, "current_model_path", settings.MODEL_PATH)
        await to_thread.run_sync(
            create_and_save_heatmap, model_path, file_path, heatmap_path, index
        )

        # 3. Create Initial DB Log
        db_log = SkinAnalysisLog(
            user_id=user_id,
            patient_name=patient_name,
            age=age,
            image_path=file_path,
            prediction=prediction,
            accuracy=confidence,
            llm_recommendation="Generating recommendation...",
            llm_provider=settings.LLM_PROVIDER,
        )
        await commit_to_db(db, db_log)
        log_id = db_log.id

        async def combined_generator():

            full_recommendation = []

            # Metadata chunk
            metadata = {
                "user_id": user_id,
                "patient_name": patient_name,
                "age": age,
                "prediction": prediction,
                "accuracy": confidence,
                "heatmap_path": heatmap_path,
                "created_at": (
                    db_log.created_at.isoformat()
                    if db_log.created_at
                    else datetime.now(timezone.utc).isoformat()
                ),
            }
            yield json.dumps(metadata) + "||METADATA_END||"

            # 4. Stream LLM Recommendation
            async for token in advisor_service.get_recommendation_stream(
                prediction, confidence
            ):
                full_recommendation.append(token)
                yield token

            # 5. Schedule update for the full text
            final_text = "".join(full_recommendation)
            background_tasks.add_task(update_llm_recommendation, log_id, final_text)

        return StreamingResponse(combined_generator(), media_type="text/event-stream")

    except HTTPException:
        raise

    except Exception as e:
        logger.error(f"Single API Error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Server Error")


@router.get("/history/{user_id}", response_model=List[AnalysisResponse])
async def get_user_history(user_id: str, db: AsyncSession = Depends(get_db)):
    """Fetch all past scans for a specific user ID."""
    user_id = user_id.strip()
    result = await db.execute(
        select(SkinAnalysisLog)
        .filter(SkinAnalysisLog.user_id == user_id)
        .order_by(SkinAnalysisLog.created_at.desc())
    )
    logs = result.scalars().all()
    return logs
