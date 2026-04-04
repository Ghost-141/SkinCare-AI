from fastapi import APIRouter, Depends, HTTPException
import requests
import shutil
import os
from sqlalchemy.orm import Session
from core.db import check_db_status
from core.dependency import get_db, get_skin_service
from core.config import settings
from services.skin_service import SkinService
from core.logger import logger

router = APIRouter()

@router.get("/health")
async def health_check(
    db: Session = Depends(get_db),
    skin_service: SkinService = Depends(get_skin_service)
):
    health_status = {
        "status": "healthy",
        "services": {}
    }

    # 1. Check Database
    status = check_db_status()
    health_status["services"]["database"] = status
    if "offline" in status:
        health_status["status"] = "unhealthy"

    # 2. Check Skin Classification Model
    try:
        if skin_service.model is not None:
            health_status["services"]["skin_model"] = {
                "status": "loaded",
                "device": str(skin_service.device),
                "model_path": settings.MODEL_PATH
            }
        else:
            health_status["services"]["skin_model"] = "not_loaded"
            health_status["status"] = "degraded"
    except Exception:
        health_status["services"]["skin_model"] = "error"
        health_status["status"] = "unhealthy"

    # 4. Check LLM Provider Availability
    llm_provider = settings.LLM_PROVIDER.lower()
    try:
        if llm_provider == "ollama":
            resp = requests.get(f"{settings.OLLAMA_BASE_URL}/api/tags", timeout=5)
            if resp.status_code == 200:
                available_models = [m['name'] for m in resp.json().get('models', [])]
                is_pulled = any(
                    settings.OLLAMA_MODEL == m or m.startswith(f"{settings.OLLAMA_MODEL}:")
                    for m in available_models
                )
                health_status["services"]["llm"] = {
                    "provider": "Ollama",
                    "status": "online" if is_pulled else "model_not_pulled",
                    "model": settings.OLLAMA_MODEL,
                    "available_local_models": available_models
                }
                if not is_pulled:
                    health_status["status"] = "degraded"
            else:
                health_status["services"]["llm"] = {"provider": "Ollama", "status": "server_unreachable"}
                health_status["status"] = "degraded"
        
        elif llm_provider == "groq":
            if not settings.GROQ_API_KEY or "your_groq_api_key" in settings.GROQ_API_KEY:
                health_status["services"]["llm"] = {"provider": "Groq", "status": "missing_key"}
                health_status["status"] = "degraded"
            else:
                # Actual connectivity/auth check with Groq
                try:
                    resp = requests.get(
                        "https://api.groq.com/openai/v1/models",
                        headers={"Authorization": f"Bearer {settings.GROQ_API_KEY}"},
                        timeout=5
                    )
                    if resp.status_code == 200:
                        health_status["services"]["llm"] = {
                            "provider": "Groq", 
                            "status": "online",
                            "model": settings.GROQ_MODEL
                        }
                    else:
                        health_status["services"]["llm"] = {
                            "provider": "Groq", 
                            "status": f"api_error: {resp.status_code}",
                            "model": settings.GROQ_MODEL
                        }
                        health_status["status"] = "degraded"
                except Exception as e:
                    health_status["services"]["llm"] = {"provider": "Groq", "status": f"unreachable: {str(e)}"}
                    health_status["status"] = "degraded"
        else:
            # Add a fallback for unknown providers
            health_status["services"]["llm"] = {"provider": settings.LLM_PROVIDER, "status": "unknown_provider"}
            health_status["status"] = "degraded"
                    
    except Exception as e:
        health_status["services"]["llm"] = {"provider": settings.LLM_PROVIDER, "status": f"error: {str(e)}"}

    return health_status

@router.get("/models")
async def list_models(skin_service: SkinService = Depends(get_skin_service)):
    """List all available model files in the weights directory."""
    models = skin_service.list_models()
    return {"available_models": models, "active_model": settings.MODEL_PATH}

@router.post("/models/select")
async def select_model(model_name: str, skin_service: SkinService = Depends(get_skin_service)):
    """Switch the active AI model in memory."""
    success = skin_service.load_model_by_name(model_name)
    if success:
        return {"message": f"Successfully loaded model: {model_name}"}
    raise HTTPException(status_code=404, detail="Model file not found")
