from fastapi import FastAPI
from fastapi.responses import JSONResponse
from api.v1.router import api_router
from core.config import settings
from core.dependency import create_db_and_tables
from core.logger import logger
import uvicorn
import os
from pathlib import Path
from contextlib import asynccontextmanager


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize Database
    create_db_and_tables()
    try:
        os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    except PermissionError:
        fallback_path = os.path.join(Path.home(), ".skincare_ai", "uploads")
        os.makedirs(fallback_path, exist_ok=True)
        settings.UPLOAD_DIR = fallback_path
        logger.warning(
            f"Upload directory not accessible. Using fallback: {fallback_path}"
        )
    logger.info("Application starting up...")
    yield
    logger.info("Application shutting down...")


app = FastAPI(
    title=settings.APP_NAME,
    description="Skin Disease Detection & LLM Advisor System",
    version="1.0.0",
    lifespan=lifespan,
)


# Override the default message of API
@app.get("/")
async def root():
    return JSONResponse(
        content={
            "app": settings.APP_NAME,
            "status": "running",
            "docs": "http://127.0.0.1:8000/docs",
        }
    )


# Include API Router
app.include_router(api_router, prefix="/api/v1")

if __name__ == "__main__":
    is_dev = settings.ENV_MODE == "dev"
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=is_dev)
