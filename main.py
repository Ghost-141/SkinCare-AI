from fastapi import FastAPI
from api.v1.router import api_router
from core.config import settings
from core.dependency import create_db_and_tables
from utils.logger import logger
import uvicorn
import os

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize Database
    create_db_and_tables()
    # Create upload directory if it doesn't exist
    if not os.path.exists(settings.UPLOAD_DIR):
        os.makedirs(settings.UPLOAD_DIR)
    logger.info("Application starting up...")
    yield
    logger.info("Application shutting down...")

app = FastAPI(
    title=settings.APP_NAME,
    description="Skin Disease Detection & LLM Advisor System",
    version="1.0.0",
    lifespan=lifespan
)

# Include API Router
app.include_router(api_router, prefix="/api/v1")

if __name__ == "__main__":
    is_dev = settings.ENV_MODE == "dev"
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=is_dev)
