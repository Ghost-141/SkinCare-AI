from fastapi import FastAPI
from api.v1.router import api_router
from core.config import settings
from core.dependency import create_db_and_tables
from utils.logger import logger
import uvicorn
import os

app = FastAPI(
    title=settings.APP_NAME,
    description="Skin Disease Detection & LLM Advisor System",
    version="1.0.0"
)

# Initialize Database
@app.on_event("startup")
def startup_event():
    create_db_and_tables()
    # Create upload directory if it doesn't exist
    if not os.path.exists(settings.UPLOAD_DIR):
        os.makedirs(settings.UPLOAD_DIR)
    logger.info("Application starting up...")

# Include API Router
app.include_router(api_router, prefix="/api/v1")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
