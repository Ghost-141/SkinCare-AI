from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from api.v1.router import api_router
from core.config import settings
from core.dependency import create_db_and_tables
from core.logger import logger
import uvicorn
import os
import time
from pathlib import Path
from contextlib import asynccontextmanager


@asynccontextmanager
async def lifespan(app: FastAPI):
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
    logger.info("Application started")
    yield
    logger.info("Application shutdown")


app = FastAPI(
    title=settings.APP_NAME,
    description="Skin Disease Detection & LLM Advisor System",
    version="1.0.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()

    # Log request
    logger.info(
        f"→ {request.method} {request.url.path} | Client: {request.client.host if request.client else 'unknown'}"
    )

    try:
        response = await call_next(request)
        duration = time.time() - start_time

        # Log response based on status
        if response.status_code >= 400:
            logger.warning(
                f"← {request.method} {request.url.path} | "
                f"Status: {response.status_code} | "
                f"Time: {duration:.3f}s"
            )
        else:
            logger.info(
                f"← {request.method} {request.url.path} | "
                f"Status: {response.status_code} | "
                f"Time: {duration:.3f}s"
            )

        return response

    except Exception as e:
        duration = time.time() - start_time
        logger.error(
            f"✗ {request.method} {request.url.path} | "
            f"Error: {str(e)} | "
            f"Time: {duration:.3f}s"
        )
        raise


@app.get("/")
async def root():
    return JSONResponse(
        content={
            "app": settings.APP_NAME,
            "status": "running",
            "docs": "http://127.0.0.1:8000/docs",
        }
    )


app.include_router(api_router, prefix="/api/v1")

if __name__ == "__main__":
    is_dev = settings.ENV_MODE == "dev"
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=is_dev,
        log_level="warning",  # Suppress uvicorn's INFO logs
    )
