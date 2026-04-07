from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
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
    # Ensure database directory exists
    if "sqlite" in settings.DATABASE_URL:
        # Handle both sqlite:/// and sqlite+aiosqlite:///
        db_path = settings.DATABASE_URL.split(":///")[-1]
        db_dir = os.path.dirname(db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)

    # Ensure upload directory exists
    try:
        os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    except PermissionError:
        fallback_path = os.path.join(Path.home(), ".skincare_ai", "uploads")
        os.makedirs(fallback_path, exist_ok=True)
        settings.UPLOAD_DIR = fallback_path
        logger.warning(
            f"Upload directory not accessible. Using fallback: {fallback_path}"
        )

    # Initialize tables
    await create_db_and_tables()
    
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


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.warning(
        f"Validation Error | "
        f"Path: {request.url.path} | "
        f"Errors: {exc.errors()} | "
        f"Body: {exc.body}"
    )
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors(), "body": exc.body},
    )


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    if exc.status_code >= 500:
        logger.error(
            f"Server Error {exc.status_code} | "
            f"Path: {request.url.path} | "
            f"Detail: {exc.detail}"
        )
    else:
        logger.warning(
            f"HTTP Error {exc.status_code} | "
            f"Path: {request.url.path} | "
            f"Detail: {exc.detail}"
        )
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
    )


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
