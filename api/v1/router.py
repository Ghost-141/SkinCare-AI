from fastapi import APIRouter
from api.v1.endpoints import skin_analysis, health

api_router = APIRouter()
api_router.include_router(skin_analysis.router, tags=["Skin Analysis"])
api_router.include_router(health.router, tags=["System Health"])
