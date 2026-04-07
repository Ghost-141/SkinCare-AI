from core.db import AsyncSessionLocal, create_db_and_tables
from services.skin_service import SkinService
from services.advisor_service import AdvisorService
from sqlalchemy.ext.asyncio import AsyncSession

_skin_service = SkinService()
_advisor_service = AdvisorService()


async def get_db() -> AsyncSession:
    async with AsyncSessionLocal() as db:
        try:
            yield db
        finally:
            await db.close()


def get_skin_service() -> SkinService:
    """Return the shared SkinService instance."""
    return _skin_service


def get_advisor_service() -> AdvisorService:
    """Return the shared AdvisorService instance."""
    return _advisor_service
