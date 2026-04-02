from core.db import SessionLocal, create_db_and_tables
from services.skin_service import SkinService
from services.advisor_service import AdvisorService

_skin_service = SkinService()
_advisor_service = AdvisorService()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_skin_service() -> SkinService:
    """Return the shared SkinService instance."""
    return _skin_service


def get_advisor_service() -> AdvisorService:
    """Return the shared AdvisorService instance."""
    return _advisor_service
