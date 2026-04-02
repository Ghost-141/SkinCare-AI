from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from core.config import settings
from models.db_models import Base
from utils.logger import logger

# Create the SQLAlchemy engine
engine = create_engine(settings.DATABASE_URL, connect_args={"check_same_thread": False})

# Create a session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def create_db_and_tables():
    """Load and initialize database tables."""
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables initialized successfully.")
    except Exception as e:
        logger.error(f"Error initializing database: {str(e)}")
        raise

def check_db_status():
    """Check the database connection status."""
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        return "online"
    except Exception as e:
        logger.error(f"Database status check failed: {str(e)}")
        return f"offline: {str(e)}"

def commit_to_db(db_session, model_instance):
    """Commit an instance to the database safely."""
    try:
        db_session.add(model_instance)
        db_session.commit()
        db_session.refresh(model_instance)
        return model_instance
    except Exception as e:
        db_session.rollback()
        logger.error(f"Database commit error: {str(e)}")
        raise
