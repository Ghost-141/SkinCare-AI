from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy import text
from core.config import settings
from models.db_models import Base
from core.logger import logger


# --- ADD THIS LINE ---
logger.info(f"Effective DATABASE_URL: {settings.DATABASE_URL}")


# Create the asynchronous SQLAlchemy engine
engine = create_async_engine(settings.DATABASE_URL)

# Create an async session factory
AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    autocommit=False,
    autoflush=False,
    expire_on_commit=False,
)


async def create_db_and_tables():
    """Load and initialize database tables asynchronously."""
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("Database tables initialized successfully.")
    except Exception as e:
        logger.error(f"Error initializing database: {str(e)}")
        raise


async def check_db_status():
    """Check the database connection status asynchronously."""
    try:
        async with engine.connect() as connection:
            await connection.execute(text("SELECT 1"))
        return "online"
    except Exception as e:
        logger.error(f"Database status check failed: {str(e)}")
        return f"offline: {str(e)}"


async def commit_to_db(db_session: AsyncSession, model_instance):
    """Commit an instance to the database safely using AsyncSession."""
    try:
        db_session.add(model_instance)
        await db_session.commit()
        await db_session.refresh(model_instance)
        return model_instance
    except Exception as e:
        await db_session.rollback()
        logger.error(f"Database commit error: {str(e)}")
        raise
