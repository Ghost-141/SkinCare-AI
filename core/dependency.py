from core.db import SessionLocal, create_db_and_tables

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
