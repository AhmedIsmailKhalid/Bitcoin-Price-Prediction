import logging
from .config import settings
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine, MetaData, text
from sqlalchemy.ext.declarative import declarative_base

logger = logging.getLogger(__name__)

# Create database engine
engine = create_engine(
    str(settings.database_url),
    echo=settings.debug,  # Log SQL queries in debug mode
    pool_pre_ping=True,   # Verify connections before use
)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create base class for models
Base = declarative_base()

# Metadata for schema management
metadata = MetaData()


def get_db():
    """Dependency for getting database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def test_connection():
    """Test database connection"""
    try:
        with engine.connect() as connection:
            _ = connection.execute(text("SELECT 1"))  # Changed this line
            logger.info("Database connection successful")
            return True
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        return False