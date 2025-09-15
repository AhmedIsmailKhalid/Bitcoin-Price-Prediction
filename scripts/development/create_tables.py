"""Create database tables"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from sqlalchemy import text  

from src.shared.database import engine, Base
from src.shared.logging import setup_logging, get_logger
from src.shared.models import PriceData, CollectionMetadata, NewsData, SentimentData



def create_tables():
    """Create all database tables"""
    setup_logging()
    logger = get_logger(__name__)
    
    try:
        logger.info("Creating database tables...")
        Base.metadata.create_all(bind=engine)
        logger.info("✅ Database tables created successfully")
        
        # Test table creation by checking if tables exist
        with engine.connect() as conn:
            result = conn.execute(text(
                "SELECT table_name FROM information_schema.tables WHERE table_schema='public'"
            ))
            tables = [row[0] for row in result]
            logger.info(f"Created tables: {tables}")
            
        return True
    except Exception as e:
        logger.error(f"❌ Failed to create tables: {e}")
        return False


if __name__ == "__main__":
    success = create_tables()
    sys.exit(0 if success else 1)