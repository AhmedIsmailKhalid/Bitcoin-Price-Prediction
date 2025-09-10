"""Test basic setup and connections"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))


from src.shared.config import settings
from src.shared.database import test_connection
from src.shared.logging import setup_logging, get_logger


def main():
    # Setup logging
    setup_logging()
    logger = get_logger(__name__)
    
    logger.info("Testing basic setup...")
    
    # Test configuration loading
    logger.info(f"Environment: {settings.environment}")
    logger.info(f"Database URL configured: {'Yes' if settings.database_url else 'No'}")
    logger.info(f"Reddit client ID configured: {'Yes' if settings.reddit_client_id else 'No'}")
    
    # Test database connection
    if test_connection():
        logger.info("✅ Database connection successful")
    else:
        logger.error("❌ Database connection failed")
        return False
    
    logger.info("✅ Basic setup test completed successfully")
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)