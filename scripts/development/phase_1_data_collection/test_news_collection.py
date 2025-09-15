"""Test news data collection"""
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.shared.database import SessionLocal
from src.shared.logging import get_logger, setup_logging
from src.shared.models import CollectionMetadata, NewsData
from src.data_engineering.collectors.news_collector import MultiSourceNewsCollector


def test_news_collection():
    """Test the complete news collection workflow"""
    setup_logging()
    logger = get_logger(__name__)
    
    logger.info("Testing CoinDesk RSS news collection...")
    
    # Create collector
    collector = MultiSourceNewsCollector()
    
    # Test connection first
    logger.info("Testing connection to CoinDesk RSS feed...")
    if not collector.test_connection():
        logger.error("❌ Connection test failed")
        return False
    
    # Run collection
    success = collector.run_collection()
    
    if success:
        # Verify data was stored
        db = SessionLocal()
        try:
            # Check news data
            news_count = db.query(NewsData).count()
            logger.info(f"Total news articles in database: {news_count}")
            
            # Check recent articles
            recent_news = db.query(NewsData).order_by(NewsData.collected_at.desc()).limit(3).all()
            logger.info("Recent news articles:")
            for article in recent_news:
                logger.info(f"  Title: {article.title}")
                logger.info(f"  Source: {article.data_source}")
                logger.info(f"  Word count: {article.word_count}")
                logger.info(f"  Published: {article.published_at}")
                logger.info("  ---")
            
            # Check metadata
            recent_metadata = db.query(CollectionMetadata).filter(
                CollectionMetadata.collector_name == "CoinDeskRSS"
            ).order_by(CollectionMetadata.start_time.desc()).first()
            
            if recent_metadata:
                logger.info(f"Latest collection: {recent_metadata.collector_name} - {recent_metadata.status} - {recent_metadata.records_collected} articles")
            
        finally:
            db.close()
    
    return success


if __name__ == "__main__":
    success = test_news_collection()
    if success:
        print("\n✅ News collection test completed successfully!")
    else:
        print("\n❌ News collection test failed!")
    
    sys.exit(0 if success else 1)