"""Test price data collection"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.shared.logging import setup_logging, get_logger
from src.data_engineering.collectors.price_collector import CoinGeckoCollector
from src.shared.database import SessionLocal
from src.shared.models import PriceData, CollectionMetadata
from sqlalchemy import text


def test_price_collection():
    """Test the complete price collection workflow"""
    setup_logging()
    logger = get_logger(__name__)
    
    logger.info("Testing CoinGecko price collection...")
    
    # Create collector
    collector = CoinGeckoCollector()
    
    # Test connection first
    if not collector.test_connection():
        logger.error("❌ Connection test failed")
        return False
    
    # Run collection
    success = collector.run_collection()
    
    if success:
        # Verify data was stored
        db = SessionLocal()
        try:
            # Check price data
            price_count = db.query(PriceData).count()
            logger.info(f"Total price records in database: {price_count}")
            
            # Check recent data
            recent_prices = db.query(PriceData).order_by(PriceData.collected_at.desc()).limit(5).all()
            logger.info("Recent price data:")
            for price in recent_prices:
                logger.info(f"  {price.symbol}: ${price.price_usd} (24h: {price.change_24h}%)")
            
            # Check metadata
            metadata_count = db.query(CollectionMetadata).count()
            logger.info(f"Collection metadata records: {metadata_count}")
            
            recent_metadata = db.query(CollectionMetadata).order_by(CollectionMetadata.start_time.desc()).first()
            if recent_metadata:
                logger.info(f"Latest collection: {recent_metadata.collector_name} - {recent_metadata.status} - {recent_metadata.records_collected} records")
            
        finally:
            db.close()
    
    return success


if __name__ == "__main__":
    success = test_price_collection()
    if success:
        print("\n✅ Price collection test completed successfully!")
    else:
        print("\n❌ Price collection test failed!")
    
    sys.exit(0 if success else 1)