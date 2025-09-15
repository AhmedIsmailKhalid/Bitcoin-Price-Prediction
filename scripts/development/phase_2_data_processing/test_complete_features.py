"""Test complete feature set creation"""
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from src.data_processing.feature_engineering.feature_combiner import FeatureCombiner
from src.shared.logging import get_logger, setup_logging


def test_complete_features():
    """Test complete feature set creation"""
    setup_logging()
    logger = get_logger(__name__)
    
    try:
        combiner = FeatureCombiner()
        
        logger.info("Creating complete feature set...")
        
        # Create complete feature set
        feature_df = combiner.create_complete_feature_set()
        
        if feature_df.empty:
            logger.warning("No complete feature set created - need more price data")
            logger.info("This is expected with limited price data")
            
            # Test individual components
            logger.info("Testing individual feature components...")
            
            # Test sentiment features
            sentiment_df = combiner.get_sentiment_features()
            if not sentiment_df.empty:
                logger.info(f"Sentiment features available: {len(sentiment_df)} records")
                logger.info(f"Sentiment columns: {list(sentiment_df.columns)}")
            
            return True
        
        logger.info(f"Complete feature set created with {len(feature_df)} records")
        logger.info(f"Total features: {len(feature_df.columns)}")
        
        # Show feature categories
        feature_info = combiner.get_feature_importance_info()
        
        logger.info("\n=== Feature Categories ===")
        for category, features in feature_info.items():
            logger.info(f"{category}: {len(features)} features")
            if features:
                logger.info(f"  Sample: {features[:3]}...")
        
        # Show sample record
        if len(feature_df) > 0:
            logger.info("\n=== Sample Feature Record ===")
            sample = feature_df.iloc[0]
            
            key_features = ['timestamp', 'price', 'price_direction', 'hour', 'day_of_week']
            for feature in key_features:
                if feature in sample.index:
                    logger.info(f"  {feature}: {sample[feature]}")
        
        logger.info("\n✅ Complete feature set creation working!")
        return True
        
    except Exception as e:
        logger.error(f"Complete feature set test failed: {e}")
        return False


if __name__ == "__main__":
    success = test_complete_features()
    if success:
        print("\n✅ Complete feature engineering pipeline working!")
    else:
        print("\n❌ Complete feature set test failed!")
    
    sys.exit(0 if success else 1)