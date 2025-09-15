"""Test price feature extraction"""
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

import numpy as np

from src.data_processing.feature_engineering.price_features import PriceFeatureExtractor
from src.data_processing.feature_engineering.temporal_features import TemporalFeatureExtractor
from src.shared.logging import get_logger, setup_logging


def test_price_features():
    """Test price feature extraction pipeline"""
    setup_logging()
    logger = get_logger(__name__)
    
    try:
        # Initialize feature extractors
        price_extractor = PriceFeatureExtractor()
        temporal_extractor = TemporalFeatureExtractor()
        
        logger.info("Testing price feature extraction...")
        
        # Extract price features
        price_df = price_extractor.extract_all_features()
        
        if price_df.empty:
            logger.error("No price data available for feature extraction")
            return False
        
        logger.info(f"Extracted features from {len(price_df)} price records")
        logger.info(f"Price features created: {len(price_extractor.get_feature_names())}")
        
        # Show sample of price features
        price_features = price_extractor.get_feature_names()
        logger.info(f"Price feature names: {price_features[:10]}...")  # Show first 10
        
        # Extract temporal features
        temporal_df = temporal_extractor.extract_all_temporal_features(price_df.copy())
        
        temporal_features = temporal_extractor.get_feature_names()
        logger.info(f"Temporal features created: {len(temporal_features)}")
        logger.info(f"Temporal feature names: {temporal_features}")
        
        # Show sample data
        logger.info("\n=== Sample Feature Data ===")
        logger.info("Most recent record features:")
        
        # Get latest record
        latest_idx = temporal_df.index[-1]
        latest_record = temporal_df.iloc[latest_idx]
        
        # Show key features
        key_features = ['price', 'sma_5', 'ema_5', 'volatility_5', 'price_change_1', 'hour', 'day_of_week']
        
        for feature in key_features:
            if feature in temporal_df.columns:
                value = latest_record[feature]
                if isinstance(value, (int, float)) and not np.isnan(value):
                    logger.info(f"  {feature}: {value:.6f}")
                else:
                    logger.info(f"  {feature}: {value}")
        
        # Basic validation
        total_features = len(price_features) + len(temporal_features)
        logger.info(f"\nTotal features created: {total_features}")
        
        # Check for NaN values
        nan_counts = temporal_df.isnull().sum()
        features_with_nans = nan_counts[nan_counts > 0]
        
        if len(features_with_nans) > 0:
            logger.info(f"Features with NaN values: {len(features_with_nans)}")
            for feature, count in features_with_nans.head().items():
                logger.info(f"  {feature}: {count} NaN values")
        else:
            logger.info("No NaN values found in features")
        
        logger.info("\n✅ Price and temporal feature extraction working correctly!")
        return True
        
    except Exception as e:
        logger.error(f"Price feature extraction test failed: {e}")
        return False


if __name__ == "__main__":
    success = test_price_features()
    if success:
        print("\n✅ Price feature extraction working correctly!")
    else:
        print("\n❌ Price feature extraction test failed!")
    
    sys.exit(0 if success else 1)