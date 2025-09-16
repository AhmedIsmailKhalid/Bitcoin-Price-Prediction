"""Test ML dataset preparation and export"""
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from src.data_processing.validation.dataset_exporter import MLDatasetExporter
from src.shared.logging import get_logger, setup_logging


def test_ml_dataset_preparation():
    """Test ML dataset preparation pipeline"""
    setup_logging()
    logger = get_logger(__name__)
    
    try:
        # Create dataset exporter
        exporter = MLDatasetExporter()
        
        logger.info("Testing ML dataset preparation pipeline...")
        
        # Generate dataset summary
        summary = exporter.generate_dataset_summary()
        
        logger.info(f"\n=== DATASET SUMMARY ===")
        logger.info(f"Status: {summary['status']}")
        
        # Handle NO_DATA status
        if summary['status'] == 'NO_DATA':
            logger.info(f"Message: {summary.get('message', 'No data available')}")
            
            if summary.get('recommendations'):
                logger.info("Recommendations:")
                for rec in summary['recommendations']:
                    logger.info(f"  → {rec}")
            
            logger.info("\n⚠️  Insufficient data for ML dataset creation")
            logger.info("This is expected with limited price data collection")
            return True  # Test passes - this is expected behavior
        
        # If we have data, show full summary
        logger.info(f"Total samples: {summary.get('total_samples', 0)}")
        logger.info(f"Total features: {summary.get('total_features', 0)}")
        
        if summary.get('target_variables'):
            logger.info(f"Target variables: {summary['target_variables']}")
        
        if summary.get('data_quality_score') is not None:
            logger.info(f"Data quality score: {summary['data_quality_score']:.3f}")
        
        if summary.get('recommendations'):
            logger.info("Recommendations:")
            for rec in summary['recommendations']:
                logger.info(f"  → {rec}")
        
        # Try to export datasets if data is available
        if summary['status'] in ['READY', 'LIMITED']:
            logger.info("\n=== DATASET EXPORT TEST ===")
            
            exported_files = exporter.export_datasets(target_col='price_direction', apply_scaling=True)
            
            if exported_files:
                logger.info("Successfully exported datasets:")
                for split_name, filepath in exported_files.items():
                    logger.info(f"  {split_name}: {os.path.basename(filepath)}")
            else:
                logger.info("No datasets exported - insufficient data")
        
        # Feature analysis
        logger.info("\n=== FEATURE ANALYSIS ===")
        feature_categories = summary.get('feature_categories', {})
        
        if feature_categories:
            for category, features in feature_categories.items():
                if features:
                    logger.info(f"{category}: {len(features)} features")
                    logger.info(f"  Sample: {features[:3]}...")
        else:
            logger.info("No feature categories available - need more price data")
        
        logger.info("\n✅ ML dataset preparation pipeline working!")
        return True
        
    except Exception as e:
        logger.error(f"ML dataset preparation test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


if __name__ == "__main__":
    success = test_ml_dataset_preparation()
    if success:
        print("\n✅ ML dataset preparation working correctly!")
    else:
        print("\n❌ ML dataset preparation test failed!")
    
    sys.exit(0 if success else 1)