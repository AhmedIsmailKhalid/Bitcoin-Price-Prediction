"""Validate data quality for ML pipeline"""
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from src.shared.logging import get_logger, setup_logging
from src.data_processing.validation.data_validator import DataQualityValidator


def validate_data_quality():
    """Run comprehensive data quality validation"""
    setup_logging()
    logger = get_logger(__name__)
    
    try:
        validator = DataQualityValidator()
        
        logger.info("Running comprehensive data quality validation...")
        
        # Generate report
        report = validator.generate_comprehensive_report()
        
        logger.info("\n=== DATA QUALITY REPORT ===")
        logger.info(f"Overall Status: {report['overall_status']}")
        logger.info(f"Generated: {report['timestamp']}")
        
        # Price data validation
        price_val = report['price_validation']
        logger.info("\n=== PRICE DATA ===")
        logger.info(f"Records: {price_val['total_records']}")
        
        if price_val['issues']:
            logger.info("Issues:")
            for issue in price_val['issues']:
                logger.info(f"  - {issue}")
        
        if price_val['recommendations']:
            logger.info("Recommendations:")
            for rec in price_val['recommendations']:
                logger.info(f"  → {rec}")
        
        # News data validation
        news_val = report['news_validation']
        logger.info("\n=== NEWS DATA ===")
        logger.info(f"Records: {news_val['total_records']}")
        
        if news_val['issues']:
            logger.info("Issues:")
            for issue in news_val['issues']:
                logger.info(f"  - {issue}")
        
        # Sentiment data validation
        sentiment_val = report['sentiment_validation']
        logger.info("\n=== SENTIMENT DATA ===")
        logger.info(f"Records: {sentiment_val['total_records']}")
        
        if sentiment_val['issues']:
            logger.info("Issues:")
            for issue in sentiment_val['issues']:
                logger.info(f"  - {issue}")
        
        # Alignment validation
        align_val = report['alignment_validation']
        logger.info("\n=== DATA ALIGNMENT ===")
        logger.info(f"Price: {align_val['price_records']} | News: {align_val['news_records']} | Sentiment: {align_val['sentiment_records']}")
        
        if align_val['issues']:
            logger.info("Issues:")
            for issue in align_val['issues']:
                logger.info(f"  - {issue}")
        
        if align_val['recommendations']:
            logger.info("Recommendations:")
            for rec in align_val['recommendations']:
                logger.info(f"  → {rec}")
        
        logger.info("\n✅ Data quality validation complete!")
        return True
        
    except Exception as e:
        logger.error(f"Data quality validation failed: {e}")
        return False


if __name__ == "__main__":
    success = validate_data_quality()
    if success:
        print("\n✅ Data quality validation complete!")
    else:
        print("\n❌ Data quality validation failed!")
    
    sys.exit(0 if success else 1)