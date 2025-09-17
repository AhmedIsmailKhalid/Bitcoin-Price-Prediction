"""Test baseline model training"""
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from src.ml_models.training_pipeline import MLTrainingPipeline
from src.shared.logging import get_logger, setup_logging


def test_baseline_models():
    """Test baseline model training pipeline"""
    setup_logging()
    logger = get_logger(__name__)
    
    try:
        logger.info("Testing baseline model training pipeline...")
        
        # Create training pipeline
        pipeline = MLTrainingPipeline()
        
        # Run complete pipeline
        results = pipeline.run_complete_pipeline()
        
        if 'error' in results:
            logger.error(f"Pipeline failed: {results['error']}")
            return False
        
        # Display results
        logger.info("\n=== TRAINING RESULTS ===")
        logger.info(f"Data: {results['data_info']['total_samples']} samples, {results['data_info']['feature_count']} features")
        logger.info(f"Target distribution: {results['data_info']['target_distribution']}")
        
        logger.info("\n=== MODEL PERFORMANCE ===")
        for model_name, result in results['model_results'].items():
            if 'error' in result:
                logger.error(f"{model_name}: {result['error']}")
            else:
                train_acc = result['training_metrics'].get('accuracy', 0.0)
                test_acc = result.get('test_metrics', {}).get('accuracy', 'N/A')
                logger.info(f"{model_name}: Train Acc={train_acc:.3f}, Test Acc={test_acc}")
        
        # Summary
        summary = results['summary']
        logger.info("\n=== SUMMARY ===")
        logger.info(f"Best model: {summary['best_model']} (accuracy: {summary['best_accuracy']:.3f})")
        logger.info(f"Models trained: {summary['models_trained']}")
        
        logger.info("\n✅ Baseline model training pipeline working!")
        return True
        
    except Exception as e:
        logger.error(f"Baseline model test failed: {e}")
        return False


if __name__ == "__main__":
    success = test_baseline_models()
    if success:
        print("\n✅ Baseline models working correctly!")
    else:
        print("\n❌ Baseline model test failed!")
    
    sys.exit(0 if success else 1)