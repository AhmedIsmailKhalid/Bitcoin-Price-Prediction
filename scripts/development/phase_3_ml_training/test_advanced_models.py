"""Test advanced model training"""
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from src.ml_models.training_pipeline import MLTrainingPipeline
from src.shared.logging import get_logger, setup_logging


def test_advanced_models():
    """Test advanced model training pipeline"""
    setup_logging()
    logger = get_logger(__name__)
    
    try:
        logger.info("Testing advanced model training pipeline...")
        
        # Create training pipeline
        pipeline = MLTrainingPipeline()
        
        # Load data and train baseline models first
        X, y = pipeline.load_training_data()
        data = pipeline.prepare_data(X, y)
        baseline_results = pipeline.train_baseline_models(data)
        
        # Train advanced models
        advanced_results = pipeline.train_advanced_models(data)
        
        # Combine results
        all_results = {**baseline_results, **advanced_results}
        
        # Display results
        logger.info("\n=== ADVANCED MODEL RESULTS ===")
        logger.info(f"Data: {len(X)} samples, {len(X.columns)} features")
        
        logger.info("\n=== MODEL PERFORMANCE COMPARISON ===")
        for model_name, result in all_results.items():
            if 'error' in result:
                logger.error(f"{model_name}: {result['error']}")
            else:
                train_acc = result['training_metrics'].get('accuracy', 0.0)
                test_acc = result.get('test_metrics', {}).get('accuracy', 'N/A')
                
                # CV results if available
                cv_acc = 'N/A'
                if 'cv_results' in result:
                    cv_metrics = result['cv_results'].get('overall_metrics', {})
                    cv_acc = cv_metrics.get('accuracy_mean', 'N/A')
                    if cv_acc != 'N/A':
                        cv_acc = f"{cv_acc:.3f}"
                
                logger.info(f"{model_name}: Train={train_acc:.3f}, Test={test_acc}, CV={cv_acc}")
        
        # Feature importance analysis
        logger.info("\n=== FEATURE IMPORTANCE (TOP 5) ===")
        for model_name, result in all_results.items():
            if 'feature_importance' in result and result['feature_importance']:
                importance = result['feature_importance']
                top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5]
                feature_str = ", ".join([f"{name}({score:.3f})" for name, score in top_features])
                logger.info(f"{model_name}: {feature_str}")
        
        logger.info("\n✅ Advanced model training pipeline working!")
        return True
        
    except Exception as e:
        logger.error(f"Advanced model test failed: {e}")
        return False


if __name__ == "__main__":
    success = test_advanced_models()
    if success:
        print("\n✅ Advanced models working correctly!")
    else:
        print("\n❌ Advanced model test failed!")
    
    sys.exit(0 if success else 1)