"""Deploy trained models to the serving infrastructure"""
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from src.shared.logging import get_logger, setup_logging
from src.ml_models.training_pipeline import MLTrainingPipeline
from src.ml_models.persistence.model_manager import ModelManager


def deploy_models():
    """Train and deploy all models to serving infrastructure"""
    setup_logging()
    logger = get_logger(__name__)
    
    try:
        logger.info("Starting model training and deployment...")
        
        # Train models
        pipeline = MLTrainingPipeline()
        X, y = pipeline.load_training_data()
        data = pipeline.prepare_data(X, y)
        
        # Train baseline models
        baseline_results = pipeline.train_baseline_models(data)
        
        # Train advanced models
        advanced_results = pipeline.train_advanced_models(data)
        
        # Combine all results
        all_results = {**baseline_results, **advanced_results}
        
        # Deploy models
        model_manager = ModelManager()
        deployed_count = 0
        
        for model_name, result in all_results.items():
            if 'error' not in result and model_name in pipeline.models:
                model = pipeline.models[model_name]
                
                try:
                    # Prepare deployment metadata
                    metadata = {
                        "deployment_date": "2025-09-17",
                        "data_samples": len(X),
                        "feature_count": len(X.columns),
                        "class_distribution": y.value_counts().to_dict(),
                        "training_results": result
                    }
                    
                    # Save model
                    version = model_manager.save_model(model, metadata=metadata)
                    logger.info(f"Deployed {model_name} v{version}")
                    deployed_count += 1
                    
                except Exception as e:
                    logger.error(f"Failed to deploy {model_name}: {e}")
        
        # Deployment summary
        logger.info("\n=== DEPLOYMENT SUMMARY ===")
        logger.info(f"Models trained: {len(all_results)}")
        logger.info(f"Models deployed: {deployed_count}")
        
        # List deployed models
        available_models = model_manager.list_models()
        logger.info(f"\nAvailable models: {list(available_models.keys())}")
        
        return deployed_count > 0
        
    except Exception as e:
        logger.error(f"Deployment failed: {e}")
        return False


if __name__ == "__main__":
    success = deploy_models()
    if success:
        print("\n✅ Model deployment successful!")
        print("Run: poetry run python -m src.api.main")
        print("API docs: http://localhost:8000/docs")
    else:
        print("\n❌ Model deployment failed!")
    
    sys.exit(0 if success else 1)