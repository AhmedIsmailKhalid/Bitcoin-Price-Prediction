"""Model persistence and versioning management"""
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List

from src.shared.logging import get_logger
from src.ml_models.base_model import BaseMLModel


class ModelManager:
    """Manage model persistence, versioning, and metadata"""
    
    def __init__(self, models_dir: str = "models/saved_models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.logger = get_logger(__name__)
        
        # Model registry
        self.registry_file = self.models_dir / "model_registry.json"
        self._load_registry()
    
    def save_model(self, model: BaseMLModel, version: str = None, 
                   metadata: Dict[str, Any] = None) -> str:
        """Save model with versioning and metadata"""
        
        if not model.is_trained:
            raise ValueError("Cannot save untrained model")
        
        # Generate version if not provided
        if version is None:
            version = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create model directory
        model_dir = self.models_dir / model.model_name / version
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model file
        model_file = model_dir / "model.pkl"
        model.save_model(str(model_file))
        
        # Prepare metadata
        model_metadata = {
            "model_name": model.model_name,
            "version": version,
            "saved_at": datetime.now().isoformat(),
            "model_file": str(model_file),
            "feature_names": model.feature_names,
            "training_metrics": model.training_metrics,
            "validation_metrics": model.validation_metrics,
            "feature_count": len(model.feature_names),
            "is_trained": model.is_trained
        }
        
        # Add custom metadata
        if metadata:
            model_metadata.update(metadata)
        
        # Save metadata
        metadata_file = model_dir / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(model_metadata, f, indent=2)
        
        # Update registry
        self._update_registry(model.model_name, version, model_metadata)
        
        self.logger.info(f"Model saved: {model.model_name} v{version}")
        return version
    
    def load_model(self, model_name: str, version: str = "latest") -> BaseMLModel:
        """Load model by name and version"""
        
        if version == "latest":
            version = self.get_latest_version(model_name)
            if not version:
                raise ValueError(f"No models found for {model_name}")
        
        model_dir = self.models_dir / model_name / version
        if not model_dir.exists():
            raise FileNotFoundError(f"Model not found: {model_name} v{version}")
        
        # Load metadata
        metadata_file = model_dir / "metadata.json"
        with open(metadata_file, 'r') as f:
            _ = json.load(f)
        
        # Import the appropriate model class
        model_class = self._get_model_class(model_name)
        model = model_class()
        
        # Load the saved model
        model_file = model_dir / "model.pkl"
        model.load_model(str(model_file))
        
        self.logger.info(f"Model loaded: {model_name} v{version}")
        return model
    
    def list_models(self) -> Dict[str, List[str]]:
        """List all available models and versions"""
        return dict(self.registry)
    
    def get_model_metadata(self, model_name: str, version: str = "latest") -> Dict[str, Any]:
        """Get model metadata"""
        
        if version == "latest":
            version = self.get_latest_version(model_name)
        
        metadata_file = self.models_dir / model_name / version / "metadata.json"
        with open(metadata_file, 'r') as f:
            return json.load(f)
    
    def get_latest_version(self, model_name: str) -> Optional[str]:
        """Get latest version of a model"""
        if model_name not in self.registry:
            return None
        
        versions = self.registry[model_name]
        if not versions:
            return None
        
        # Sort versions by timestamp (assuming YYYYMMDD_HHMMSS format)
        return sorted(versions, reverse=True)[0]
    
    def delete_model(self, model_name: str, version: str) -> bool:
        """Delete a specific model version"""
        model_dir = self.models_dir / model_name / version
        
        if not model_dir.exists():
            return False
        
        # Remove directory
        import shutil
        shutil.rmtree(model_dir)
        
        # Update registry
        if model_name in self.registry:
            self.registry[model_name].remove(version)
            if not self.registry[model_name]:
                del self.registry[model_name]
        
        self._save_registry()
        self.logger.info(f"Model deleted: {model_name} v{version}")
        return True
    
    def _load_registry(self):
        """Load model registry"""
        if self.registry_file.exists():
            with open(self.registry_file, 'r') as f:
                self.registry = json.load(f)
        else:
            self.registry = {}
    
    def _save_registry(self):
        """Save model registry"""
        with open(self.registry_file, 'w') as f:
            json.dump(self.registry, f, indent=2)
    
    def _update_registry(self, model_name: str, version: str, metadata: Dict[str, Any]):
        """Update model registry"""
        if model_name not in self.registry:
            self.registry[model_name] = []
        
        if version not in self.registry[model_name]:
            self.registry[model_name].append(version)
        
        self._save_registry()
    
    def _get_model_class(self, model_name: str):
        """Get model class by name"""
        model_classes = {
            "LogisticRegression": "src.ml_models.classifiers.logistic_regression.LogisticRegressionModel",
            "RandomForest": "src.ml_models.classifiers.random_forest.RandomForestModel",
            "XGBoost": "src.ml_models.classifiers.xgboost_model.XGBoostModel",
            "GradientBoosting": "src.ml_models.classifiers.gradient_boosting.GradientBoostingModel",
            "VotingEnsemble": "src.ml_models.ensembles.voting_ensemble.VotingEnsemble"
        }
        
        if model_name not in model_classes:
            raise ValueError(f"Unknown model type: {model_name}")
        
        # Dynamic import
        module_path, class_name = model_classes[model_name].rsplit('.', 1)
        module = __import__(module_path, fromlist=[class_name])
        return getattr(module, class_name)