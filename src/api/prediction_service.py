"""Core prediction service logic"""
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple

from src.shared.logging import get_logger
from src.ml_models.persistence.model_manager import ModelManager
from src.data_processing.validation.dataset_exporter import MLDatasetExporter


class PredictionService:
    """Core service for making predictions"""
    
    def __init__(self):
        self.model_manager = ModelManager()
        self.dataset_exporter = MLDatasetExporter()
        self.logger = get_logger(__name__)
        
        # Cache for loaded models
        self.model_cache = {}
    
    def predict(self, features: Dict[str, float], model_name: str = "RandomForest", 
                version: str = "latest") -> Tuple[int, float, str]:
        """Make prediction using specified model"""
        
        # Load model (with caching)
        model_key = f"{model_name}_{version}"
        if model_key not in self.model_cache:
            try:
                model = self.model_manager.load_model(model_name, version)
                self.model_cache[model_key] = model
                self.logger.info(f"Loaded model into cache: {model_key}")
            except Exception as e:
                self.logger.error(f"Failed to load model {model_key}: {e}")
                raise ValueError(f"Model not available: {model_name} v{version}")
        
        model = self.model_cache[model_key]
        
        # Prepare features
        feature_df = self._prepare_features(features, model.feature_names)
        
        # Make prediction
        try:
            prediction = int(model.predict(feature_df)[0])
            
            # Get prediction probability if available
            try:
                probabilities = model.predict_proba(feature_df)[0]
                probability = float(probabilities[prediction])
                confidence = self._calculate_confidence(probabilities)
            except:  # noqa: E722
                probability = None
                confidence = "unknown"
            
            self.logger.info(f"Prediction made: {prediction} with {model_name}")
            return prediction, probability, confidence
            
        except Exception as e:
            self.logger.error(f"Prediction failed: {e}")
            raise ValueError(f"Prediction failed: {str(e)}")
    
    def get_available_models(self) -> Dict[str, Any]:
        """Get list of available models"""
        models = self.model_manager.list_models()
        
        model_info = {}
        for model_name, versions in models.items():
            if versions:
                latest_version = max(versions)
                try:
                    metadata = self.model_manager.get_model_metadata(model_name, latest_version)
                    model_info[model_name] = {
                        "latest_version": latest_version,
                        "all_versions": versions,
                        "feature_count": metadata.get("feature_count", 0),
                        "training_accuracy": metadata.get("training_metrics", {}).get("accuracy"),
                        "saved_at": metadata.get("saved_at"),
                        "is_available": True
                    }
                except Exception as e:
                    model_info[model_name] = {
                        "latest_version": latest_version,
                        "all_versions": versions,
                        "is_available": False,
                        "error": str(e)
                    }
        
        return model_info
    
    def validate_features(self, features: Dict[str, float], model_name: str, 
                         version: str = "latest") -> Tuple[bool, str]:
        """Validate input features for a model"""
        try:
            # Get model metadata
            metadata = self.model_manager.get_model_metadata(model_name, version)
            required_features = metadata.get("feature_names", [])
            
            # Check if all required features are present
            missing_features = set(required_features) - set(features.keys())
            if missing_features:
                return False, f"Missing features: {list(missing_features)}"
            
            # Check for extra features
            extra_features = set(features.keys()) - set(required_features)
            if extra_features:
                self.logger.warning(f"Extra features provided: {list(extra_features)}")
            
            # Validate feature values
            for feature, value in features.items():
                if not isinstance(value, (int, float)) or np.isnan(value) or np.isinf(value):
                    return False, f"Invalid value for feature {feature}: {value}"
            
            return True, "Features valid"
            
        except Exception as e:
            return False, f"Validation error: {str(e)}"
    
    def _prepare_features(self, features: Dict[str, float], required_features: list) -> pd.DataFrame:
        """Prepare features for model input"""
        
        # Create DataFrame with required features in correct order
        feature_data = {}
        for feature_name in required_features:
            if feature_name in features:
                feature_data[feature_name] = [features[feature_name]]
            else:
                # Use median value for missing features (basic imputation)
                feature_data[feature_name] = [0.0]
                self.logger.warning(f"Missing feature {feature_name}, using default value")
        
        return pd.DataFrame(feature_data)
    
    def _calculate_confidence(self, probabilities: np.ndarray) -> str:
        """Calculate confidence level based on prediction probabilities"""
        max_prob = max(probabilities)
        
        if max_prob >= 0.8:
            return "high"
        elif max_prob >= 0.6:
            return "medium"
        else:
            return "low"