"""Random Forest model for Bitcoin price prediction"""
from src.ml_models.base_model import BaseMLModel
from sklearn.ensemble import RandomForestClassifier


class RandomForestModel(BaseMLModel):
    """Random Forest classifier for price direction prediction"""
    
    def __init__(self, **kwargs):
        super().__init__("RandomForest")
        self.model_params = kwargs
    
    def _create_model(self):
        """Create random forest model"""
        default_params = {
            'n_estimators': 100,
            'random_state': 42,
            'max_depth': 10,
            'min_samples_split': 5,
            'min_samples_leaf': 2
        }
        
        # Merge with user params
        params = {**default_params, **self.model_params}
        
        return RandomForestClassifier(**params)