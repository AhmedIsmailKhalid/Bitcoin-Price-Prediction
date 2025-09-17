"""Gradient Boosting model for Bitcoin price prediction"""
from sklearn.ensemble import GradientBoostingClassifier
from src.ml_models.base_model import BaseMLModel


class GradientBoostingModel(BaseMLModel):
    """Gradient Boosting classifier for price direction prediction"""
    
    def __init__(self, **kwargs):
        super().__init__("GradientBoosting")
        self.model_params = kwargs
    
    def _create_model(self):
        """Create gradient boosting model"""
        default_params = {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 6,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'subsample': 0.8,
            'random_state': 42
        }
        
        # Merge with user params
        params = {**default_params, **self.model_params}
        
        return GradientBoostingClassifier(**params)