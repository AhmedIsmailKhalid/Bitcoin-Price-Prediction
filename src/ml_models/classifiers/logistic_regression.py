"""Logistic Regression model for Bitcoin price prediction"""
from src.ml_models.base_model import BaseMLModel
from sklearn.linear_model import LogisticRegression


class LogisticRegressionModel(BaseMLModel):
    """Logistic Regression classifier for price direction prediction"""
    
    def __init__(self, **kwargs):
        super().__init__("LogisticRegression")
        self.model_params = kwargs
    
    def _create_model(self):
        """Create logistic regression model"""
        default_params = {
            'random_state': 42,
            'max_iter': 1000,
            'solver': 'liblinear'
        }
        
        # Merge with user params
        params = {**default_params, **self.model_params}
        
        return LogisticRegression(**params)