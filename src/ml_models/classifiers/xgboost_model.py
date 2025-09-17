"""XGBoost model for Bitcoin price prediction"""
import xgboost as xgb
from src.ml_models.base_model import BaseMLModel


class XGBoostModel(BaseMLModel):
    """XGBoost classifier for price direction prediction"""
    
    def __init__(self, **kwargs):
        super().__init__("XGBoost")
        self.model_params = kwargs
    
    def _create_model(self):
        """Create XGBoost model"""
        default_params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'max_depth': 3,  # Reduced for small dataset
            'learning_rate': 0.3,  # Higher for small dataset
            'n_estimators': 50,  # Reduced for small dataset
            'subsample': 1.0,  # Use all data
            'colsample_bytree': 1.0,  # Use all features
            'random_state': 42,
            'use_label_encoder': False,
            'base_score': 0.5  # Explicit base score
        }
        
        # Merge with user params
        params = {**default_params, **self.model_params}
        
        return xgb.XGBClassifier(**params)