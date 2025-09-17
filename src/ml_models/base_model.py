"""Base model class for all ML implementations"""
import os
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from src.shared.logging import get_logger
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class BaseMLModel(ABC):
    """Abstract base class for all ML models"""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = None
        self.is_trained = False
        self.feature_names = []
        self.logger = get_logger(__name__)
        
        # Performance tracking
        self.training_metrics = {}
        self.validation_metrics = {}
        
    @abstractmethod
    def _create_model(self) -> Any:
        """Create the underlying model instance"""
        pass
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, 
              X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None) -> Dict[str, float]:
        """Train the model"""
        self.logger.info(f"Training {self.model_name} model...")
        
        # Store feature names
        self.feature_names = list(X_train.columns)
        
        # Create model if not exists
        if self.model is None:
            self.model = self._create_model()
        
        # Fit the model
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # Calculate training metrics
        y_train_pred = self.model.predict(X_train)
        self.training_metrics = self._calculate_metrics(y_train, y_train_pred)
        
        # Calculate validation metrics if validation set provided
        if X_val is not None and y_val is not None:
            y_val_pred = self.model.predict(X_val)
            self.validation_metrics = self._calculate_metrics(y_val, y_val_pred)
        
        self.logger.info(f"{self.model_name} training completed")
        return self.training_metrics
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions"""
        if not self.is_trained:
            raise ValueError(f"Model {self.model_name} must be trained before making predictions")
        
        return self.model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get prediction probabilities"""
        if not self.is_trained:
            raise ValueError(f"Model {self.model_name} must be trained before making predictions")
        
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        else:
            raise NotImplementedError(f"Model {self.model_name} does not support probability predictions")
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """Evaluate model performance"""
        y_pred = self.predict(X_test)
        return self._calculate_metrics(y_test, y_pred)
    
    def cross_validate(self, X: pd.DataFrame, y: pd.Series, cv_folds: int = 5) -> Dict[str, float]:
        """Perform time-series cross-validation"""
        if not self.is_trained:
            raise ValueError(f"Model {self.model_name} must be trained before cross-validation")
        
        # Use TimeSeriesSplit for temporal data
        tscv = TimeSeriesSplit(n_splits=cv_folds)
        
        # Calculate cross-validation scores
        cv_scores = cross_val_score(self.model, X, y, cv=tscv, scoring='accuracy')
        
        return {
            'cv_accuracy_mean': cv_scores.mean(),
            'cv_accuracy_std': cv_scores.std(),
            'cv_scores': cv_scores.tolist()
        }
    
    def _calculate_metrics(self, y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate classification metrics"""
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1_score': f1_score(y_true, y_pred, average='weighted', zero_division=0)
        }
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get feature importance if available"""
        if not self.is_trained:
            return None
        
        if hasattr(self.model, 'feature_importances_'):
            importance_scores = self.model.feature_importances_
            return dict(zip(self.feature_names, importance_scores))
        elif hasattr(self.model, 'coef_'):
            # For linear models, use absolute coefficients
            coef_scores = np.abs(self.model.coef_[0] if len(self.model.coef_.shape) > 1 else self.model.coef_)
            return dict(zip(self.feature_names, coef_scores))
        else:
            return None
    
    def save_model(self, filepath: str) -> str:
        """Save trained model to disk"""
        if not self.is_trained:
            raise ValueError(f"Model {self.model_name} must be trained before saving")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save model with metadata
        model_data = {
            'model': self.model,
            'model_name': self.model_name,
            'feature_names': self.feature_names,
            'training_metrics': self.training_metrics,
            'validation_metrics': self.validation_metrics,
            'trained_at': datetime.now().isoformat()
        }
        
        joblib.dump(model_data, filepath)
        self.logger.info(f"Model saved to {filepath}")
        return filepath
    
    def load_model(self, filepath: str):
        """Load trained model from disk"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.model_name = model_data['model_name']
        self.feature_names = model_data['feature_names']
        self.training_metrics = model_data.get('training_metrics', {})
        self.validation_metrics = model_data.get('validation_metrics', {})
        self.is_trained = True
        
        self.logger.info(f"Model loaded from {filepath}")
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Get model summary information"""
        return {
            'model_name': self.model_name,
            'is_trained': self.is_trained,
            'feature_count': len(self.feature_names),
            'training_metrics': self.training_metrics,
            'validation_metrics': self.validation_metrics
        }