"""Time-series aware cross-validation"""
import pandas as pd
import numpy as np
from typing import Dict, Any
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from src.shared.logging import get_logger


class TimeSeriesValidator:
    """Time-series cross-validation for financial data"""
    
    def __init__(self, n_splits: int = 3, test_size: int = None):
        self.n_splits = n_splits
        self.test_size = test_size
        self.logger = get_logger(__name__)
    
    def validate_model(self, model, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Perform time-series cross-validation"""
        
        # Check class distribution
        class_counts = y.value_counts()
        min_class_count = class_counts.min()
        
        if len(X) < self.n_splits + 1 or min_class_count < 2:
            self.logger.warning("Skipping CV due to insufficient data or extreme class imbalance")
            return {
                'overall_metrics': {'validation_type': 'skipped', 'reason': 'insufficient_data'},
                'fold_results': []
            }
        
        # Create time series splits
        tscv = TimeSeriesSplit(n_splits=self.n_splits, test_size=self.test_size)
        
        fold_results = []
        
        for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
            self.logger.info(f"Evaluating fold {fold + 1}/{self.n_splits}")
            
            # Split data
            X_train_fold = X.iloc[train_idx]
            X_test_fold = X.iloc[test_idx]
            y_train_fold = y.iloc[train_idx]
            y_test_fold = y.iloc[test_idx]
            
            # Train model on fold
            model_copy = self._copy_model(model)
            model_copy.train(X_train_fold, y_train_fold)
            
            # Evaluate on test fold
            y_pred = model_copy.predict(X_test_fold)
            
            # Calculate metrics
            fold_metrics = self._calculate_fold_metrics(y_test_fold, y_pred)
            fold_metrics['fold'] = fold + 1
            fold_metrics['train_size'] = len(X_train_fold)
            fold_metrics['test_size'] = len(X_test_fold)
            
            fold_results.append(fold_metrics)
        
        # Aggregate results
        return self._aggregate_cv_results(fold_results)
    
    def _simple_validation(self, model, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Simple validation for insufficient data"""
        
        if len(X) < 3:
            # Use all data for training and testing (leave-one-out style)
            y_pred = model.predict(X)
            metrics = self._calculate_fold_metrics(y, y_pred)
            metrics['validation_type'] = 'leave_one_out'
            return {'overall_metrics': metrics, 'fold_results': [metrics]}
        
        # Use last 1/3 as test set
        split_idx = int(len(X) * 0.67)
        X_train = X.iloc[:split_idx]
        X_test = X.iloc[split_idx:]
        y_train = y.iloc[:split_idx]
        y_test = y.iloc[split_idx:]
        
        # Train and evaluate
        model_copy = self._copy_model(model)
        model_copy.train(X_train, y_train)
        y_pred = model_copy.predict(X_test)
        
        metrics = self._calculate_fold_metrics(y_test, y_pred)
        metrics['validation_type'] = 'simple_split'
        metrics['train_size'] = len(X_train)
        metrics['test_size'] = len(X_test)
        
        return {'overall_metrics': metrics, 'fold_results': [metrics]}
    
    def _copy_model(self, model):
        """Create a copy of the model for cross-validation"""
        # Import the model class and create new instance
        model_class = model.__class__
        return model_class(**getattr(model, 'model_params', {}))
    
    def _calculate_fold_metrics(self, y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate metrics for a single fold"""
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1_score': f1_score(y_true, y_pred, average='weighted', zero_division=0)
        }
        
        # Add AUC if we have both classes
        if len(y_true.unique()) > 1:
            try:
                metrics['roc_auc'] = roc_auc_score(y_true, y_pred)
            except ValueError:
                metrics['roc_auc'] = 0.0
        else:
            metrics['roc_auc'] = 0.0
        
        return metrics
    
    def _aggregate_cv_results(self, fold_results) -> Dict[str, Any]:
        """Aggregate cross-validation results"""
        
        # Calculate means and standard deviations
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        
        overall_metrics = {}
        for metric in metrics:
            values = [fold[metric] for fold in fold_results]
            overall_metrics[f'{metric}_mean'] = np.mean(values)
            overall_metrics[f'{metric}_std'] = np.std(values)
        
        overall_metrics['validation_type'] = 'time_series_cv'
        overall_metrics['n_folds'] = len(fold_results)
        
        return {
            'overall_metrics': overall_metrics,
            'fold_results': fold_results
        }