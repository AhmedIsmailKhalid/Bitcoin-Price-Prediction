"""Voting ensemble combining multiple classifiers"""
from sklearn.ensemble import VotingClassifier
from src.ml_models.base_model import BaseMLModel


class VotingEnsemble(BaseMLModel):
    """Voting classifier ensemble"""
    
    def __init__(self, estimators, voting='soft', **kwargs):
        super().__init__("VotingEnsemble")
        self.estimators = estimators
        self.voting = voting
        self.model_params = kwargs
    
    def _create_model(self):
        """Create voting ensemble"""
        return VotingClassifier(
            estimators=self.estimators,
            voting=self.voting,
            **self.model_params
        )
    
    def get_feature_importance(self):
        """Get averaged feature importance from ensemble members"""
        if not self.is_trained:
            return None
        
        valid_estimators = []
        
        # Collect importance from estimators that support it
        for name, estimator in self.model.named_estimators_.items():
            if hasattr(estimator, 'feature_importances_'):
                valid_estimators.append((name, estimator.feature_importances_))
            elif hasattr(estimator, 'coef_'):
                # For linear models, use absolute coefficients
                coef_scores = abs(estimator.coef_[0] if len(estimator.coef_.shape) > 1 else estimator.coef_)
                valid_estimators.append((name, coef_scores))
        
        if not valid_estimators:
            return None
        
        # Average importance scores
        import numpy as np
        avg_importance = np.mean([scores for _, scores in valid_estimators], axis=0)
        
        return dict(zip(self.feature_names, avg_importance))