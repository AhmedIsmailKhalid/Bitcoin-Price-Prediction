"""ML training pipeline for Bitcoin price prediction"""
import pandas as pd
from typing import Dict, Any, Tuple
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.ml_models.classifiers.logistic_regression import LogisticRegressionModel
from src.ml_models.classifiers.random_forest import RandomForestModel
from src.data_processing.validation.dataset_exporter import MLDatasetExporter
from src.shared.logging import get_logger


class MLTrainingPipeline:
    """Complete ML training pipeline"""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.dataset_exporter = MLDatasetExporter()
        self.models = {}
        self.results = {}
        
    def load_training_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Load training data from the latest dataset"""
        self.logger.info("Loading training data...")
        
        # Get optimized feature set
        feature_df = self.dataset_exporter.feature_selector.create_optimized_feature_set()
        
        if feature_df.empty:
            raise ValueError("No training data available. Ensure sufficient price and sentiment data is collected.")
        
        # Separate features and target
        target_col = 'price_direction'
        if target_col not in feature_df.columns:
            raise ValueError(f"Target column '{target_col}' not found in dataset")
        
        X = feature_df.drop(columns=[target_col])
        y = feature_df[target_col]
        
        self.logger.info(f"Loaded {len(X)} samples with {len(X.columns)} features")
        return X, y
    
    def prepare_data(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2) -> Dict[str, Any]:
        """Prepare data for training"""
        self.logger.info("Preparing data for training...")
        
        # Check for NaN values
        nan_counts = X.isnull().sum()
        if nan_counts.sum() > 0:
            self.logger.warning(f"Found NaN values in features: {nan_counts[nan_counts > 0].to_dict()}")
            
            # Fill NaN values - use median for numeric features
            X_clean = X.fillna(X.median())
            self.logger.info("Filled NaN values with median values")
        else:
            X_clean = X.copy()
        
        # Check class distribution
        class_counts = y.value_counts()
        self.logger.info(f"Class distribution: {class_counts.to_dict()}")
        
        # Check if we have enough data for splitting
        min_class_size = class_counts.min()
        if len(X_clean) < 5 or min_class_size < 2:
            self.logger.warning(f"Limited data ({len(X_clean)} samples) or class imbalance. Using all data for training, no test split.")
            
            # Scale features for full dataset
            scaler = StandardScaler()
            X_scaled = pd.DataFrame(
                scaler.fit_transform(X_clean),
                columns=X_clean.columns,
                index=X_clean.index
            )
            
            return {
                'X_train': X_scaled,
                'y_train': y,
                'X_test': None,
                'y_test': None,
                'scaler': scaler
            }
        
        # Split data (only if we have balanced classes)
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X_clean, y, test_size=test_size, random_state=42, 
                stratify=y if min_class_size >= 2 else None
            )
        except ValueError:
            # Fallback: random split without stratification
            X_train, X_test, y_train, y_test = train_test_split(
                X_clean, y, test_size=test_size, random_state=42, stratify=None
            )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )
        X_test_scaled = pd.DataFrame(
            scaler.transform(X_test),
            columns=X_test.columns,
            index=X_test.index
        )
        
        self.logger.info(f"Data split: {len(X_train)} train, {len(X_test)} test samples")
        
        return {
            'X_train': X_train_scaled,
            'X_test': X_test_scaled,
            'y_train': y_train,
            'y_test': y_test,
            'scaler': scaler
        }

    
    def train_baseline_models(self, data: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
        """Train baseline models"""
        self.logger.info("Training baseline models...")
        
        X_train = data['X_train']
        y_train = data['y_train']
        X_test = data['X_test']
        y_test = data['y_test']
        
        # Initialize models
        models = {
            'logistic_regression': LogisticRegressionModel(),
            'random_forest': RandomForestModel()
        }
        
        results = {}
        
        for model_name, model in models.items():
            self.logger.info(f"Training {model_name}...")
            
            try:
                # Train model
                training_metrics = model.train(X_train, y_train)
                
                # Evaluate on test set if available
                if X_test is not None and y_test is not None:
                    test_metrics = model.evaluate(X_test, y_test)
                else:
                    test_metrics = {}
                
                # Store model and results
                self.models[model_name] = model
                results[model_name] = {
                    'training_metrics': training_metrics,
                    'test_metrics': test_metrics,
                    'feature_importance': model.get_feature_importance()
                }
                
                self.logger.info(f"{model_name} training completed")
                
            except Exception as e:
                self.logger.error(f"Failed to train {model_name}: {e}")
                results[model_name] = {'error': str(e)}
        
        self.results = results
        return results
    
    def run_complete_pipeline(self) -> Dict[str, Any]:
        """Run the complete training pipeline"""
        try:
            # Load data
            X, y = self.load_training_data()
            
            # Prepare data
            data = self.prepare_data(X, y)
            
            # Train models
            results = self.train_baseline_models(data)
            
            # Generate summary
            summary = self.generate_training_summary(results, len(X))
            
            return {
                'data_info': {
                    'total_samples': len(X),
                    'feature_count': len(X.columns),
                    'target_distribution': y.value_counts().to_dict()
                },
                'model_results': results,
                'summary': summary
            }
            
        except Exception as e:
            self.logger.error(f"Training pipeline failed: {e}")
            return {'error': str(e)}
    
    def generate_training_summary(self, results: Dict[str, Any], sample_count: int) -> Dict[str, Any]:
        """Generate training summary"""
        summary = {
            'total_samples': sample_count,
            'models_trained': len([r for r in results.values() if 'error' not in r]),
            'best_model': None,
            'best_accuracy': 0.0
        }
        
        # Find best performing model
        for model_name, result in results.items():
            if 'error' not in result and 'training_metrics' in result:
                accuracy = result['training_metrics'].get('accuracy', 0.0)
                if accuracy > summary['best_accuracy']:
                    summary['best_accuracy'] = accuracy
                    summary['best_model'] = model_name
        
        return summary