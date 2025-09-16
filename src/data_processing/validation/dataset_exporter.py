import os
import joblib
import pandas as pd
from typing import Dict
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from src.shared.logging import get_logger
from src.data_processing.validation.feature_selector import FeatureSelector



class MLDatasetExporter:
    """Export ML-ready datasets for Bitcoin price prediction"""
    
    def __init__(self, output_dir: str = "data/ml_datasets"):
        self.logger = get_logger(__name__)
        self.feature_selector = FeatureSelector()
        self.output_dir = output_dir
        self.scalers = {}
        self.label_encoders = {}
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
    
    def prepare_training_data(self, df: pd.DataFrame, target_col: str = 'price_direction') -> Dict[str, pd.DataFrame]:
        """Prepare data for ML training with proper splits"""
        
        if df.empty or target_col not in df.columns:
            self.logger.error(f"Invalid dataset or missing target column: {target_col}")
            return {}
        
        # Remove any rows with NaN in target
        clean_df = df.dropna(subset=[target_col]).copy()
        
        if len(clean_df) < 5:
            self.logger.warning(f"Insufficient data for training: {len(clean_df)} samples")
            return {'full_dataset': clean_df}
        
        # Separate features and target
        feature_cols = [col for col in clean_df.columns if col != target_col]
        X = clean_df[feature_cols].fillna(0)  # Fill missing features with 0
        y = clean_df[target_col]
        
        # Create train/validation/test splits
        if len(clean_df) >= 10:
            # Split into train (60%), validation (20%), test (20%)
            X_temp, X_test, y_temp, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y if len(y.unique()) > 1 else None
            )
            
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp if len(y_temp.unique()) > 1 else None
            )
            
            datasets = {
                'train': pd.concat([X_train, y_train], axis=1),
                'validation': pd.concat([X_val, y_val], axis=1),
                'test': pd.concat([X_test, y_test], axis=1),
                'full_dataset': clean_df
            }
            
        else:
            # Too few samples for proper splitting
            datasets = {
                'full_dataset': clean_df
            }
        
        return datasets
    
    def scale_features(self, datasets: Dict[str, pd.DataFrame], target_col: str) -> Dict[str, pd.DataFrame]:
        """Apply feature scaling to datasets"""
        
        if 'train' not in datasets:
            return datasets
        
        # Get feature columns
        feature_cols = [col for col in datasets['train'].columns if col != target_col]
        
        # Fit scaler on training data
        scaler = StandardScaler()
        train_features = datasets['train'][feature_cols]
        scaler.fit(train_features)
        
        # Store scaler for later use
        self.scalers[target_col] = scaler
        
        # Apply scaling to all datasets
        scaled_datasets = {}
        for split_name, df in datasets.items():
            scaled_df = df.copy()
            scaled_df[feature_cols] = scaler.transform(df[feature_cols])
            scaled_datasets[split_name] = scaled_df
        
        return scaled_datasets
    
    def export_datasets(self, target_col: str = 'price_direction', apply_scaling: bool = True) -> Dict[str, str]:
        """Export complete ML datasets"""
        
        self.logger.info(f"Preparing ML dataset for target: {target_col}")
        
        # Get optimized feature set
        optimized_df = self.feature_selector.create_optimized_feature_set()
        
        if optimized_df.empty:
            self.logger.warning("No data available for ML dataset export")
            return {}
        
        # Prepare training splits
        datasets = self.prepare_training_data(optimized_df, target_col)
        
        if not datasets:
            return {}
        
        # Apply feature scaling if requested
        if apply_scaling and 'train' in datasets:
            datasets = self.scale_features(datasets, target_col)
        
        # Export datasets
        exported_files = {}
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for split_name, df in datasets.items():
            filename = f"bitcoin_prediction_{target_col}_{split_name}_{timestamp}.csv"
            filepath = os.path.join(self.output_dir, filename)
            
            df.to_csv(filepath, index=False)
            exported_files[split_name] = filepath
            
            self.logger.info(f"Exported {split_name}: {len(df)} samples to {filename}")
        
        # Export feature metadata
        self.export_feature_metadata(optimized_df, target_col, timestamp)
        
        # Export scalers if used
        if apply_scaling and self.scalers:
            self.export_preprocessing_objects(target_col, timestamp)
        
        return exported_files
    
    def export_feature_metadata(self, df: pd.DataFrame, target_col: str, timestamp: str):
        """Export feature metadata and analysis"""
        
        # Analyze features
        analysis = self.feature_selector.analyze_feature_importance(df, target_col)
        
        # Create metadata
        metadata = {
            'export_timestamp': timestamp,
            'target_column': target_col,
            'total_features': len(df.columns) - 1,  # Exclude target
            'total_samples': len(df),
            'feature_analysis': analysis,
            'feature_list': [col for col in df.columns if col != target_col],
            'data_quality': {
                'missing_values': df.isnull().sum().to_dict(),
                'feature_types': df.dtypes.astype(str).to_dict()
            }
        }
        
        # Export as JSON
        import json
        metadata_file = os.path.join(self.output_dir, f"feature_metadata_{target_col}_{timestamp}.json")
        
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        self.logger.info(f"Exported feature metadata to {metadata_file}")
    
    def export_preprocessing_objects(self, target_col: str, timestamp: str):
        """Export preprocessing objects (scalers, encoders)"""
        
        preprocessing_file = os.path.join(self.output_dir, f"preprocessing_{target_col}_{timestamp}.joblib")
        
        preprocessing_objects = {
            'scalers': self.scalers,
            'label_encoders': self.label_encoders,
            'target_column': target_col,
            'export_timestamp': timestamp
        }
        
        joblib.dump(preprocessing_objects, preprocessing_file)
        self.logger.info(f"Exported preprocessing objects to {preprocessing_file}")
    
    def generate_dataset_summary(self) -> Dict[str, any]:
        """Generate summary of available datasets"""
        
        # Get feature set
        optimized_df = self.feature_selector.create_optimized_feature_set()
        
        if optimized_df.empty:
            return {
                'status': 'NO_DATA',
                'message': 'Insufficient data for ML dataset creation',
                'recommendations': [
                    'Collect more Bitcoin price data (need 10+ records)',
                    'Ensure temporal overlap between price and news data'
                ]
            }
        
        # Analyze dataset
        summary = {
            'status': 'READY' if len(optimized_df) >= 10 else 'LIMITED',
            'total_samples': len(optimized_df),
            'total_features': len(optimized_df.columns) - 1,
            'target_variables': [col for col in optimized_df.columns if col in ['price_direction', 'price_movement']],
            'feature_categories': self.feature_selector.combiner.get_feature_importance_info(),
            'data_quality_score': self._calculate_quality_score(optimized_df),
            'recommendations': []
        }
        
        # Add recommendations
        if len(optimized_df) < 10:
            summary['recommendations'].append("Collect more price data for robust ML training")
        
        if len(optimized_df) < 50:
            summary['recommendations'].append("Additional data will improve model performance")
        
        return summary
    
    def _calculate_quality_score(self, df: pd.DataFrame) -> float:
        """Calculate overall data quality score (0-1)"""
        
        if df.empty:
            return 0.0
        
        # Factors affecting quality
        sample_size_score = min(len(df) / 50, 1.0)  # 50+ samples = full score
        feature_completeness = 1.0 - (df.isnull().sum().sum() / (len(df) * len(df.columns)))
        feature_variety_score = min(len(df.columns) / 20, 1.0)  # 20+ features = full score
        
        # Weighted average
        quality_score = (0.4 * sample_size_score + 0.4 * feature_completeness + 0.2 * feature_variety_score)
        
        return round(quality_score, 3)