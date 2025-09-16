import pandas as pd
import numpy as np
from typing import Dict, List
from scipy.stats import pearsonr
from sklearn.feature_selection import SelectKBest, f_classif

from src.shared.logging import get_logger
from src.data_processing.feature_engineering.feature_combiner import FeatureCombiner


class FeatureSelector:
    """Advanced feature selection for Bitcoin price prediction"""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.combiner = FeatureCombiner()
        self.selected_features = []
        self.feature_scores = {}
    
    def analyze_feature_correlations(self, df: pd.DataFrame, target_col: str = 'price_direction') -> Dict[str, float]:
        """Analyze correlations between features and target variable"""
        
        if df.empty or target_col not in df.columns:
            return {}
        
        # Select numeric columns only
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove target column from features
        if target_col in numeric_cols:
            numeric_cols.remove(target_col)
        
        correlations = {}
        target_values = df[target_col].dropna()
        
        for feature in numeric_cols:
            feature_values = df[feature].dropna()
            
            # Only calculate correlation if we have matching indices
            common_idx = target_values.index.intersection(feature_values.index)
            
            if len(common_idx) > 1:
                try:
                    corr_coeff, p_value = pearsonr(
                        target_values.loc[common_idx], 
                        feature_values.loc[common_idx]
                    )
                    
                    if not np.isnan(corr_coeff):
                        correlations[feature] = abs(corr_coeff)  # Use absolute correlation
                        
                except Exception:
                    continue
        
        return correlations
    
    def remove_highly_correlated_features(self, df: pd.DataFrame, threshold: float = 0.95) -> List[str]:
        """Remove features with high inter-correlation"""
        
        numeric_df = df.select_dtypes(include=[np.number])
        
        # Calculate correlation matrix
        corr_matrix = numeric_df.corr().abs()
        
        # Create upper triangle mask
        _ = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        
        # Find highly correlated pairs
        high_corr_pairs = []
        features_to_drop = set()
        
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if corr_matrix.iloc[i, j] > threshold:
                    feature1 = corr_matrix.columns[i]
                    feature2 = corr_matrix.columns[j]
                    high_corr_pairs.append((feature1, feature2, corr_matrix.iloc[i, j]))
                    
                    # Keep the feature with more variance
                    var1 = numeric_df[feature1].var()
                    var2 = numeric_df[feature2].var()
                    
                    if var1 < var2:
                        features_to_drop.add(feature1)
                    else:
                        features_to_drop.add(feature2)
        
        if high_corr_pairs:
            self.logger.info(f"Found {len(high_corr_pairs)} highly correlated feature pairs")
            self.logger.info(f"Removing {len(features_to_drop)} redundant features")
        
        remaining_features = [col for col in numeric_df.columns if col not in features_to_drop]
        return remaining_features
    
    def select_univariate_features(self, df: pd.DataFrame, target_col: str, k: int = 10) -> List[str]:
        """Select top k features using univariate statistical tests"""
        
        if df.empty or target_col not in df.columns:
            return []
        
        # Prepare features and target
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if target_col in numeric_cols:
            numeric_cols.remove(target_col)
        
        if len(numeric_cols) == 0:
            return []
        
        # Remove rows with NaN in target
        clean_df = df[[target_col] + numeric_cols].dropna()
        
        if len(clean_df) < 2:
            self.logger.warning("Insufficient data for feature selection")
            return numeric_cols[:k]  # Return first k features
        
        X = clean_df[numeric_cols]
        y = clean_df[target_col]
        
        # Handle any remaining NaN values
        X = X.fillna(X.mean())
        
        try:
            # Use f_classif for classification targets
            selector = SelectKBest(score_func=f_classif, k=min(k, len(numeric_cols)))
            _ = selector.fit_transform(X, y)
            
            # Get selected feature names
            selected_features = [numeric_cols[i] for i in selector.get_support(indices=True)]
            
            # Store scores
            feature_scores = dict(zip(numeric_cols, selector.scores_))
            self.feature_scores.update(feature_scores)
            
            return selected_features
            
        except Exception as e:
            self.logger.warning(f"Univariate selection failed: {e}")
            return numeric_cols[:k]
    
    def analyze_feature_importance(self, df: pd.DataFrame, target_col: str = 'price_direction') -> Dict[str, Dict]:
        """Comprehensive feature importance analysis"""
        
        analysis = {
            'correlations': {},
            'univariate_scores': {},
            'recommendations': []
        }
        
        if df.empty:
            analysis['recommendations'].append("No data available for feature analysis")
            return analysis
        
        # Correlation analysis
        correlations = self.analyze_feature_correlations(df, target_col)
        analysis['correlations'] = correlations
        
        # Univariate feature selection
        top_features = self.select_univariate_features(df, target_col, k=15)
        analysis['univariate_scores'] = {f: self.feature_scores.get(f, 0) for f in top_features}
        
        # Generate recommendations
        if len(correlations) > 0:
            top_corr_features = sorted(correlations.items(), key=lambda x: x[1], reverse=True)[:5]
            analysis['recommendations'].append(f"Top correlated features: {[f[0] for f in top_corr_features]}")
        
        if len(top_features) > 0:
            analysis['recommendations'].append(f"Selected {len(top_features)} features via univariate analysis")
        
        # Check for feature redundancy
        redundant_features = self.remove_highly_correlated_features(df)
        removed_count = len(df.select_dtypes(include=[np.number]).columns) - len(redundant_features)
        
        if removed_count > 0:
            analysis['recommendations'].append(f"Consider removing {removed_count} highly correlated features")
        
        return analysis
    
    def create_optimized_feature_set(self, max_features: int = 20) -> pd.DataFrame:
        """Create optimized feature set for ML training"""
        
        # Get complete feature set
        complete_df = self.combiner.create_complete_feature_set()
        
        if complete_df.empty:
            self.logger.warning("No complete feature set available - insufficient price data")
            return pd.DataFrame()
        
        # Analyze features
        _ = self.analyze_feature_importance(complete_df)
        
        # Remove highly correlated features
        remaining_features = self.remove_highly_correlated_features(complete_df)
        
        # Select best features
        target_col = 'price_direction' if 'price_direction' in complete_df.columns else None
        
        if target_col:
            # Prepare optimized dataset
            feature_cols = [col for col in remaining_features if col not in ['price_direction', 'price_movement', 'price_change_target']]
            
            optimized_df = complete_df[feature_cols + [target_col]].copy()
            
            # Select top features
            if len(feature_cols) > max_features:
                correlations = self.analyze_feature_correlations(optimized_df, target_col)
                top_features = sorted(correlations.items(), key=lambda x: x[1], reverse=True)[:max_features]
                selected_feature_names = [f[0] for f in top_features]
                optimized_df = complete_df[selected_feature_names + [target_col]]
            
            self.selected_features = list(optimized_df.columns)
            return optimized_df
        
        return complete_df