import pandas as pd
import numpy as np
from typing import List


class TemporalFeatureExtractor:
    """Extract time-based features from timestamps"""
    
    def __init__(self):
        self.feature_names = []
    
    def extract_datetime_features(self, df: pd.DataFrame, timestamp_col: str = 'timestamp') -> pd.DataFrame:
        """Extract basic datetime features"""
        
        df = df.copy()
        
        # Ensure timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
            df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        
        # Extract basic time components
        df['hour'] = df[timestamp_col].dt.hour
        df['day_of_week'] = df[timestamp_col].dt.dayofweek  # 0=Monday, 6=Sunday
        df['day_of_month'] = df[timestamp_col].dt.day
        df['month'] = df[timestamp_col].dt.month
        df['quarter'] = df[timestamp_col].dt.quarter
        
        self.feature_names.extend(['hour', 'day_of_week', 'day_of_month', 'month', 'quarter'])
        
        return df
    
    def extract_cyclical_features(self, df: pd.DataFrame, timestamp_col: str = 'timestamp') -> pd.DataFrame:
        """Extract cyclical time features using sine/cosine encoding"""
        
        df = df.copy()
        
        # Ensure timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
            df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        
        # Hour cycle (24 hours)
        df['hour_sin'] = np.sin(2 * np.pi * df[timestamp_col].dt.hour / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df[timestamp_col].dt.hour / 24)
        
        # Day of week cycle (7 days)
        df['dow_sin'] = np.sin(2 * np.pi * df[timestamp_col].dt.dayofweek / 7)
        df['dow_cos'] = np.cos(2 * np.pi * df[timestamp_col].dt.dayofweek / 7)
        
        # Month cycle (12 months)
        df['month_sin'] = np.sin(2 * np.pi * df[timestamp_col].dt.month / 12)
        df['month_cos'] = np.cos(2 * np.pi * df[timestamp_col].dt.month / 12)
        
        self.feature_names.extend(['hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'month_sin', 'month_cos'])
        
        return df
    
    def extract_time_since_features(self, df: pd.DataFrame, timestamp_col: str = 'timestamp') -> pd.DataFrame:
        """Extract time-since-event features"""
        
        df = df.copy()
        
        # Ensure timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
            df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        
        # Time since start of dataset
        min_time = df[timestamp_col].min()
        df['time_since_start_hours'] = (df[timestamp_col] - min_time).dt.total_seconds() / 3600
        
        # Time since start of day (minutes)
        df['minutes_since_midnight'] = df[timestamp_col].dt.hour * 60 + df[timestamp_col].dt.minute
        
        self.feature_names.extend(['time_since_start_hours', 'minutes_since_midnight'])
        
        return df
    
    def extract_business_time_features(self, df: pd.DataFrame, timestamp_col: str = 'timestamp') -> pd.DataFrame:
        """Extract business/trading time related features"""
        
        df = df.copy()
        
        # Ensure timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
            df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        
        # Business hours (rough approximation for global crypto markets)
        # Crypto trades 24/7, but traditional market hours may still influence sentiment
        hour = df[timestamp_col].dt.hour
        
        # US market hours (9 AM - 4 PM EST, approximated in UTC)
        df['is_us_market_hours'] = ((hour >= 14) & (hour <= 21)).astype(int)  # Rough UTC conversion
        
        # Weekend indicator
        df['is_weekend'] = (df[timestamp_col].dt.dayofweek >= 5).astype(int)
        
        # Early morning hours (might have different volatility patterns)
        df['is_early_morning'] = ((hour >= 0) & (hour <= 6)).astype(int)
        
        # Late evening hours
        df['is_late_evening'] = ((hour >= 20) & (hour <= 23)).astype(int)
        
        self.feature_names.extend(['is_us_market_hours', 'is_weekend', 'is_early_morning', 'is_late_evening'])
        
        return df
    
    def extract_all_temporal_features(self, df: pd.DataFrame, timestamp_col: str = 'timestamp') -> pd.DataFrame:
        """Extract all temporal features"""
        
        # Reset feature names
        self.feature_names = []
        
        # Apply all temporal feature extraction methods
        df = self.extract_datetime_features(df, timestamp_col)
        df = self.extract_cyclical_features(df, timestamp_col)
        df = self.extract_time_since_features(df, timestamp_col)
        df = self.extract_business_time_features(df, timestamp_col)
        
        return df
    
    def get_feature_names(self) -> List[str]:
        """Get list of all temporal feature names created"""
        return self.feature_names.copy()