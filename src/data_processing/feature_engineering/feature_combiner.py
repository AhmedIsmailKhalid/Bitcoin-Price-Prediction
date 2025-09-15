import pandas as pd
from typing import Dict, List
from datetime import timedelta

from src.shared.database import SessionLocal
from src.shared.models import NewsData, SentimentData
from src.data_processing.feature_engineering.price_features import PriceFeatureExtractor
from src.data_processing.feature_engineering.temporal_features import TemporalFeatureExtractor


class FeatureCombiner:
    """Combine price, sentiment, and temporal features for ML training"""
    
    def __init__(self):
        self.price_extractor = PriceFeatureExtractor()
        self.temporal_extractor = TemporalFeatureExtractor()
    
    def get_sentiment_features(self, time_window_hours: int = 24) -> pd.DataFrame:
        """Get sentiment features aggregated by time windows"""
        
        db = SessionLocal()
        try:
            # Get all sentiment data with associated news data
            query = db.query(
                SentimentData, NewsData
            ).join(NewsData, SentimentData.news_article_id == NewsData.id)
            
            sentiment_records = query.all()
            
            if not sentiment_records:
                return pd.DataFrame()
            
            # Convert to DataFrame
            data = []
            for sentiment, news in sentiment_records:
                data.append({
                    'timestamp': news.published_at or news.collected_at,
                    'sentiment_score': sentiment.combined_sentiment,
                    'vader_compound': sentiment.vader_compound,
                    'textblob_polarity': sentiment.textblob_polarity,
                    'textblob_subjectivity': sentiment.textblob_subjectivity,
                    'sentiment_category': sentiment.sentiment_category,
                    'news_source': news.data_source,
                    'title_sentiment': sentiment.title_sentiment,
                    'content_sentiment': sentiment.content_sentiment,
                    'sentiment_divergence': sentiment.sentiment_divergence,
                    'sentiment_alignment': int(sentiment.sentiment_alignment) if sentiment.sentiment_alignment is not None else 0
                })
            
            df = pd.DataFrame(data)
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            return df
            
        finally:
            db.close()
    
    def aggregate_sentiment_by_time(self, sentiment_df: pd.DataFrame, time_window_hours: int = 24) -> pd.DataFrame:
        """Aggregate sentiment data into time windows"""
        
        if sentiment_df.empty:
            return pd.DataFrame()
        
        # Set timestamp as index for resampling
        sentiment_df = sentiment_df.copy()
        sentiment_df['timestamp'] = pd.to_datetime(sentiment_df['timestamp'])
        sentiment_df.set_index('timestamp', inplace=True)
        
        # Resample by time window (e.g., 24H for daily aggregation)
        resampling_rule = f'{time_window_hours}h'
        
        # Aggregate sentiment features
        agg_funcs = {
            'sentiment_score': ['mean', 'std', 'min', 'max', 'count'],
            'vader_compound': ['mean', 'std'],
            'textblob_polarity': ['mean', 'std'],
            'textblob_subjectivity': 'mean',
            'sentiment_divergence': 'mean',
            'sentiment_alignment': 'mean'
        }
        
        aggregated = sentiment_df.resample(resampling_rule).agg(agg_funcs)
        
        # Flatten column names
        aggregated.columns = ['_'.join(col).strip() for col in aggregated.columns]
        
        # Reset index to get timestamp as column
        aggregated = aggregated.reset_index()
        
        # Handle NaN values
        aggregated = aggregated.fillna(0)
        
        return aggregated
    
    def create_target_variable(self, price_df: pd.DataFrame, prediction_horizon_hours: int = 1) -> pd.DataFrame:
        """Create target variable for price prediction"""
        
        if price_df.empty or len(price_df) < 2:
            return price_df
        
        price_df = price_df.copy()
        
        # Sort by timestamp
        price_df = price_df.sort_values('timestamp').reset_index(drop=True)
        
        # Calculate future price change (target variable)
        price_df['future_price'] = price_df['price'].shift(-prediction_horizon_hours)
        price_df['price_change_target'] = ((price_df['future_price'] - price_df['price']) / price_df['price'])
        
        # Create binary target (price increase/decrease)
        price_df['price_direction'] = (price_df['price_change_target'] > 0).astype(int)
        
        # Create categorical target (significant moves)
        price_df['price_movement'] = 0  # No significant movement
        price_df.loc[price_df['price_change_target'] > 0.02, 'price_movement'] = 1  # Significant increase (>2%)
        price_df.loc[price_df['price_change_target'] < -0.02, 'price_movement'] = -1  # Significant decrease (<-2%)
        
        return price_df
    
    def align_features_by_time(
        self, 
        price_df: pd.DataFrame, 
        sentiment_df: pd.DataFrame, 
        alignment_tolerance_hours: int = 2
    ) -> pd.DataFrame:
        """Align price and sentiment features by timestamp"""
        
        if price_df.empty or sentiment_df.empty:
            return pd.DataFrame()
        
        aligned_data = []
        
        for _, price_row in price_df.iterrows():
            price_time = pd.to_datetime(price_row['timestamp'])
            
            # Find sentiment data within time window
            sentiment_df['timestamp'] = pd.to_datetime(sentiment_df['timestamp'])
            time_mask = abs(sentiment_df['timestamp'] - price_time) <= timedelta(hours=alignment_tolerance_hours)
            
            matching_sentiment = sentiment_df[time_mask]
            
            if not matching_sentiment.empty:
                # Use the closest sentiment data
                time_diffs = abs(matching_sentiment['timestamp'] - price_time)
                closest_idx = time_diffs.idxmin()
                sentiment_row = matching_sentiment.loc[closest_idx]
                
                # Combine price and sentiment features
                combined_row = {**price_row.to_dict(), **sentiment_row.to_dict()}
                aligned_data.append(combined_row)
        
        return pd.DataFrame(aligned_data)
    
    def create_complete_feature_set(
        self, 
        prediction_horizon_hours: int = 1,
        sentiment_window_hours: int = 24
    ) -> pd.DataFrame:
        """Create complete feature set for ML training"""
        
        # Get price features
        price_df = self.price_extractor.extract_all_features()
        
        if price_df.empty:
            return pd.DataFrame()
        
        # Add temporal features
        price_df = self.temporal_extractor.extract_all_temporal_features(price_df)
        
        # Create target variables
        price_df = self.create_target_variable(price_df, prediction_horizon_hours)
        
        # Get sentiment features
        sentiment_df = self.get_sentiment_features()
        
        if not sentiment_df.empty:
            # Aggregate sentiment by time windows
            sentiment_agg = self.aggregate_sentiment_by_time(sentiment_df, sentiment_window_hours)
            
            # Align price and sentiment features
            combined_df = self.align_features_by_time(price_df, sentiment_agg)
        else:
            combined_df = price_df
        
        # Remove rows with NaN targets
        if 'price_change_target' in combined_df.columns:
            combined_df = combined_df.dropna(subset=['price_change_target'])
        
        return combined_df
    
    def get_feature_importance_info(self) -> Dict[str, List[str]]:
        """Get categorized feature names for analysis"""
        
        return {
            'price_features': self.price_extractor.get_feature_names(),
            'temporal_features': self.temporal_extractor.get_feature_names(),
            'sentiment_features': [
                'sentiment_score_mean', 'sentiment_score_std', 'sentiment_score_count',
                'vader_compound_mean', 'textblob_polarity_mean', 'sentiment_alignment_mean'
            ]
        }