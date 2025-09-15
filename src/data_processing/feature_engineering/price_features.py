import numpy as np
import pandas as pd
from typing import List, Optional

from src.shared.database import SessionLocal
from src.shared.models import PriceData


class PriceFeatureExtractor:
    """Extract technical indicators and price-based features from Bitcoin data"""
    
    def __init__(self):
        self.feature_names = []
    
    def get_price_data(self, limit: Optional[int] = None) -> pd.DataFrame:
        """Retrieve price data from database as DataFrame"""
        
        db = SessionLocal()
        try:
            query = db.query(PriceData).order_by(PriceData.collected_at.desc())
            
            if limit:
                query = query.limit(limit)
            
            price_records = query.all()
            
            if not price_records:
                return pd.DataFrame()
            
            # Convert to DataFrame - FIXED FIELD NAMES
            data = []
            for record in price_records:
                data.append({
                    'timestamp': record.collected_at,
                    'price': record.price_usd,
                    'market_cap': record.market_cap,  # Changed from market_cap_usd
                    'volume': record.volume_24h,      # Changed from volume_24h_usd
                    'change_1h': record.change_1h,
                    'change_24h': record.change_24h,
                    'change_7d': record.change_7d
                })
            
            df = pd.DataFrame(data)
            
            # Sort by timestamp (oldest first for technical indicators)
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            return df
            
        finally:
            db.close()
    
    def calculate_moving_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate various moving averages"""
        
        df = df.copy()
        
        # Simple Moving Averages
        periods = [3, 5, 7, 14]  # Short periods due to limited data
        
        for period in periods:
            if len(df) >= period:
                df[f'sma_{period}'] = df['price'].rolling(window=period).mean()
                self.feature_names.append(f'sma_{period}')
        
        # Exponential Moving Averages
        for period in periods:
            if len(df) >= period:
                df[f'ema_{period}'] = df['price'].ewm(span=period).mean()
                self.feature_names.append(f'ema_{period}')
        
        return df
    
    def calculate_price_ratios(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate price ratios and relative positions"""
        
        df = df.copy()
        
        # Current price relative to moving averages
        for period in [3, 5, 7]:
            sma_col = f'sma_{period}'
            ema_col = f'ema_{period}'
            
            if sma_col in df.columns:
                df[f'price_sma_{period}_ratio'] = df['price'] / df[sma_col]
                self.feature_names.append(f'price_sma_{period}_ratio')
            
            if ema_col in df.columns:
                df[f'price_ema_{period}_ratio'] = df['price'] / df[ema_col]
                self.feature_names.append(f'price_ema_{period}_ratio')
        
        return df
    
    def calculate_volatility_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate volatility and momentum indicators"""
        
        df = df.copy()
        
        # Price changes and returns
        df['price_change_1'] = df['price'].pct_change(1)  # 1-period return
        df['price_change_2'] = df['price'].pct_change(2)  # 2-period return
        self.feature_names.extend(['price_change_1', 'price_change_2'])
        
        # Rolling volatility (standard deviation of returns)
        for window in [3, 5, 7]:
            if len(df) >= window:
                df[f'volatility_{window}'] = df['price_change_1'].rolling(window=window).std()
                self.feature_names.append(f'volatility_{window}')
        
        # Volume-price indicators
        if 'volume' in df.columns:
            df['price_volume_ratio'] = df['price'] / (df['volume'] + 1)  # Add 1 to avoid division by zero
            df['volume_change'] = df['volume'].pct_change(1)
            self.feature_names.extend(['price_volume_ratio', 'volume_change'])
        
        return df
    
    def calculate_momentum_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate momentum-based technical indicators"""
        
        df = df.copy()
        
        # Rate of Change (ROC)
        for period in [2, 3, 5]:
            if len(df) > period:
                df[f'roc_{period}'] = ((df['price'] - df['price'].shift(period)) / df['price'].shift(period)) * 100
                self.feature_names.append(f'roc_{period}')
        
        # RSI-like indicator (simplified)
        window = 5  # Shorter window due to limited data
        if len(df) >= window:
            delta = df['price'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
            rs = gain / (loss + 1e-10)  # Add small value to avoid division by zero
            df['rsi_like'] = 100 - (100 / (1 + rs))
            self.feature_names.append('rsi_like')
        
        return df
    
    def calculate_trend_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate trend-based indicators"""
        
        df = df.copy()
        
        # Price position within recent range
        for window in [3, 5, 7]:
            if len(df) >= window:
                rolling_min = df['price'].rolling(window=window).min()
                rolling_max = df['price'].rolling(window=window).max()
                
                # Avoid division by zero
                price_range = rolling_max - rolling_min
                df[f'price_position_{window}'] = np.where(
                    price_range > 0,
                    (df['price'] - rolling_min) / price_range,
                    0.5  # Neutral position if no range
                )
                self.feature_names.append(f'price_position_{window}')
        
        # Trend direction (simple slope)
        for window in [3, 5]:
            if len(df) >= window:
                # Calculate slope of price over window
                df[f'trend_{window}'] = df['price'].rolling(window=window).apply(
                    lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == window else np.nan
                )
                self.feature_names.append(f'trend_{window}')
        
        return df
    
    def calculate_market_cap_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate market cap related features"""
        
        df = df.copy()
        
        if 'market_cap' in df.columns:
            # Market cap changes
            df['market_cap_change'] = df['market_cap'].pct_change(1)
            self.feature_names.append('market_cap_change')
            
            # Market cap to price ratio (circulating supply indicator)
            df['supply_indicator'] = df['market_cap'] / df['price']
            self.feature_names.append('supply_indicator')
        
        return df
    
    def extract_all_features(self, limit: Optional[int] = None) -> pd.DataFrame:
        """Extract all price-based features"""
        
        # Reset feature names
        self.feature_names = []
        
        # Get price data
        df = self.get_price_data(limit=limit)
        
        if df.empty:
            return df
        
        # Apply all feature extraction methods
        df = self.calculate_moving_averages(df)
        df = self.calculate_price_ratios(df)
        df = self.calculate_volatility_indicators(df)
        df = self.calculate_momentum_indicators(df)
        df = self.calculate_trend_indicators(df)
        df = self.calculate_market_cap_features(df)
        
        return df
    
    def get_feature_names(self) -> List[str]:
        """Get list of all feature names created"""
        return self.feature_names.copy()