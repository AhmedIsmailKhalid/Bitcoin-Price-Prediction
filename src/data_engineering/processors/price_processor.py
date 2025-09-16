"""Price data processing utilities"""
from typing import Dict, Any
from src.shared.logging import get_logger


class PriceProcessor:
    """Process raw price data for analysis"""
    
    def __init__(self):
        self.logger = get_logger(__name__)
    
    def validate_price_data(self, price_data: Dict[str, Any]) -> bool:
        """Validate price data has required fields"""
        required_fields = ['price_usd', 'volume_24h', 'market_cap']
        return all(field in price_data and price_data[field] is not None for field in required_fields)
    
    def calculate_metrics(self, price_data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate basic price metrics"""
        try:
            price = float(price_data.get('price_usd', 0))
            volume = float(price_data.get('volume_24h', 0))
            market_cap = float(price_data.get('market_cap', 0))
            
            return {
                'price_valid': price > 0,
                'volume_valid': volume >= 0,
                'market_cap_valid': market_cap > 0,
                'price_to_volume_ratio': price / (volume + 1e-10)  # Avoid division by zero
            }
        except (ValueError, TypeError):
            return {'price_valid': False, 'volume_valid': False, 'market_cap_valid': False}