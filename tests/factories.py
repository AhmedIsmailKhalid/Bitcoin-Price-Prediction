# tests/factories.py - Test data factories
import random
from datetime import datetime, timedelta
from typing import List, Dict, Any

class NewsArticleFactory:
    """Factory for creating mock news articles"""
    
    @staticmethod
    def create() -> Dict[str, Any]:
        """Create a single mock news article"""
        titles = [
            "Bitcoin reaches new all-time high amid institutional adoption",
            "Cryptocurrency market shows bearish signals as regulations tighten", 
            "Major exchange announces Bitcoin ETF approval",
            "Technical analysis suggests Bitcoin consolidation phase",
            "Central bank digital currencies gain momentum worldwide"
        ]
        
        contents = [
            "The cryptocurrency market experienced significant volatility today as Bitcoin reached unprecedented levels. Market analysts attribute this movement to increased institutional adoption and regulatory clarity.",
            "Bearish sentiment dominates the crypto space as regulatory concerns mount. Investors are exercising caution amid uncertainty about future policy directions.",
            "Positive developments in the Bitcoin ETF space have energized traders. The approval represents a major milestone for cryptocurrency mainstream adoption.",
            "Technical indicators suggest Bitcoin is entering a consolidation phase after recent gains. Support levels remain strong despite profit-taking pressure.",
            "Central bank digital currencies are reshaping the financial landscape. This development could impact traditional cryptocurrencies in the coming years."
        ]
        
        sources = ["coindesk_rss", "cointelegraph_rss", "decrypt_rss"]
        
        return {
            "id": random.randint(1, 1000),
            "title": random.choice(titles),
            "content": random.choice(contents),
            "url": f"https://example.com/article-{random.randint(1, 1000)}",
            "data_source": random.choice(sources),
            "published_at": datetime.now() - timedelta(hours=random.randint(1, 24)),
            "word_count": random.randint(100, 500)
        }
    
    @classmethod
    def create_batch(cls, count: int) -> List[Dict[str, Any]]:
        """Create multiple mock news articles"""
        return [cls.create() for _ in range(count)]
    
    # Allow calling the class directly
    def __call__(self):
        return self.create()

class FeatureSetFactory:
    """Factory for creating mock feature sets"""
    
    @staticmethod
    def create() -> Dict[str, float]:
        """Create mock feature set for ML predictions"""
        return {
            # Price features
            "price": 45000.0 + random.uniform(-5000, 5000),
            "sma_5": 44000.0 + random.uniform(-3000, 3000),
            "ema_5": 44500.0 + random.uniform(-3000, 3000),
            "volatility_5": random.uniform(0.01, 0.05),
            "price_change_1": random.uniform(-0.1, 0.1),
            
            # Temporal features
            "hour": random.randint(0, 23),
            "day_of_week": random.randint(0, 6),
            "is_weekend": random.choice([0, 1]),
            "hour_sin": random.uniform(-1, 1),
            "hour_cos": random.uniform(-1, 1),
            
            # Sentiment features
            "sentiment_score_mean": random.uniform(-1, 1),
            "vader_compound_mean": random.uniform(-1, 1),
            "textblob_polarity_mean": random.uniform(-1, 1),
            "sentiment_alignment_mean": random.uniform(0, 1)
        }
    
    def __call__(self):
        return self.create()

# Create singleton instances that can be called directly
NewsArticleFactory = NewsArticleFactory()
FeatureSetFactory = FeatureSetFactory()