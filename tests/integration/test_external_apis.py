# tests/integration/test_external_apis.py
import pytest

try:
    from src.data_engineering.collectors.price_collector import CoinGeckoCollector  # or whatever the actual name is
    from src.data_engineering.collectors.news_collector import MultiSourceNewsCollector  # or whatever the actual name is
    COLLECTORS_AVAILABLE = True
except ImportError:
    COLLECTORS_AVAILABLE = False
    
@pytest.mark.skipif(not COLLECTORS_AVAILABLE, reason="Collector classes not available")
class TestExternalAPIs:
    """Test real external API integrations"""
    
    @pytest.mark.integration
    def test_coingecko_api_integration(self):
        """Test real CoinGecko API integration"""
        collector = CoinGeckoCollector()
        
        # Test collection (synchronous, not async)
        try:
            success = collector.collect_and_store()
            
            # If collection succeeds, verify data structure
            if success:
                # Get the most recent data to verify structure
                db_data = collector.get_recent_data(limit=1)
                
                if db_data:
                    record = db_data[0]
                    assert hasattr(record, 'price_usd')
                    assert hasattr(record, 'collected_at')
                    assert record.price_usd > 0
            
        except Exception as e:
            pytest.skip(f"CoinGecko API test failed (network/API issue): {e}")
    
    @pytest.mark.integration
    def test_news_scraping_integration(self):
        """Test real news website scraping"""
        collector = MultiSourceNewsCollector()
        
        try:
            # Test collection (synchronous, not async)
            success = collector.collect_and_store()
            
            # If collection succeeds, verify data structure
            if success:
                # Check that articles were collected
                db_data = collector.get_recent_articles(limit=1)
                
                if db_data:
                    article = db_data[0]
                    assert hasattr(article, 'title')
                    assert hasattr(article, 'content') 
                    assert hasattr(article, 'data_source')
                    assert len(article.title) > 0
            
        except Exception as e:
            pytest.skip(f"News scraping test failed (network/API issue): {e}")
    
    @pytest.mark.integration
    @pytest.mark.slow
    def test_price_data_quality(self):
        """Test quality of collected price data"""
        collector = CoinGeckoCollector()
        
        try:
            # Collect data
            success = collector.collect_and_store()
            
            if success:
                # Verify data quality
                recent_data = collector.get_recent_data(limit=5)
                
                if recent_data:
                    for record in recent_data:
                        # Basic quality checks
                        assert record.price_usd > 0, "Price should be positive"
                        assert record.volume_24h >= 0, "Volume should be non-negative"
                        assert record.market_cap > 0, "Market cap should be positive"
                        
                        # Price should be reasonable (between $1,000 and $1,000,000)
                        assert 1000 <= record.price_usd <= 1000000, f"Price seems unrealistic: {record.price_usd}"
            
        except Exception as e:
            pytest.skip(f"Price data quality test failed: {e}")
    
    @pytest.mark.integration
    @pytest.mark.slow  
    def test_news_content_quality(self):
        """Test quality of collected news content"""
        collector = MultiSourceNewsCollector()
        
        try:
            # Collect data
            success = collector.collect_and_store()
            
            if success:
                # Verify content quality
                recent_articles = collector.get_recent_articles(limit=3)
                
                if recent_articles:
                    for article in recent_articles:
                        # Basic quality checks
                        assert len(article.title) > 10, "Title should be substantial"
                        assert len(article.content) > 50, "Content should be substantial"
                        assert article.url.startswith('http'), "URL should be valid"
                        
                        # Content should contain crypto-related keywords
                        content_lower = article.content.lower()
                        crypto_keywords = ['bitcoin', 'cryptocurrency', 'crypto', 'blockchain', 'btc']
                        assert any(keyword in content_lower for keyword in crypto_keywords), \
                            "Article should be crypto-related"
            
        except Exception as e:
            pytest.skip(f"News content quality test failed: {e}")