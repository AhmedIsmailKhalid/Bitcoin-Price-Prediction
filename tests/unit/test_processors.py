"""Unit tests for data processors"""
from src.data_engineering.processors.news_processor import NewsProcessor
from src.data_engineering.processors.price_processor import PriceProcessor
from src.data_engineering.processors.social_processor import SocialProcessor


class TestNewsProcessor:
    """Test news data processing"""
    
    def test_news_processor_initialization(self):
        """Test processor can be initialized"""
        processor = NewsProcessor()
        assert processor is not None
        assert hasattr(processor, 'validate_article')
        assert hasattr(processor, 'clean_content')
        assert hasattr(processor, 'extract_metadata')
    
    def test_validate_article_valid(self):
        """Test article validation with valid article"""
        processor = NewsProcessor()
        
        valid_article = {
            'title': 'Bitcoin Rises to New Heights',
            'content': 'Bitcoin has shown remarkable performance in recent trading sessions.',
            'url': 'https://example.com/bitcoin-news',
            'data_source': 'test_source'
        }
        
        assert processor.validate_article(valid_article) is True
    
    def test_validate_article_invalid(self):
        """Test article validation with invalid article"""
        processor = NewsProcessor()
        
        invalid_article = {
            'title': 'Bitcoin News',
            # Missing content and url
            'data_source': 'test_source'
        }
        
        assert processor.validate_article(invalid_article) is False
    
    def test_clean_content(self):
        """Test content cleaning"""
        processor = NewsProcessor()
        
        messy_content = "   Bitcoin   has    shown   remarkable\n\nperformance   "
        cleaned = processor.clean_content(messy_content)
        
        assert cleaned == "Bitcoin has shown remarkable performance"
    
    def test_clean_content_empty(self):
        """Test cleaning empty content"""
        processor = NewsProcessor()
        
        assert processor.clean_content("") == ""
        assert processor.clean_content(None) == ""
    
    def test_extract_metadata(self):
        """Test metadata extraction"""
        processor = NewsProcessor()
        
        article = {
            'title': 'Bitcoin Market Analysis',
            'content': 'This is a detailed analysis of Bitcoin market trends.',
            'author': 'Test Author',
            'data_source': 'test_source'
        }
        
        metadata = processor.extract_metadata(article)
        
        assert 'word_count' in metadata
        assert 'title_length' in metadata
        assert 'has_author' in metadata
        assert 'source' in metadata
        
        assert metadata['word_count'] == 9  # Word count of content
        assert metadata['has_author'] is True
        assert metadata['source'] == 'test_source'


class TestPriceProcessor:
    """Test price data processing"""
    
    def test_price_processor_initialization(self):
        """Test processor can be initialized"""
        processor = PriceProcessor()
        assert processor is not None
        assert hasattr(processor, 'validate_price_data')
        assert hasattr(processor, 'calculate_metrics')
    
    def test_validate_price_data_valid(self):
        """Test price data validation with valid data"""
        processor = PriceProcessor()
        
        valid_price_data = {
            'price_usd': 45000.0,
            'volume_24h': 20000000000.0,
            'market_cap': 850000000000.0
        }
        
        assert processor.validate_price_data(valid_price_data) is True
    
    def test_validate_price_data_invalid(self):
        """Test price data validation with invalid data"""
        processor = PriceProcessor()
        
        invalid_price_data = {
            'price_usd': 45000.0,
            # Missing volume_24h and market_cap
        }
        
        assert processor.validate_price_data(invalid_price_data) is False
    
    def test_calculate_metrics_valid(self):
        """Test metrics calculation with valid data"""
        processor = PriceProcessor()
        
        price_data = {
            'price_usd': 45000.0,
            'volume_24h': 20000000000.0,
            'market_cap': 850000000000.0
        }
        
        metrics = processor.calculate_metrics(price_data)
        
        assert 'price_valid' in metrics
        assert 'volume_valid' in metrics
        assert 'market_cap_valid' in metrics
        assert 'price_to_volume_ratio' in metrics
        
        assert metrics['price_valid'] is True
        assert metrics['volume_valid'] is True
        assert metrics['market_cap_valid'] is True
        assert isinstance(metrics['price_to_volume_ratio'], float)
    
    def test_calculate_metrics_invalid_data(self):
        """Test metrics calculation with invalid data types"""
        processor = PriceProcessor()
        
        invalid_price_data = {
            'price_usd': 'invalid',
            'volume_24h': None,
            'market_cap': 'also invalid'
        }
        
        metrics = processor.calculate_metrics(invalid_price_data)
        
        assert metrics['price_valid'] is False
        assert metrics['volume_valid'] is False
        assert metrics['market_cap_valid'] is False


class TestSocialProcessor:
    """Test social media data processing"""
    
    def test_social_processor_initialization(self):
        """Test processor can be initialized"""
        processor = SocialProcessor()
        assert processor is not None
        assert hasattr(processor, 'validate_social_post')
        assert hasattr(processor, 'extract_engagement_metrics')
    
    def test_validate_social_post_valid(self):
        """Test social post validation with valid post"""
        processor = SocialProcessor()
        
        valid_post = {
            'content': 'Bitcoin is performing well today!',
            'platform': 'reddit',
            'created_at': '2025-01-01T12:00:00Z'
        }
        
        assert processor.validate_social_post(valid_post) is True
    
    def test_validate_social_post_invalid(self):
        """Test social post validation with invalid post"""
        processor = SocialProcessor()
        
        invalid_post = {
            'content': 'Bitcoin news',
            # Missing platform and created_at
        }
        
        assert processor.validate_social_post(invalid_post) is False
    
    def test_extract_engagement_metrics(self):
        """Test engagement metrics extraction"""
        processor = SocialProcessor()
        
        post = {
            'upvotes': 150,
            'downvotes': 10,
            'comment_count': 25,
            'share_count': 5
        }
        
        metrics = processor.extract_engagement_metrics(post)
        
        assert metrics['upvotes'] == 150
        assert metrics['downvotes'] == 10
        assert metrics['comments'] == 25
        assert metrics['shares'] == 5
    
    def test_extract_engagement_metrics_missing_data(self):
        """Test engagement metrics with missing data"""
        processor = SocialProcessor()
        
        post = {}  # Empty post
        
        metrics = processor.extract_engagement_metrics(post)
        
        # Should default to 0 for missing metrics
        assert metrics['upvotes'] == 0
        assert metrics['downvotes'] == 0
        assert metrics['comments'] == 0
        assert metrics['shares'] == 0