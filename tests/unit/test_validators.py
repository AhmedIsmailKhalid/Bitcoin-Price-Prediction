"""Unit tests for data validators"""
from datetime import datetime, timedelta
from src.data_engineering.validators.quality_checker import DataQualityChecker
from src.data_engineering.validators.schema_validator import SchemaValidator


class TestDataQualityChecker:
    """Test data quality checking"""
    
    def test_quality_checker_initialization(self):
        """Test checker can be initialized"""
        checker = DataQualityChecker()
        assert checker is not None
        assert hasattr(checker, 'check_data_freshness')
        assert hasattr(checker, 'check_completeness')
        assert hasattr(checker, 'check_data_consistency')
    
    def test_check_data_freshness_fresh(self):
        """Test freshness check with fresh data"""
        checker = DataQualityChecker()
        
        fresh_timestamp = datetime.utcnow() - timedelta(hours=1)
        is_fresh, message = checker.check_data_freshness(fresh_timestamp, max_age_hours=24)
        
        assert is_fresh is True
        assert "1.0 hours" in message
    
    def test_check_data_freshness_stale(self):
        """Test freshness check with stale data"""
        checker = DataQualityChecker()
        
        stale_timestamp = datetime.utcnow() - timedelta(hours=25)
        is_fresh, message = checker.check_data_freshness(stale_timestamp, max_age_hours=24)
        
        assert is_fresh is False
        assert "25.0 hours" in message
    
    def test_check_data_freshness_no_timestamp(self):
        """Test freshness check with no timestamp"""
        checker = DataQualityChecker()
        
        is_fresh, message = checker.check_data_freshness(None)
        
        assert is_fresh is False
        assert message == "No timestamp provided"
    
    def test_check_completeness_complete(self):
        """Test completeness check with complete data"""
        checker = DataQualityChecker()
        
        complete_data = {
            'field1': 'value1',
            'field2': 'value2',
            'field3': 'value3'
        }
        required_fields = ['field1', 'field2', 'field3']
        
        completeness, missing = checker.check_completeness(complete_data, required_fields)
        
        assert completeness == 1.0
        assert len(missing) == 0
    
    def test_check_completeness_partial(self):
        """Test completeness check with partial data"""
        checker = DataQualityChecker()
        
        partial_data = {
            'field1': 'value1',
            'field2': '',  # Empty value
            'field3': 'value3'
            # Missing field4
        }
        required_fields = ['field1', 'field2', 'field3', 'field4']
        
        completeness, missing = checker.check_completeness(partial_data, required_fields)
        
        assert completeness == 0.5  # 2 out of 4 fields valid
        assert 'field2' in missing  # Empty value
        assert 'field4' in missing  # Missing field
    
    def test_check_data_consistency_consistent(self):
        """Test consistency check with consistent data"""
        checker = DataQualityChecker()
        
        consistent_data = [
            {'field1': 'value1', 'field2': 'value2'},
            {'field1': 'value3', 'field2': 'value4'},
            {'field1': 'value5', 'field2': 'value6'}
        ]
        
        result = checker.check_data_consistency(consistent_data)
        
        assert result['consistent'] is True
        assert len(result['issues']) == 0
    
    def test_check_data_consistency_inconsistent(self):
        """Test consistency check with inconsistent data"""
        checker = DataQualityChecker()
        
        inconsistent_data = [
            {'field1': 'value1', 'field2': 'value2'},
            {'field1': 'value3', 'field3': 'value4'},  # Different field name
            {'field1': 'value5', 'field2': 'value6'}
        ]
        
        result = checker.check_data_consistency(inconsistent_data)
        
        assert result['consistent'] is False
        assert len(result['issues']) > 0
    
    def test_check_data_consistency_empty(self):
        """Test consistency check with empty data"""
        checker = DataQualityChecker()
        
        result = checker.check_data_consistency([])
        
        assert result['consistent'] is True
        assert len(result['issues']) == 0


class TestSchemaValidator:
    """Test schema validation"""
    
    def test_schema_validator_initialization(self):
        """Test validator can be initialized"""
        validator = SchemaValidator()
        assert validator is not None
        assert hasattr(validator, 'validate_against_schema')
        assert hasattr(validator, 'schemas')
    
    def test_validate_news_article_valid(self):
        """Test validation of valid news article"""
        validator = SchemaValidator()
        
        valid_article = {
            'title': 'Bitcoin News',
            'content': 'Bitcoin content here',
            'url': 'https://example.com',
            'data_source': 'test_source',
            'author': 'Test Author'
        }
        
        result = validator.validate_against_schema(valid_article, 'news_article')
        
        assert result['valid'] is True
        assert len(result['errors']) == 0
    
    def test_validate_news_article_invalid(self):
        """Test validation of invalid news article"""
        validator = SchemaValidator()
        
        invalid_article = {
            'title': 'Bitcoin News',
            # Missing required fields: content, url, data_source
            'author': 'Test Author'
        }
        
        result = validator.validate_against_schema(invalid_article, 'news_article')
        
        assert result['valid'] is False
        assert len(result['errors']) == 3  # Missing 3 required fields
        assert any('content' in error for error in result['errors'])
        assert any('url' in error for error in result['errors'])
        assert any('data_source' in error for error in result['errors'])
    
    def test_validate_price_data_valid(self):
        """Test validation of valid price data"""
        validator = SchemaValidator()
        
        valid_price_data = {
            'price_usd': 45000.0,
            'volume_24h': 20000000000.0,
            'market_cap': 850000000000.0,
            'change_1h': 1.5,
            'change_24h': -2.3,
            'change_7d': 5.7
        }
        
        result = validator.validate_against_schema(valid_price_data, 'price_data')
        
        assert result['valid'] is True
        assert len(result['errors']) == 0
    
    def test_validate_price_data_invalid(self):
        """Test validation of invalid price data"""
        validator = SchemaValidator()
        
        invalid_price_data = {
            'price_usd': 45000.0,
            # Missing required fields: volume_24h, market_cap
        }
        
        result = validator.validate_against_schema(invalid_price_data, 'price_data')
        
        assert result['valid'] is False
        assert len(result['errors']) == 2
    
    def test_validate_unknown_schema(self):
        """Test validation against unknown schema"""
        validator = SchemaValidator()
        
        data = {'field': 'value'}
        result = validator.validate_against_schema(data, 'unknown_schema')
        
        assert result['valid'] is False
        assert 'Unknown schema' in result['errors'][0]