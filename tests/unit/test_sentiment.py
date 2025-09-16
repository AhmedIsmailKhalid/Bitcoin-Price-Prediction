"""Unit tests for sentiment analysis"""
import pytest
from src.data_processing.text_processing.sentiment_analyzer import SentimentAnalyzer


class TestSentimentAnalyzer:
    """Test sentiment analysis functionality"""
    
    def test_analyzer_initialization(self):
        """Test analyzer can be initialized"""
        analyzer = SentimentAnalyzer()
        assert analyzer is not None
        # Check actual methods exist
        assert hasattr(analyzer, 'analyze_vader')
        assert hasattr(analyzer, 'analyze_textblob')
        assert hasattr(analyzer, 'analyze_comprehensive')
    
    def test_positive_sentiment_detection(self):
        """Test positive sentiment detection"""
        analyzer = SentimentAnalyzer()
        
        positive_text = "Bitcoin is performing excellently and showing great bullish momentum!"
        # Use actual method name: analyze_comprehensive
        result = analyzer.analyze_comprehensive(positive_text)
        
        assert result is not None
        assert 'combined_sentiment' in result
        # Expect positive sentiment
        assert result['combined_sentiment'] > 0
    
    def test_negative_sentiment_detection(self):
        """Test negative sentiment detection"""
        analyzer = SentimentAnalyzer()
        
        negative_text = "Bitcoin is crashing terribly and showing bearish signals everywhere!"
        result = analyzer.analyze_comprehensive(negative_text)
        
        assert result is not None
        assert 'combined_sentiment' in result
        # Expect negative sentiment
        assert result['combined_sentiment'] < 0
    
    def test_neutral_sentiment_detection(self):
        """Test neutral sentiment detection"""
        analyzer = SentimentAnalyzer()
        
        neutral_text = "Bitcoin price data is available in the market."
        result = analyzer.analyze_comprehensive(neutral_text)
        
        assert result is not None
        assert 'combined_sentiment' in result
        # Expect neutral sentiment (close to 0)
        assert abs(result['combined_sentiment']) < 0.5  # More lenient threshold
    
    def test_vader_analysis(self):
        """Test VADER sentiment analysis specifically"""
        analyzer = SentimentAnalyzer()
        
        text = "Bitcoin is great!"
        result = analyzer.analyze_vader(text)
        
        assert result is not None
        assert 'compound' in result
        assert 'positive' in result
        assert 'neutral' in result
        assert 'negative' in result
    
    def test_textblob_analysis(self):
        """Test TextBlob sentiment analysis specifically"""
        analyzer = SentimentAnalyzer()
        
        text = "Bitcoin is performing well today."
        result = analyzer.analyze_textblob(text)
        
        assert result is not None
        assert 'polarity' in result
        assert 'subjectivity' in result
    
    def test_title_vs_content_analysis(self):
        """Test title vs content sentiment comparison"""
        analyzer = SentimentAnalyzer()
        
        title = "Bitcoin Soars to New Heights"
        content = "Bitcoin has shown remarkable performance in recent trading sessions."
        
        result = analyzer.analyze_title_vs_content(title, content)
        
        assert result is not None
        assert 'title_sentiment' in result
        assert 'content_sentiment' in result
        assert 'sentiment_divergence' in result