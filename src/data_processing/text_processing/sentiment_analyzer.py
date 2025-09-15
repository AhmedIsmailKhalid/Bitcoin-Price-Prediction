import nltk
from typing import Dict
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from .preprocessor import TextPreprocessor


class SentimentAnalyzer:
    """Multi-method sentiment analysis for news articles"""
    
    def __init__(self):
        self.preprocessor = TextPreprocessor()
        self.vader_analyzer = SentimentIntensityAnalyzer()
        self.download_textblob_data()
    
    def download_textblob_data(self):
        """Download required TextBlob corpora"""
        try:
            # Try to access the corpora
            TextBlob("test").sentiment
        except Exception:
            # Download if not available
            import ssl
            try:
                _create_unverified_https_context = ssl._create_unverified_context
            except AttributeError:
                pass
            else:
                ssl._create_default_https_context = _create_unverified_https_context
            
            try:
                nltk.download('brown', quiet=True)
                nltk.download('punkt', quiet=True)
            except Exception:
                pass  # Continue if download fails
    
    def analyze_vader(self, text: str) -> Dict[str, float]:
        """Analyze sentiment using VADER"""
        
        if not text:
            return {
                'compound': 0.0,
                'positive': 0.0,
                'neutral': 0.0,
                'negative': 0.0
            }
        
        # Preprocess for sentiment (light cleaning)
        processed_text = self.preprocessor.preprocess_for_sentiment(text)
        
        # VADER analysis
        scores = self.vader_analyzer.polarity_scores(processed_text)
        
        return {
            'compound': scores['compound'],
            'positive': scores['pos'],
            'neutral': scores['neu'],
            'negative': scores['neg']
        }
    
    def analyze_textblob(self, text: str) -> Dict[str, float]:
        """Analyze sentiment using TextBlob"""
        
        if not text:
            return {
                'polarity': 0.0,
                'subjectivity': 0.0
            }
        
        try:
            # Preprocess for sentiment
            processed_text = self.preprocessor.preprocess_for_sentiment(text)
            
            # TextBlob analysis
            blob = TextBlob(processed_text)
            sentiment = blob.sentiment
            
            return {
                'polarity': sentiment.polarity,      # -1 (negative) to 1 (positive)
                'subjectivity': sentiment.subjectivity  # 0 (objective) to 1 (subjective)
            }
        
        except Exception:
            # Fallback if TextBlob fails
            return {
                'polarity': 0.0,
                'subjectivity': 0.0
            }
    
    def analyze_comprehensive(self, text: str) -> Dict[str, float]:
        """Comprehensive sentiment analysis using multiple methods"""
        
        # Get VADER scores
        vader_scores = self.analyze_vader(text)
        
        # Get TextBlob scores
        textblob_scores = self.analyze_textblob(text)
        
        # Create comprehensive result
        result = {
            # VADER scores
            'vader_compound': vader_scores['compound'],
            'vader_positive': vader_scores['positive'],
            'vader_neutral': vader_scores['neutral'],
            'vader_negative': vader_scores['negative'],
            
            # TextBlob scores
            'textblob_polarity': textblob_scores['polarity'],
            'textblob_subjectivity': textblob_scores['subjectivity'],
        }
        
        # Create combined sentiment score (weighted average)
        # VADER compound is already normalized -1 to 1
        # TextBlob polarity is also -1 to 1
        # Take weighted average favoring VADER for news text
        result['combined_sentiment'] = (
            0.6 * vader_scores['compound'] + 
            0.4 * textblob_scores['polarity']
        )
        
        # Create sentiment categories
        result['sentiment_category'] = self._categorize_sentiment(result['combined_sentiment'])
        
        return result
    
    def _categorize_sentiment(self, sentiment_score: float) -> str:
        """Categorize sentiment score into discrete categories"""
        
        if sentiment_score >= 0.1:
            return 'positive'
        elif sentiment_score <= -0.1:
            return 'negative'
        else:
            return 'neutral'
    
    def analyze_title_vs_content(self, title: str, content: str) -> Dict[str, any]:
        """Compare sentiment between title and content"""
        
        title_sentiment = self.analyze_comprehensive(title)
        content_sentiment = self.analyze_comprehensive(content)
        
        # Calculate sentiment divergence
        title_score = title_sentiment['combined_sentiment']
        content_score = content_sentiment['combined_sentiment']
        
        return {
            'title_sentiment': title_score,
            'content_sentiment': content_score,
            'sentiment_divergence': abs(title_score - content_score),
            'title_more_positive': title_score > content_score,
            'sentiment_alignment': self._categorize_sentiment(title_score) == self._categorize_sentiment(content_score)
        }