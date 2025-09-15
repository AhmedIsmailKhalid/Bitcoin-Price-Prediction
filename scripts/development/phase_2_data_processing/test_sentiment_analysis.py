"""Test sentiment analysis on collected news articles"""
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from src.data_processing.text_processing.sentiment_analyzer import SentimentAnalyzer
from src.shared.database import SessionLocal
from src.shared.logging import get_logger, setup_logging
from src.shared.models import NewsData


def test_sentiment_analysis():
    """Test sentiment analysis on collected articles"""
    setup_logging()
    logger = get_logger(__name__)
    
    # Initialize sentiment analyzer
    analyzer = SentimentAnalyzer()
    
    db = SessionLocal()
    try:
        # Get recent articles
        articles = db.query(NewsData).order_by(NewsData.collected_at.desc()).limit(5).all()
        
        if not articles:
            logger.error("No articles found in database")
            return False
        
        logger.info(f"Testing sentiment analysis on {len(articles)} articles")
        
        for i, article in enumerate(articles, 1):
            logger.info(f"\n=== Article {i}: {article.title[:60]}... ===")
            
            # Analyze title sentiment
            title_sentiment = analyzer.analyze_comprehensive(article.title)
            
            # Analyze content sentiment
            content_sentiment = analyzer.analyze_comprehensive(article.content or "")
            
            # Compare title vs content
            comparison = analyzer.analyze_title_vs_content(article.title, article.content or "")
            
            # Display results
            logger.info(f"Title Sentiment: {title_sentiment['combined_sentiment']:.3f} ({title_sentiment['sentiment_category']})")
            logger.info(f"Content Sentiment: {content_sentiment['combined_sentiment']:.3f} ({content_sentiment['sentiment_category']})")
            logger.info(f"VADER Compound: {content_sentiment['vader_compound']:.3f}")
            logger.info(f"TextBlob Polarity: {content_sentiment['textblob_polarity']:.3f}")
            logger.info(f"Sentiment Alignment: {comparison['sentiment_alignment']}")
        
        logger.info("\n✅ Sentiment analysis test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Sentiment analysis test failed: {e}")
        return False
        
    finally:
        db.close()


if __name__ == "__main__":
    success = test_sentiment_analysis()
    if success:
        print("\n✅ Sentiment analysis working correctly!")
    else:
        print("\n❌ Sentiment analysis test failed!")
    
    sys.exit(0 if success else 1)