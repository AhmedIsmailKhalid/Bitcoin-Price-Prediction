"""Process sentiment for all news articles"""
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from src.data_processing.text_processing.sentiment_analyzer import SentimentAnalyzer
from src.shared.database import SessionLocal
from src.shared.logging import get_logger, setup_logging
from src.shared.models import NewsData, SentimentData


def process_all_sentiments():
    """Process sentiment for all news articles without sentiment data"""
    setup_logging()
    logger = get_logger(__name__)
    
    analyzer = SentimentAnalyzer()
    
    db = SessionLocal()
    try:
        # Find articles without sentiment data
        unprocessed = db.query(NewsData).outerjoin(SentimentData).filter(
            SentimentData.news_article_id.is_(None)
        ).all()
        
        logger.info(f"Found {len(unprocessed)} articles without sentiment data")
        
        processed_count = 0
        
        for article in unprocessed:
            try:
                logger.info(f"Processing sentiment for: {article.title[:60]}...")
                
                # Analyze comprehensive sentiment
                sentiment_scores = analyzer.analyze_comprehensive(article.content or article.title)
                
                # Analyze title vs content
                comparison = analyzer.analyze_title_vs_content(
                    article.title, 
                    article.content or ""
                )
                
                # Create sentiment record
                sentiment_data = SentimentData(
                    news_article_id=article.id,
                    vader_compound=sentiment_scores['vader_compound'],
                    vader_positive=sentiment_scores['vader_positive'],
                    vader_neutral=sentiment_scores['vader_neutral'],
                    vader_negative=sentiment_scores['vader_negative'],
                    textblob_polarity=sentiment_scores['textblob_polarity'],
                    textblob_subjectivity=sentiment_scores['textblob_subjectivity'],
                    combined_sentiment=sentiment_scores['combined_sentiment'],
                    sentiment_category=sentiment_scores['sentiment_category'],
                    title_sentiment=comparison['title_sentiment'],
                    content_sentiment=comparison['content_sentiment'],
                    sentiment_divergence=comparison['sentiment_divergence'],
                    sentiment_alignment=comparison['sentiment_alignment']
                )
                
                db.add(sentiment_data)
                processed_count += 1
                
            except Exception as e:
                logger.error(f"Failed to process sentiment for article {article.id}: {e}")
                continue
        
        # Commit all changes
        db.commit()
        logger.info(f"Successfully processed sentiment for {processed_count} articles")
        
        # Show sample results
        recent_sentiments = db.query(SentimentData).order_by(SentimentData.processed_at.desc()).limit(3).all()
        
        logger.info("\n=== Sample Sentiment Results ===")
        for sentiment in recent_sentiments:
            article = db.query(NewsData).filter(NewsData.id == sentiment.news_article_id).first()
            logger.info(f"Article: {article.title[:60]}...")
            logger.info(f"Combined Sentiment: {sentiment.combined_sentiment:.3f} ({sentiment.sentiment_category})")
            logger.info(f"VADER: {sentiment.vader_compound:.3f} | TextBlob: {sentiment.textblob_polarity:.3f}")
            logger.info("---")
        
        return True
        
    except Exception as e:
        db.rollback()
        logger.error(f"Failed to process sentiments: {e}")
        return False
        
    finally:
        db.close()


if __name__ == "__main__":
    success = process_all_sentiments()
    if success:
        print("\n✅ Sentiment processing completed successfully!")
    else:
        print("\n❌ Sentiment processing failed!")
    
    sys.exit(0 if success else 1)