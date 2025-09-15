"""Generate sentiment analysis summary and insights"""
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from src.shared.database import SessionLocal
from src.shared.logging import get_logger, setup_logging
from src.shared.models import NewsData, SentimentData


def generate_sentiment_summary():
    """Generate comprehensive sentiment analysis summary"""
    setup_logging()
    logger = get_logger(__name__)
    
    db = SessionLocal()
    try:
        # Get all sentiment data
        sentiments = db.query(SentimentData).all()
        
        if not sentiments:
            logger.error("No sentiment data found")
            return False
        
        print(f"\n=== SENTIMENT ANALYSIS SUMMARY ===")
        print(f"Total articles analyzed: {len(sentiments)}")
        
        # Calculate statistics
        sentiment_scores = [s.combined_sentiment for s in sentiments]
        avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)
        
        positive_count = len([s for s in sentiments if s.sentiment_category == 'positive'])
        negative_count = len([s for s in sentiments if s.sentiment_category == 'negative'])
        neutral_count = len([s for s in sentiments if s.sentiment_category == 'neutral'])
        
        print(f"\n=== OVERALL SENTIMENT ===")
        print(f"Average sentiment score: {avg_sentiment:.3f}")
        print(f"Positive articles: {positive_count} ({positive_count/len(sentiments)*100:.1f}%)")
        print(f"Negative articles: {negative_count} ({negative_count/len(sentiments)*100:.1f}%)")
        print(f"Neutral articles: {neutral_count} ({neutral_count/len(sentiments)*100:.1f}%)")
        
        # Sentiment by source
        print(f"\n=== SENTIMENT BY NEWS SOURCE ===")
        source_sentiment = {}
        
        for sentiment in sentiments:
            article = db.query(NewsData).filter(NewsData.id == sentiment.news_article_id).first()
            source = article.data_source
            
            if source not in source_sentiment:
                source_sentiment[source] = []
            source_sentiment[source].append(sentiment.combined_sentiment)
        
        for source, scores in source_sentiment.items():
            avg_score = sum(scores) / len(scores)
            print(f"{source}: {avg_score:.3f} (n={len(scores)})")
        
        # Title vs Content alignment
        aligned_count = len([s for s in sentiments if s.sentiment_alignment])
        alignment_rate = aligned_count / len(sentiments) * 100
        
        print(f"\n=== TITLE vs CONTENT ANALYSIS ===")
        print(f"Sentiment alignment rate: {alignment_rate:.1f}%")
        print(f"Articles with aligned sentiment: {aligned_count}/{len(sentiments)}")
        
        # Most positive and negative articles
        most_positive = max(sentiments, key=lambda s: s.combined_sentiment)
        most_negative = min(sentiments, key=lambda s: s.combined_sentiment)
        
        pos_article = db.query(NewsData).filter(NewsData.id == most_positive.news_article_id).first()
        neg_article = db.query(NewsData).filter(NewsData.id == most_negative.news_article_id).first()
        
        print(f"\n=== SENTIMENT EXTREMES ===")
        print(f"Most positive ({most_positive.combined_sentiment:.3f}): {pos_article.title[:60]}...")
        print(f"Most negative ({most_negative.combined_sentiment:.3f}): {neg_article.title[:60]}...")
        
        print(f"\n✅ Sentiment analysis complete - ready for ML feature engineering!")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to generate sentiment summary: {e}")
        return False
        
    finally:
        db.close()


if __name__ == "__main__":
    generate_sentiment_summary()