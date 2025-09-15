import numpy as np
from typing import Dict
from datetime import datetime

from src.shared.database import SessionLocal
from src.shared.models import PriceData, NewsData, SentimentData


class DataQualityValidator:
    """Validate data quality for ML pipeline"""
    
    def __init__(self):
        self.validation_results = {}
    
    def validate_price_data(self) -> Dict[str, any]:
        """Validate price data quality"""
        
        db = SessionLocal()
        try:
            price_records = db.query(PriceData).all()
            
            results = {
                'total_records': len(price_records),
                'issues': [],
                'recommendations': []
            }
            
            if len(price_records) == 0:
                results['issues'].append("No price data available")
                results['recommendations'].append("Collect Bitcoin price data using price collector")
                return results
            
            # Check for minimum data requirements
            if len(price_records) < 10:
                results['issues'].append(f"Insufficient price data: {len(price_records)} records (need ≥10 for basic ML)")
                results['recommendations'].append("Collect more price data over time for better features")
            
            # Check for data freshness
            latest_record = max(price_records, key=lambda x: x.collected_at)
            hours_since_latest = (datetime.utcnow() - latest_record.collected_at.replace(tzinfo=None)).total_seconds() / 3600
            
            if hours_since_latest > 24:
                results['issues'].append(f"Price data is {hours_since_latest:.1f} hours old")
                results['recommendations'].append("Run price collection to get fresh data")
            
            # Check for missing values
            missing_fields = []
            for record in price_records:
                if record.price_usd is None:
                    missing_fields.append('price_usd')
                if record.volume_24h is None:
                    missing_fields.append('volume_24h')
            
            if missing_fields:
                unique_missing = list(set(missing_fields))
                results['issues'].append(f"Missing values in fields: {unique_missing}")
            
            return results
            
        finally:
            db.close()
    
    def validate_news_data(self) -> Dict[str, any]:
        """Validate news data quality"""
        
        db = SessionLocal()
        try:
            news_records = db.query(NewsData).all()
            
            results = {
                'total_records': len(news_records),
                'issues': [],
                'recommendations': []
            }
            
            if len(news_records) == 0:
                results['issues'].append("No news data available")
                results['recommendations'].append("Collect news data using news collector")
                return results
            
            # Check content quality
            short_content_count = 0
            missing_content_count = 0
            
            for record in news_records:
                if not record.content or len(record.content.strip()) == 0:
                    missing_content_count += 1
                elif len(record.content) < 100:
                    short_content_count += 1
            
            if missing_content_count > 0:
                results['issues'].append(f"{missing_content_count} articles have no content")
            
            if short_content_count > len(news_records) * 0.3:  # More than 30% short
                results['issues'].append(f"{short_content_count} articles have very short content (<100 chars)")
                results['recommendations'].append("Check news extraction - may need better content selectors")
            
            # Check source diversity
            sources = [record.data_source for record in news_records]
            unique_sources = list(set(sources))
            
            if len(unique_sources) < 2:
                results['issues'].append(f"Limited news source diversity: {len(unique_sources)} sources")
                results['recommendations'].append("Enable additional news sources for diverse perspectives")
            
            return results
            
        finally:
            db.close()
    
    def validate_sentiment_data(self) -> Dict[str, any]:
        """Validate sentiment analysis quality"""
        
        db = SessionLocal()
        try:
            sentiment_records = db.query(SentimentData).all()
            
            results = {
                'total_records': len(sentiment_records),
                'issues': [],
                'recommendations': []
            }
            
            if len(sentiment_records) == 0:
                results['issues'].append("No sentiment data available")
                results['recommendations'].append("Process sentiment analysis on news articles")
                return results
            
            # Check sentiment score distribution
            scores = [s.combined_sentiment for s in sentiment_records]
            avg_sentiment = np.mean(scores)
            sentiment_std = np.std(scores)
            
            # Check for reasonable sentiment distribution
            if sentiment_std < 0.1:
                results['issues'].append(f"Low sentiment variance (σ={sentiment_std:.3f}) - may indicate poor sentiment detection")
            
            # Check for extreme bias
            if abs(avg_sentiment) > 0.5:
                results['issues'].append(f"Highly biased sentiment distribution (μ={avg_sentiment:.3f})")
                results['recommendations'].append("Review news sources for balanced coverage")
            
            # Check sentiment categories
            categories = [s.sentiment_category for s in sentiment_records]
            category_counts = {cat: categories.count(cat) for cat in set(categories)}
            
            if len(category_counts) < 2:
                results['issues'].append("Only one sentiment category detected")
                results['recommendations'].append("Check sentiment analysis thresholds")
            
            return results
            
        finally:
            db.close()
    
    def validate_data_alignment(self) -> Dict[str, any]:
        """Validate data alignment across price, news, and sentiment"""
        
        db = SessionLocal()
        try:
            # Get data counts
            price_count = db.query(PriceData).count()
            news_count = db.query(NewsData).count()
            sentiment_count = db.query(SentimentData).count()
            
            results = {
                'price_records': price_count,
                'news_records': news_count,
                'sentiment_records': sentiment_count,
                'issues': [],
                'recommendations': []
            }
            
            # Check data balance
            if sentiment_count < news_count:
                missing_sentiment = news_count - sentiment_count
                results['issues'].append(f"{missing_sentiment} news articles lack sentiment analysis")
                results['recommendations'].append("Run sentiment processing on all news articles")
            
            if price_count == 0:
                results['issues'].append("No price data for feature alignment")
                results['recommendations'].append("Collect price data for ML training")
            
            # Check time coverage overlap
            if price_count > 0 and news_count > 0:
                price_times = [p.collected_at for p in db.query(PriceData).all()]
                news_times = [n.published_at or n.collected_at for n in db.query(NewsData).all()]
                
                price_range = (min(price_times), max(price_times))
                news_range = (min(news_times), max(news_times))
                
                # Check for time overlap
                overlap_start = max(price_range[0], news_range[0])
                overlap_end = min(price_range[1], news_range[1])
                
                if overlap_start >= overlap_end:
                    results['issues'].append("No time overlap between price and news data")
                    results['recommendations'].append("Collect data from overlapping time periods")
            
            return results
            
        finally:
            db.close()
    
    def generate_comprehensive_report(self) -> Dict[str, any]:
        """Generate comprehensive data quality report"""
        
        report = {
            'timestamp': datetime.utcnow().isoformat(),
            'price_validation': self.validate_price_data(),
            'news_validation': self.validate_news_data(),
            'sentiment_validation': self.validate_sentiment_data(),
            'alignment_validation': self.validate_data_alignment()
        }
        
        # Overall assessment
        total_issues = (
            len(report['price_validation']['issues']) +
            len(report['news_validation']['issues']) +
            len(report['sentiment_validation']['issues']) +
            len(report['alignment_validation']['issues'])
        )
        
        if total_issues == 0:
            report['overall_status'] = "EXCELLENT"
        elif total_issues <= 3:
            report['overall_status'] = "GOOD"
        elif total_issues <= 6:
            report['overall_status'] = "NEEDS_IMPROVEMENT"
        else:
            report['overall_status'] = "CRITICAL_ISSUES"
        
        return report