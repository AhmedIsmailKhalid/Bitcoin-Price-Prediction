from sqlalchemy.sql import func
from ..shared.database import Base
from sqlalchemy import Boolean, Column, ForeignKey, Integer, String, Float, DateTime, Index, Text


class PriceData(Base):
    __tablename__ = "price_data"
    
    # Primary key
    id = Column(Integer, primary_key=True, index=True)
    
    # Cryptocurrency identifier
    symbol = Column(String(10), nullable=False)  # BTC, ETH, etc.
    name = Column(String(50), nullable=False)    # Bitcoin, Ethereum, etc.
    
    # Price data
    price_usd = Column(Float, nullable=False)
    market_cap = Column(Float, nullable=True)
    volume_24h = Column(Float, nullable=True)
    
    # Price changes
    change_1h = Column(Float, nullable=True)
    change_24h = Column(Float, nullable=True)
    change_7d = Column(Float, nullable=True)
    
    # Metadata
    data_source = Column(String(50), nullable=False)  # coingecko, cryptocompare
    collected_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Create indexes for common queries
    __table_args__ = (
        Index('idx_symbol_collected', 'symbol', 'collected_at'),
        Index('idx_source_collected', 'data_source', 'collected_at'),
    )
    
    def __repr__(self):
        return f"<PriceData(symbol={self.symbol}, price=${self.price_usd}, collected_at={self.collected_at})>"


class CollectionMetadata(Base):
    __tablename__ = "collection_metadata"
    
    # Primary key
    id = Column(Integer, primary_key=True, index=True)
    
    # Collection info
    collector_name = Column(String(100), nullable=False)
    collection_type = Column(String(50), nullable=False)  # price, news, social
    status = Column(String(20), nullable=False)           # success, error, partial
    
    # Metrics
    records_collected = Column(Integer, default=0)
    start_time = Column(DateTime(timezone=True), nullable=False)
    end_time = Column(DateTime(timezone=True), nullable=True)
    
    # Error tracking
    error_message = Column(String(500), nullable=True)
    
    def __repr__(self):
        return f"<CollectionMetadata(collector={self.collector_name}, status={self.status}, records={self.records_collected})>"


class NewsData(Base):
    __tablename__ = "news_data"
    
    # Primary key
    id = Column(Integer, primary_key=True, index=True)
    
    # Article identification
    title = Column(String(500), nullable=False)
    url = Column(String(1000), nullable=False, unique=True)  # Prevent duplicates
    
    # Content
    content = Column(Text, nullable=False)  # Full article text
    summary = Column(String(1000), nullable=True)  # Article summary/excerpt
    
    # Metadata
    author = Column(String(200), nullable=True)
    published_at = Column(DateTime(timezone=True), nullable=True)  # Article publish date
    data_source = Column(String(50), nullable=False)  # coindesk, cryptoslate, etc.
    
    # Processing metadata
    collected_at = Column(DateTime(timezone=True), server_default=func.now())
    word_count = Column(Integer, nullable=True)
    
    # Create indexes for common queries
    __table_args__ = (
        Index('idx_news_published', 'published_at'),
        Index('idx_news_source_collected', 'data_source', 'collected_at'),
        Index('idx_news_url', 'url'),  # For duplicate prevention
    )
    
    def __repr__(self):
        return f"<NewsData(title='{self.title[:50]}...', source={self.data_source}, published={self.published_at})>"
    
    
class SentimentData(Base):
    __tablename__ = "sentiment_data"
    
    # Primary key
    id = Column(Integer, primary_key=True, index=True)
    
    # Foreign key to news article
    news_article_id = Column(Integer, ForeignKey('news_data.id'), nullable=False)
    
    # VADER sentiment scores
    vader_compound = Column(Float, nullable=False)
    vader_positive = Column(Float, nullable=False)
    vader_neutral = Column(Float, nullable=False)
    vader_negative = Column(Float, nullable=False)
    
    # TextBlob sentiment scores
    textblob_polarity = Column(Float, nullable=False)
    textblob_subjectivity = Column(Float, nullable=False)
    
    # Combined scores
    combined_sentiment = Column(Float, nullable=False)
    sentiment_category = Column(String(20), nullable=False)  # positive, negative, neutral
    
    # Title vs content analysis
    title_sentiment = Column(Float, nullable=True)
    content_sentiment = Column(Float, nullable=True)
    sentiment_divergence = Column(Float, nullable=True)
    sentiment_alignment = Column(Boolean, nullable=True)
    
    # Processing metadata
    processed_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Create indexes
    __table_args__ = (
        Index('idx_sentiment_article', 'news_article_id'),
        Index('idx_sentiment_category', 'sentiment_category'),
        Index('idx_sentiment_score', 'combined_sentiment'),
    )
    
    def __repr__(self):
        return f"<SentimentData(article_id={self.news_article_id}, sentiment={self.combined_sentiment:.3f}, category={self.sentiment_category})>"