from sqlalchemy import Column, Integer, String, Float, DateTime, Index
from sqlalchemy.sql import func
from src.shared.database import Base

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