import time
import logging
from datetime import datetime
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional

from sqlalchemy.orm import Session
from src.shared.database import SessionLocal
from src.shared.models import CollectionMetadata

class BaseCollector(ABC):
    """Abstract base class for all data collectors"""
    
    def __init__(self, name: str, collection_type: str):
        self.name = name
        self.collection_type = collection_type
        self.logger = logging.getLogger(f"{__name__}.{name}")
        
    @abstractmethod
    def collect_data(self) -> List[Dict[str, Any]]:
        """Collect data from the source. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def store_data(self, data: List[Dict[str, Any]], db: Session) -> int:
        """Store collected data to database. Must be implemented by subclasses."""
        pass
    
    def run_collection(self) -> bool:
        """Run the complete collection process with error handling and metadata tracking"""
        db = SessionLocal()
        start_time = datetime.utcnow()
        records_collected = 0
        
        # Create metadata record
        metadata = CollectionMetadata(
            collector_name=self.name,
            collection_type=self.collection_type,
            status="running",
            start_time=start_time
        )
        
        try:
            db.add(metadata)
            db.commit()
            
            self.logger.info(f"Starting {self.name} data collection...")
            
            # Collect data
            data = self.collect_data()
            
            if not data:
                self.logger.warning("No data collected")
                metadata.status = "success"
                metadata.records_collected = 0
            else:
                # Store data
                records_collected = self.store_data(data, db)
                
                metadata.status = "success"
                metadata.records_collected = records_collected
                
                self.logger.info(f"✅ Successfully collected and stored {records_collected} records")
            
            metadata.end_time = datetime.utcnow()
            db.commit()
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Collection failed: {e}")
            
            metadata.status = "error"
            metadata.error_message = str(e)[:500]  # Truncate long error messages
            metadata.end_time = datetime.utcnow()
            metadata.records_collected = records_collected
            
            try:
                db.commit()
            except:
                db.rollback()
            
            return False
            
        finally:
            db.close()
    
    def test_connection(self) -> bool:
        """Test connection to data source. Can be overridden by subclasses."""
        try:
            # Basic test - try to collect a small amount of data
            test_data = self.collect_data()
            self.logger.info(f"✅ Connection test successful - sample data: {len(test_data) if test_data else 0} records")
            return True
        except Exception as e:
            self.logger.error(f"❌ Connection test failed: {e}")
            return False