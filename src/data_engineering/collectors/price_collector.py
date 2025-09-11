import requests
from typing import Dict, List, Any
from sqlalchemy.orm import Session
from datetime import datetime
import time

from .base_collector import BaseCollector
from src.shared.config import settings
from src.shared.models import PriceData

class CoinGeckoCollector(BaseCollector):
    """Collector for CoinGecko price data"""
    
    def __init__(self):
        super().__init__("CoinGecko", "price")
        self.base_url = settings.coingecko_api_url
        self.api_key = settings.coingecko_api_key
        
        # Request headers
        self.headers = {
            "User-Agent": "bitcoin-prediction-bot/1.0"
        }
        
        if self.api_key:
            self.headers["x-cg-demo-api-key"] = self.api_key
    
    def collect_data(self) -> List[Dict[str, Any]]:
        """Collect cryptocurrency price data from CoinGecko"""
        
        # For now, focus on Bitcoin and a few major cryptocurrencies
        crypto_ids = ["bitcoin"]
        
        try:
            # CoinGecko API endpoint for multiple coins
            url = f"{self.base_url}/simple/price"
            
            params = {
                "ids": ",".join(crypto_ids),
                "vs_currencies": "usd",
                "include_market_cap": "true",
                "include_24hr_vol": "true",
                "include_24hr_change": "true",
                "include_1hr_change": "true",
                "include_7d_change": "true"
            }
            
            self.logger.debug(f"Making request to: {url}")
            self.logger.debug(f"Parameters: {params}")
            
            response = requests.get(url, headers=self.headers, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            self.logger.debug(f"Received data: {data}")
            
            # Transform data into our format
            collected_data = []
            
            for crypto_id, price_info in data.items():
                # Map CoinGecko IDs to symbols
                symbol_map = {
                    "bitcoin": "BTC",
                    "ethereum": "ETH", 
                    "binancecoin": "BNB",
                    "cardano": "ADA",
                    "solana": "SOL"
                }
                
                name_map = {
                    "bitcoin": "Bitcoin",
                    "ethereum": "Ethereum",
                    "binancecoin": "BNB",
                    "cardano": "Cardano", 
                    "solana": "Solana"
                }
                
                record = {
                    "symbol": symbol_map.get(crypto_id, crypto_id.upper()),
                    "name": name_map.get(crypto_id, crypto_id.title()),
                    "price_usd": price_info.get("usd"),
                    "market_cap": price_info.get("usd_market_cap"),
                    "volume_24h": price_info.get("usd_24h_vol"),
                    "change_1h": price_info.get("usd_1h_change"),
                    "change_24h": price_info.get("usd_24h_change"),
                    "change_7d": price_info.get("usd_7d_change"),
                    "data_source": "coingecko"
                }
                
                collected_data.append(record)
            
            self.logger.info(f"Collected data for {len(collected_data)} cryptocurrencies")
            return collected_data
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Request failed: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Data collection failed: {e}")
            raise
    
    def store_data(self, data: List[Dict[str, Any]], db: Session) -> int:
        """Store price data in the database"""
        
        stored_count = 0
        
        for record in data:
            try:
                price_data = PriceData(**record)
                db.add(price_data)
                stored_count += 1
                
            except Exception as e:
                self.logger.error(f"Failed to create PriceData record: {e}")
                self.logger.error(f"Record data: {record}")
                continue
        
        try:
            db.commit()
            self.logger.info(f"Stored {stored_count} price records")
            return stored_count
            
        except Exception as e:
            db.rollback()
            self.logger.error(f"Failed to commit price data: {e}")
            raise