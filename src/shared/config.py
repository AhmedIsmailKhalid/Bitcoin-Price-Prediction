from typing import Optional
from dotenv import load_dotenv
from functools import lru_cache
from pydantic import PostgresDsn
from pydantic_settings import BaseSettings

# Load environment variables from .env.dev
load_dotenv('.env.dev')


class Settings(BaseSettings):
    # Database
    database_url: PostgresDsn
    
    # Reddit API
    reddit_client_id: str
    reddit_client_secret: str
    reddit_user_agent: str = "bitcoin-prediction-bot/1.0"
    
    # External APIs
    coingecko_api_url: str = "https://api.coingecko.com/api/v3"
    coingecko_api_key: Optional[str] = None
    cryptocompare_api_url: str = "https://min-api.cryptocompare.com/data"
    cryptocompare_api_key: Optional[str] = None
    
    # Application settings
    environment: str = "development"
    debug: bool = True
    log_level: str = "debug"
    
    # Data collection settings
    collection_interval_minutes: int = 15
    historical_data_months: int = 6
    
    # News sources
    coindesk_base_url: str = "https://www.coindesk.com"
    cryptoslate_base_url: str = "https://cryptoslate.com"
    
    # Add this to allow extra fields
    model_config = {
        "env_file": ".env.dev",
        "case_sensitive": False,
        "extra": "allow"  # This allows extra environment variables
    }


@lru_cache()
def get_settings() -> Settings:
    return Settings()


# Global settings instance
settings = get_settings()