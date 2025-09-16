"""News data processing utilities"""
from typing import Dict, Any
from src.shared.logging import get_logger


class NewsProcessor:
    """Process raw news data for analysis"""
    
    def __init__(self):
        self.logger = get_logger(__name__)
    
    def validate_article(self, article: Dict[str, Any]) -> bool:
        """Validate article has required fields"""
        required_fields = ['title', 'content', 'url']
        return all(field in article and article[field] for field in required_fields)
    
    def clean_content(self, content: str) -> str:
        """Basic content cleaning"""
        if not content:
            return ""
        
        # Remove excessive whitespace
        cleaned = " ".join(content.split())
        return cleaned.strip()
    
    def extract_metadata(self, article: Dict[str, Any]) -> Dict[str, Any]:
        """Extract metadata from article"""
        return {
            'word_count': len(article.get('content', '').split()),
            'title_length': len(article.get('title', '')),
            'has_author': bool(article.get('author')),
            'source': article.get('data_source', 'unknown')
        }