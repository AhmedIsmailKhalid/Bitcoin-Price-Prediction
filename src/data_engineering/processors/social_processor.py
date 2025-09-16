"""Social media data processing utilities"""
from typing import Dict, Any
from src.shared.logging import get_logger


class SocialProcessor:
    """Process social media data for analysis"""
    
    def __init__(self):
        self.logger = get_logger(__name__)
    
    def validate_social_post(self, post: Dict[str, Any]) -> bool:
        """Validate social post has required fields"""
        required_fields = ['content', 'platform', 'created_at']
        return all(field in post and post[field] for field in required_fields)
    
    def extract_engagement_metrics(self, post: Dict[str, Any]) -> Dict[str, int]:
        """Extract engagement metrics from social post"""
        return {
            'upvotes': post.get('upvotes', 0),
            'downvotes': post.get('downvotes', 0),
            'comments': post.get('comment_count', 0),
            'shares': post.get('share_count', 0)
        }