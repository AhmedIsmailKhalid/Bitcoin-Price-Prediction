"""Schema validation for different data types"""
from typing import Dict, Any
from src.shared.logging import get_logger


class SchemaValidator:
    """Validate data against expected schemas"""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.schemas = {
            'news_article': {
                'required': ['title', 'content', 'url', 'data_source'],
                'optional': ['author', 'published_at', 'summary']
            },
            'price_data': {
                'required': ['price_usd', 'volume_24h', 'market_cap'],
                'optional': ['change_1h', 'change_24h', 'change_7d']
            },
            'social_post': {
                'required': ['content', 'platform', 'created_at'],
                'optional': ['upvotes', 'downvotes', 'author']
            }
        }
    
    def validate_against_schema(self, data: Dict[str, Any], schema_name: str) -> Dict[str, Any]:
        """Validate data against specified schema"""
        if schema_name not in self.schemas:
            return {'valid': False, 'errors': [f'Unknown schema: {schema_name}']}
        
        schema = self.schemas[schema_name]
        errors = []
        
        # Check required fields
        for field in schema['required']:
            if field not in data or data[field] is None:
                errors.append(f'Missing required field: {field}')
        
        return {'valid': len(errors) == 0, 'errors': errors}