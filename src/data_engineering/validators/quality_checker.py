"""Data quality validation utilities"""
from datetime import datetime
from typing import Dict, List, Any, Tuple
from src.shared.logging import get_logger


class DataQualityChecker:
    """Check data quality across different data types"""
    
    def __init__(self):
        self.logger = get_logger(__name__)
    
    def check_data_freshness(self, timestamp: datetime, max_age_hours: int = 24) -> Tuple[bool, str]:
        """Check if data is fresh enough"""
        if not timestamp:
            return False, "No timestamp provided"
        
        age = datetime.utcnow() - timestamp.replace(tzinfo=None)
        is_fresh = age.total_seconds() / 3600 <= max_age_hours
        
        return is_fresh, f"Data age: {age.total_seconds() / 3600:.1f} hours"
    
    def check_completeness(self, data: Dict[str, Any], required_fields: List[str]) -> Tuple[float, List[str]]:
        """Check data completeness"""
        missing_fields = [field for field in required_fields if not data.get(field)]
        completeness = (len(required_fields) - len(missing_fields)) / len(required_fields)
        
        return completeness, missing_fields
    
    def check_data_consistency(self, data_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Check consistency across data records"""
        if not data_list:
            return {'consistent': True, 'issues': []}
        
        issues = []
        
        # Check field consistency
        first_keys = set(data_list[0].keys())
        for i, record in enumerate(data_list[1:], 1):
            if set(record.keys()) != first_keys:
                issues.append(f"Record {i} has different fields")
        
        return {'consistent': len(issues) == 0, 'issues': issues}