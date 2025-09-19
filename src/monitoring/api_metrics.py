"""API performance metrics and analytics"""
import time
from functools import wraps
from typing import Dict, Any, Callable

from src.shared.logging import get_logger
from src.monitoring.model_monitor import ModelMonitor


class APIMetrics:
    """API performance metrics collector"""
    
    def __init__(self):
        self.model_monitor = ModelMonitor()
        self.logger = get_logger(__name__)
        
        # In-memory metrics (in production, use Redis or similar)
        self.request_counts = {}
        self.response_times = {}
        self.error_counts = {}
    
    def track_prediction(self, model_name: str, version: str, features: Dict[str, float],
                        prediction: int, probability: float, confidence: str,
                        response_time_ms: float):
        """Track a prediction request"""
        
        # Log to model monitor
        self.model_monitor.log_prediction(
            model_name, version, features, prediction, 
            probability, confidence, response_time_ms
        )
        
        # Update in-memory metrics
        endpoint = "/predict"
        self.request_counts[endpoint] = self.request_counts.get(endpoint, 0) + 1
        
        if endpoint not in self.response_times:
            self.response_times[endpoint] = []
        self.response_times[endpoint].append(response_time_ms)
        
        # Keep only last 1000 response times
        if len(self.response_times[endpoint]) > 1000:
            self.response_times[endpoint] = self.response_times[endpoint][-1000:]
    
    def track_error(self, endpoint: str, error_type: str):
        """Track an API error"""
        
        key = f"{endpoint}_{error_type}"
        self.error_counts[key] = self.error_counts.get(key, 0) + 1
        
        self.logger.warning(f"API error tracked: {endpoint} - {error_type}")
    
    def get_api_stats(self) -> Dict[str, Any]:
        """Get API performance statistics"""
        
        stats = {
            "request_counts": self.request_counts.copy(),
            "error_counts": self.error_counts.copy(),
            "response_times": {}
        }
        
        # Calculate response time statistics
        for endpoint, times in self.response_times.items():
            if times:
                import numpy as np
                stats["response_times"][endpoint] = {
                    "count": len(times),
                    "mean_ms": float(np.mean(times)),
                    "median_ms": float(np.median(times)),
                    "p95_ms": float(np.percentile(times, 95)),
                    "p99_ms": float(np.percentile(times, 99)),
                    "max_ms": float(np.max(times))
                }
        
        return stats
    
    def performance_monitor(self, endpoint_name: str):
        """Decorator for monitoring API endpoint performance"""
        
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            async def wrapper(*args, **kwargs):
                start_time = time.time()
                
                try:
                    # Execute function
                    result = await func(*args, **kwargs)
                    
                    # Track successful request
                    response_time = (time.time() - start_time) * 1000
                    self.request_counts[endpoint_name] = self.request_counts.get(endpoint_name, 0) + 1
                    
                    if endpoint_name not in self.response_times:
                        self.response_times[endpoint_name] = []
                    self.response_times[endpoint_name].append(response_time)
                    
                    return result
                    
                except Exception as e:
                    # Track error
                    self.track_error(endpoint_name, type(e).__name__)
                    raise
            
            return wrapper
        return decorator


# Global metrics instance
api_metrics = APIMetrics()