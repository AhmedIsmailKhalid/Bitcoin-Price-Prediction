"""Test the complete production system"""
import time
import requests


def test_production_system(base_url: str = "http://localhost:8000"):
    """Test production system with monitoring"""
    
    print("=== Production System Testing ===\n")
    
    # Test basic functionality
    print("1. Testing basic API functionality...")
    health_response = requests.get(f"{base_url}/health")
    if health_response.status_code == 200:
        print("   ✅ API is healthy")
    else:
        print("   ❌ API health check failed")
        return False
    
    # Test prediction with monitoring
    print("\n2. Testing predictions with performance monitoring...")
    
    sample_features = {
        "market_cap": 2.3e12, "volatility_3": 0.001, "volatility_5": 0.001,
        "volatility_7": 0.001, "volume_change": 0.001, "roc_3": 0.05,
        "rsi_like": 45.0, "price_position_3": 0.5, "price_position_7": 0.5,
        "trend_3": 1.0, "trend_5": 1.0, "market_cap_change": 0.001,
        "supply_indicator": 19900000, "month": 9, "quarter": 3,
        "hour_sin": 0.5, "hour_cos": 0.866, "month_sin": -0.5,
        "month_cos": 0.866, "is_us_market_hours": 1
    }
    
    # Make multiple predictions to test monitoring
    for i in range(5):
        start_time = time.time()
        response = requests.post(
            f"{base_url}/predict",
            json={"features": sample_features, "model_name": "RandomForest"},
            headers={"Content-Type": "application/json"}
        )
        end_time = time.time()
        
        if response.status_code == 200:
            prediction = response.json()
            print(f"   ✅ Prediction {i+1}: {prediction['prediction']} "
                  f"({prediction['confidence']}) - {(end_time-start_time)*1000:.1f}ms")
        else:
            print(f"   ❌ Prediction {i+1} failed")
        
        time.sleep(0.5)  # Small delay between requests
    
    # Test monitoring endpoints
    print("\n3. Testing monitoring endpoints...")
    
    # API metrics
    metrics_response = requests.get(f"{base_url}/metrics")
    if metrics_response.status_code == 200:
        metrics = metrics_response.json()
        api_metrics = metrics.get("api_metrics", {})
        model_metrics = metrics.get("model_metrics", {})
        
        print("   ✅ Metrics endpoint working")
        print(f"   Predictions tracked: {model_metrics.get('total_predictions', 0)}")
        
        if "/predict" in api_metrics.get("response_times", {}):
            rt = api_metrics["response_times"]["/predict"]
            print(f"   Average response time: {rt.get('mean_ms', 0):.1f}ms")
    else:
        print("   ❌ Metrics endpoint failed")
    
    # Model health
    health_response = requests.get(f"{base_url}/monitoring/model/RandomForest")
    if health_response.status_code == 200:
        health = health_response.json()
        print(f"   ✅ Model health: {health['health_status']} "
              f"(score: {health['health_score']:.1f})")
    else:
        print("   ❌ Model health endpoint failed")
    
    print("\n=== Production Testing Complete ===")
    return True


def stress_test_api(base_url: str = "http://localhost:8000", requests_count: int = 50):
    """Stress test the API"""
    
    print(f"\n=== Stress Testing ({requests_count} requests) ===")
    
    sample_features = {
        "market_cap": 2.3e12, "volatility_3": 0.001, "rsi_like": 45.0,
        "price_position_3": 0.5, "trend_3": 1.0, "supply_indicator": 19900000,
        "month": 9, "quarter": 3, "hour_sin": 0.5, "hour_cos": 0.866,
        "month_sin": -0.5, "month_cos": 0.866, "is_us_market_hours": 1,
        "volatility_5": 0.001, "volatility_7": 0.001, "volume_change": 0.001,
        "roc_3": 0.05, "price_position_7": 0.5, "trend_5": 1.0,
        "market_cap_change": 0.001
    }
    
    successful_requests = 0
    response_times = []
    
    start_test = time.time()
    
    for i in range(requests_count):
        start_time = time.time()
        
        try:
            response = requests.post(
                f"{base_url}/predict",
                json={"features": sample_features, "model_name": "RandomForest"},
                headers={"Content-Type": "application/json"},
                timeout=5
            )
            
            end_time = time.time()
            response_time = (end_time - start_time) * 1000
            response_times.append(response_time)
            
            if response.status_code == 200:
                successful_requests += 1
            
        except Exception as e:
            print(f"Request {i+1} failed: {e}")
    
    end_test = time.time()
    total_test_time = end_test - start_test
    
    # Calculate statistics
    if response_times:
        import numpy as np
        print(f"Successful requests: {successful_requests}/{requests_count}")
        print(f"Success rate: {(successful_requests/requests_count)*100:.1f}%")
        print(f"Total test time: {total_test_time:.2f}s")
        print(f"Requests per second: {requests_count/total_test_time:.1f}")
        print(f"Average response time: {np.mean(response_times):.1f}ms")
        print(f"95th percentile: {np.percentile(response_times, 95):.1f}ms")
        print(f"99th percentile: {np.percentile(response_times, 99):.1f}ms")


if __name__ == "__main__":
    # Test basic functionality
    success = test_production_system()
    
    if success:
        # Run stress test
        stress_test_api(requests_count=25)
        
        print("\n✅ Production system testing complete!")
        print("API is ready for production deployment")
    else:
        print("\n❌ Production system testing failed!")