"""Test the prediction API"""
import requests


def test_api(base_url: str = "http://localhost:8000"):
    """Test all API endpoints"""
    
    print("=== Testing Bitcoin Prediction API ===\n")
    
    # Test health endpoint
    print("1. Testing health check...")
    try:
        response = requests.get(f"{base_url}/health")
        if response.status_code == 200:
            health = response.json()
            print(f"   ✅ Health: {health['status']}")
            print(f"   Models available: {health['models_available']}")
        else:
            print(f"   ❌ Health check failed: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Health check error: {e}")
    
    # Test models list
    print("\n2. Testing models list...")
    try:
        response = requests.get(f"{base_url}/models")
        if response.status_code == 200:
            models = response.json()
            print(f"   ✅ Found {len(models)} model types:")
            for model_name, info in models.items():
                status = "✅" if info.get("is_available") else "❌"
                print(f"     {status} {model_name} v{info.get('latest_version', 'unknown')}")
        else:
            print(f"   ❌ Models list failed: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Models list error: {e}")
    
    # Test prediction with sample features
    print("\n3. Testing prediction...")
    
    # Sample features based on our feature engineering
    sample_features = {
        "market_cap": 2.3e12,
        "volatility_3": 0.001,
        "volatility_5": 0.001,
        "volatility_7": 0.001,
        "volume_change": 0.001,
        "roc_3": 0.05,
        "rsi_like": 45.0,
        "price_position_3": 0.5,
        "price_position_7": 0.5,
        "trend_3": 1.0,
        "trend_5": 1.0,
        "market_cap_change": 0.001,
        "supply_indicator": 19900000,
        "month": 9,
        "quarter": 3,
        "hour_sin": 0.5,
        "hour_cos": 0.866,
        "month_sin": -0.5,
        "month_cos": 0.866,
        "is_us_market_hours": 1
    }
    
    prediction_request = {
        "features": sample_features,
        "model_name": "RandomForest",
        "version": "latest"
    }
    
    try:
        response = requests.post(
            f"{base_url}/predict",
            json=prediction_request,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            prediction = response.json()
            print("   ✅ Prediction successful:")
            print(f"     Direction: {'UP' if prediction['prediction'] == 1 else 'DOWN'}")
            print(f"     Confidence: {prediction['confidence']}")
            print(f"     Model: {prediction['model_used']}")
            if prediction.get('probability'):
                print(f"     Probability: {prediction['probability']:.3f}")
        else:
            print(f"   ❌ Prediction failed: {response.status_code}")
            print(f"     Error: {response.text}")
    except Exception as e:
        print(f"   ❌ Prediction error: {e}")
    
    # Test model info
    print("\n4. Testing model info...")
    try:
        response = requests.get(f"{base_url}/models/RandomForest")
        if response.status_code == 200:
            model_info = response.json()
            print("   ✅ Model info retrieved:")
            print(f"     Features: {model_info['feature_count']}")
            print(f"     Accuracy: {model_info.get('training_accuracy', 'N/A')}")
            print(f"     Available: {model_info['is_available']}")
        else:
            print(f"   ❌ Model info failed: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Model info error: {e}")
    
    print("\n=== API Testing Complete ===")


if __name__ == "__main__":
    test_api()