# conftest.py - Pytest configuration
import pytest
from fastapi.testclient import TestClient

# Import the app when it exists, or create a mock
try:
    from src.serving.api import app
except ImportError:
    # Create a mock FastAPI app for testing if serving module doesn't exist yet
    from fastapi import FastAPI
    app = FastAPI()

class MockModel:
    """Mock ML model for testing"""
    
    def predict(self, features):
        """Mock prediction method"""
        # Return mock prediction based on input type
        if isinstance(features, list):
            return [0.5] * len(features)  # Mock predictions for batch
        else:
            return 0.5  # Mock single prediction
    
    def predict_proba(self, features):
        """Mock probability prediction"""
        if isinstance(features, list):
            return [[0.3, 0.7]] * len(features)
        else:
            return [0.3, 0.7]

class SentimentModel:
    """Mock sentiment analysis model for testing"""
    
    def __init__(self):
        self.model_loaded = True
    
    def predict(self, text):
        """Mock sentiment prediction for single text"""
        # Simple mock: positive sentiment for text containing positive words
        positive_words = ['good', 'great', 'excellent', 'positive', 'bullish', 'up', 'rise']
        negative_words = ['bad', 'terrible', 'negative', 'bearish', 'down', 'fall']
        
        text_lower = text.lower() if isinstance(text, str) else ""
        
        if any(word in text_lower for word in positive_words):
            return 0.7  # Positive sentiment
        elif any(word in text_lower for word in negative_words):
            return -0.7  # Negative sentiment
        else:
            return 0.1  # Neutral sentiment
    
    def predict_batch(self, texts):
        """Mock batch sentiment prediction"""
        return [self.predict(text) for text in texts]

@pytest.fixture(scope="session")
def client():
    return TestClient(app)

@pytest.fixture
def mock_mlflow_model(monkeypatch):
    """Mock MLflow model loading for tests"""
    def mock_load_model(model_uri):
        return MockModel()
    
    # Only patch if mlflow is available
    try:
        monkeypatch.setattr("mlflow.pyfunc.load_model", mock_load_model)
    except:
        pass  # mlflow not available, skip patching

@pytest.fixture
def sentiment_model():
    """Provide sentiment model for testing"""
    return SentimentModel()

@pytest.fixture
def mock_model():
    """Provide mock ML model for testing"""
    return MockModel()