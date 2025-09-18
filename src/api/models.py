"""Pydantic models for API requests and responses"""
from typing import Dict, Optional
from datetime import datetime
from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """Request model for predictions"""
    features: Dict[str, float] = Field(..., description="Feature values for prediction")
    model_name: Optional[str] = Field("RandomForest", description="Model to use for prediction")
    version: Optional[str] = Field("latest", description="Model version to use")


class PredictionResponse(BaseModel):
    """Response model for predictions"""
    prediction: int = Field(..., description="Predicted price direction (0=down, 1=up)")
    probability: Optional[float] = Field(None, description="Prediction probability")
    confidence: str = Field(..., description="Confidence level (low, medium, high)")
    model_used: str = Field(..., description="Model name and version used")
    features_processed: int = Field(..., description="Number of features processed")
    prediction_time: datetime = Field(..., description="Timestamp of prediction")


class ModelInfo(BaseModel):
    """Model information response"""
    model_name: str
    version: str
    feature_count: int
    training_accuracy: Optional[float]
    saved_at: datetime
    is_available: bool


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    timestamp: datetime
    models_available: int
    api_version: str


class ErrorResponse(BaseModel):
    """Error response model"""
    error: str
    details: Optional[str] = None
    timestamp: datetime