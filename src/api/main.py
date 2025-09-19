"""FastAPI application for Bitcoin prediction service"""
import time
from datetime import datetime
from typing import Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from src.monitoring.api_metrics import api_metrics

from src.api.models import (
    PredictionRequest, PredictionResponse, ModelInfo, 
    HealthResponse, ErrorResponse
)
from src.api.prediction_service import PredictionService
from src.shared.logging import get_logger, setup_logging


# Setup logging
setup_logging()
logger = get_logger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Bitcoin Price Prediction API",
    description="Machine learning API for Bitcoin price direction prediction",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize prediction service
prediction_service = PredictionService()


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {
        "message": "Bitcoin Price Prediction API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    try:
        models = prediction_service.get_available_models()
        available_count = sum(1 for model in models.values() if model.get("is_available", False))
        
        return HealthResponse(
            status="healthy",
            timestamp=datetime.now(),
            models_available=available_count,
            api_version="1.0.0"
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail="Service unhealthy")


@app.post("/predict", response_model=PredictionResponse)
@api_metrics.performance_monitor("/predict")
async def predict(request: PredictionRequest):
    """Make price direction prediction"""
    start_time = time.time()
    
    try:
        # Validate features
        is_valid, validation_message = prediction_service.validate_features(
            request.features, request.model_name, request.version
        )
        
        if not is_valid:
            api_metrics.track_error("/predict", "ValidationError")
            raise HTTPException(status_code=400, detail=validation_message)
        
        # Make prediction
        prediction, probability, confidence = prediction_service.predict(
            request.features, request.model_name, request.version
        )
        
        # Calculate response time
        response_time_ms = (time.time() - start_time) * 1000
        
        # Track prediction
        api_metrics.track_prediction(
            request.model_name, request.version, request.features,
            prediction, probability, confidence, response_time_ms
        )
        
        return PredictionResponse(
            prediction=prediction,
            probability=probability,
            confidence=confidence,
            model_used=f"{request.model_name} v{request.version}",
            features_processed=len(request.features),
            prediction_time=datetime.now()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        api_metrics.track_error("/predict", "InternalError")
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.get("/models", response_model=Dict[str, Any])
async def list_models():
    """List available models"""
    try:
        return prediction_service.get_available_models()
    except Exception as e:
        logger.error(f"Failed to list models: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve models")


@app.get("/models/{model_name}", response_model=ModelInfo)
async def get_model_info(model_name: str, version: str = "latest"):
    """Get detailed model information"""
    try:
        models = prediction_service.get_available_models()
        
        if model_name not in models:
            raise HTTPException(status_code=404, detail=f"Model not found: {model_name}")
        
        model_data = models[model_name]
        
        return ModelInfo(
            model_name=model_name,
            version=model_data["latest_version"] if version == "latest" else version,
            feature_count=model_data.get("feature_count", 0),
            training_accuracy=model_data.get("training_accuracy"),
            saved_at=datetime.fromisoformat(model_data["saved_at"]) if model_data.get("saved_at") else datetime.now(),
            is_available=model_data.get("is_available", False)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get model info: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve model information")
    
    
@app.get("/metrics", response_model=Dict[str, Any])
async def get_metrics():
    """Get API performance metrics"""
    try:
        api_stats = api_metrics.get_api_stats()
        model_stats = api_metrics.model_monitor.get_prediction_stats(24)
        
        return {
            "api_metrics": api_stats,
            "model_metrics": model_stats,
            "generated_at": datetime.now()
        }
    except Exception as e:
        logger.error(f"Failed to get metrics: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve metrics")


@app.get("/monitoring/model/{model_name}", response_model=Dict[str, Any])
async def get_model_health(model_name: str):
    """Get model health status"""
    try:
        health_status = api_metrics.model_monitor.get_model_health(model_name)
        return health_status
    except Exception as e:
        logger.error(f"Failed to get model health: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve model health")


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            details=str(exc),
            timestamp=datetime.now()
        ).dict()
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")