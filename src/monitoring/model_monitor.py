"""Model performance monitoring and drift detection"""
import json
import sqlite3
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any
from datetime import datetime, timedelta

from src.shared.logging import get_logger


class ModelMonitor:
    """Monitor model performance and data drift"""
    
    def __init__(self, db_path: str = "monitoring/model_performance.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.logger = get_logger(__name__)
        
        # Initialize database
        self._init_database()
    
    def log_prediction(self, model_name: str, version: str, features: Dict[str, float],
                      prediction: int, probability: float, confidence: str,
                      response_time_ms: float):
        """Log a prediction for monitoring"""
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO predictions 
                (timestamp, model_name, version, features_json, prediction, 
                 probability, confidence, response_time_ms)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                datetime.now().isoformat(),
                model_name,
                version,
                json.dumps(features),
                prediction,
                probability,
                confidence,
                response_time_ms
            ))
    
    def log_model_metrics(self, model_name: str, version: str, metrics: Dict[str, float]):
        """Log model performance metrics"""
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO model_metrics 
                (timestamp, model_name, version, metrics_json)
                VALUES (?, ?, ?, ?)
            """, (
                datetime.now().isoformat(),
                model_name,
                version,
                json.dumps(metrics)
            ))
    
    def get_prediction_stats(self, hours: int = 24) -> Dict[str, Any]:
        """Get prediction statistics for the last N hours"""
        
        since = datetime.now() - timedelta(hours=hours)
        
        with sqlite3.connect(self.db_path) as conn:
            # Total predictions
            total_predictions = conn.execute(
                "SELECT COUNT(*) FROM predictions WHERE timestamp > ?",
                (since.isoformat(),)
            ).fetchone()[0]
            
            # Predictions by model
            model_stats = conn.execute("""
                SELECT model_name, COUNT(*) as count, AVG(probability) as avg_prob,
                       AVG(response_time_ms) as avg_response_time
                FROM predictions 
                WHERE timestamp > ?
                GROUP BY model_name
            """, (since.isoformat(),)).fetchall()
            
            # Confidence distribution
            confidence_dist = conn.execute("""
                SELECT confidence, COUNT(*) as count
                FROM predictions 
                WHERE timestamp > ?
                GROUP BY confidence
            """, (since.isoformat(),)).fetchall()
            
            # Prediction distribution
            pred_dist = conn.execute("""
                SELECT prediction, COUNT(*) as count
                FROM predictions 
                WHERE timestamp > ?
                GROUP BY prediction
            """, (since.isoformat(),)).fetchall()
        
        return {
            "total_predictions": total_predictions,
            "time_period_hours": hours,
            "model_stats": [
                {
                    "model_name": row[0],
                    "prediction_count": row[1],
                    "avg_probability": row[2],
                    "avg_response_time_ms": row[3]
                }
                for row in model_stats
            ],
            "confidence_distribution": dict(confidence_dist),
            "prediction_distribution": dict(pred_dist)
        }
    
    def detect_drift(self, model_name: str, hours: int = 24) -> Dict[str, Any]:
        """Detect potential data drift in recent predictions"""
        
        since = datetime.now() - timedelta(hours=hours)
        
        with sqlite3.connect(self.db_path) as conn:
            recent_data = conn.execute("""
                SELECT features_json, probability, confidence, timestamp
                FROM predictions 
                WHERE model_name = ? AND timestamp > ?
            """, (model_name, since.isoformat())).fetchall()
        
        if len(recent_data) < 10:
            return {"status": "insufficient_data", "sample_count": len(recent_data)}
        
        # Parse features
        features_list = []
        probabilities = []
        
        for row in recent_data:
            features = json.loads(row[0])
            features_list.append(features)
            probabilities.append(row[1])
        
        # Convert to DataFrame for analysis
        features_df = pd.DataFrame(features_list)
        
        # Calculate basic drift indicators
        drift_indicators = {}
        
        # Feature value distribution analysis
        for column in features_df.columns:
            values = features_df[column].values
            drift_indicators[column] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "zero_ratio": float(np.mean(values == 0))
            }
        
        # Probability distribution analysis
        prob_stats = {
            "mean_probability": float(np.mean(probabilities)),
            "std_probability": float(np.std(probabilities)),
            "low_confidence_ratio": float(np.mean(np.array(probabilities) < 0.6))
        }
        
        return {
            "status": "analyzed",
            "sample_count": len(recent_data),
            "feature_drift": drift_indicators,
            "probability_stats": prob_stats,
            "analysis_period_hours": hours
        }
    
    def get_model_health(self, model_name: str) -> Dict[str, Any]:
        """Get overall model health status"""
        
        # Recent performance (last 24 hours)
        recent_stats = self.get_prediction_stats(24)
        model_recent = next(
            (m for m in recent_stats["model_stats"] if m["model_name"] == model_name),
            None
        )
        
        # Drift analysis
        drift_analysis = self.detect_drift(model_name, 24)
        
        # Health score calculation
        health_score = 100.0
        issues = []
        
        if model_recent:
            # Check response time
            if model_recent["avg_response_time_ms"] > 1000:
                health_score -= 20
                issues.append("High response time")
            
            # Check prediction confidence
            if model_recent["avg_probability"] < 0.6:
                health_score -= 30
                issues.append("Low prediction confidence")
        else:
            health_score -= 50
            issues.append("No recent predictions")
        
        # Check for drift
        if drift_analysis["status"] == "analyzed":
            prob_stats = drift_analysis["probability_stats"]
            if prob_stats["low_confidence_ratio"] > 0.5:
                health_score -= 25
                issues.append("High proportion of low-confidence predictions")
        
        # Determine health status
        if health_score >= 80:
            status = "healthy"
        elif health_score >= 60:
            status = "warning"
        else:
            status = "critical"
        
        return {
            "model_name": model_name,
            "health_status": status,
            "health_score": health_score,
            "issues": issues,
            "recent_stats": model_recent,
            "drift_analysis": drift_analysis
        }
    
    def _init_database(self):
        """Initialize monitoring database"""
        
        with sqlite3.connect(self.db_path) as conn:
            # Predictions table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    model_name TEXT,
                    version TEXT,
                    features_json TEXT,
                    prediction INTEGER,
                    probability REAL,
                    confidence TEXT,
                    response_time_ms REAL
                )
            """)
            
            # Model metrics table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS model_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    model_name TEXT,
                    version TEXT,
                    metrics_json TEXT
                )
            """)
            
            # Create indexes
            conn.execute("CREATE INDEX IF NOT EXISTS idx_predictions_timestamp ON predictions(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_predictions_model ON predictions(model_name)")