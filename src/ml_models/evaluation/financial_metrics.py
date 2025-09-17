"""Financial performance metrics for trading strategies"""
import pandas as pd
import numpy as np
from typing import Dict, Any
from sklearn.base import accuracy_score

from src.shared.logging import get_logger


class FinancialMetrics:
    """Calculate financial performance metrics for trading strategies"""
    
    def __init__(self):
        self.logger = get_logger(__name__)
    
    def calculate_trading_metrics(self, y_true: pd.Series, y_pred: np.ndarray, 
                                 returns: pd.Series = None) -> Dict[str, float]:
        """Calculate trading-specific performance metrics"""
        
        metrics = {}
        
        # Basic directional accuracy
        metrics['directional_accuracy'] = accuracy_score(y_true, y_pred)
        
        # Calculate confusion matrix components
        from sklearn.metrics import confusion_matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        # Trading-specific metrics
        metrics['true_positive_rate'] = tp / (tp + fn) if (tp + fn) > 0 else 0
        metrics['false_positive_rate'] = fp / (fp + tn) if (fp + tn) > 0 else 0
        metrics['precision_up'] = tp / (tp + fp) if (tp + fp) > 0 else 0
        metrics['precision_down'] = tn / (tn + fn) if (tn + fn) > 0 else 0
        
        # If returns data available, calculate financial metrics
        if returns is not None and len(returns) == len(y_pred):
            financial_metrics = self._calculate_returns_metrics(y_true, y_pred, returns)
            metrics.update(financial_metrics)
        
        return metrics
    
    def _calculate_returns_metrics(self, y_true: pd.Series, y_pred: np.ndarray, 
                                  returns: pd.Series) -> Dict[str, float]:
        """Calculate return-based financial metrics"""
        
        # Strategy returns: follow predictions
        strategy_returns = []
        for i, pred in enumerate(y_pred):
            if pred == 1:  # Predict up - go long
                strategy_returns.append(returns.iloc[i])
            else:  # Predict down - stay out or go short
                strategy_returns.append(0)  # Conservative: stay out of market
        
        strategy_returns = pd.Series(strategy_returns)
        
        # Calculate financial metrics
        metrics = {}
        
        # Total return
        metrics['total_return'] = strategy_returns.sum()
        metrics['mean_return'] = strategy_returns.mean()
        
        # Volatility
        metrics['volatility'] = strategy_returns.std()
        
        # Sharpe ratio (assuming risk-free rate = 0)
        metrics['sharpe_ratio'] = strategy_returns.mean() / strategy_returns.std() if strategy_returns.std() > 0 else 0
        
        # Maximum drawdown
        cumulative_returns = (1 + strategy_returns).cumprod()
        metrics['max_drawdown'] = ((cumulative_returns / cumulative_returns.cummax()) - 1).min()
        
        # Hit rate (percentage of profitable trades)
        profitable_trades = strategy_returns > 0
        metrics['hit_rate'] = profitable_trades.mean()
        
        # Benchmark comparison (buy and hold)
        benchmark_return = returns.sum()
        metrics['excess_return'] = metrics['total_return'] - benchmark_return
        
        return metrics
    
    def generate_trading_report(self, model_results: Dict[str, Any]) -> str:
        """Generate a comprehensive trading performance report"""
        
        report = []
        report.append("=== TRADING PERFORMANCE REPORT ===\n")
        
        for model_name, results in model_results.items():
            if 'trading_metrics' in results:
                metrics = results['trading_metrics']
                
                report.append(f"--- {model_name.upper()} ---")
                report.append(f"Directional Accuracy: {metrics.get('directional_accuracy', 0):.1%}")
                report.append(f"Precision (Up): {metrics.get('precision_up', 0):.1%}")
                report.append(f"Precision (Down): {metrics.get('precision_down', 0):.1%}")
                
                if 'sharpe_ratio' in metrics:
                    report.append(f"Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.3f}")
                    report.append(f"Total Return: {metrics.get('total_return', 0):.2%}")
                    report.append(f"Max Drawdown: {metrics.get('max_drawdown', 0):.2%}")
                    report.append(f"Hit Rate: {metrics.get('hit_rate', 0):.1%}")
                
                report.append("")
        
        return "\n".join(report)