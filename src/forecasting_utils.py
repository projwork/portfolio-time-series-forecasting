"""
Time series forecasting utilities for portfolio analysis.
Provides data splitting, preprocessing, and evaluation functions.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from typing import Tuple, Dict, List, Optional, Union
import warnings
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TimeSeriesUtils:
    """
    Utility class for time series forecasting operations.
    """
    
    def __init__(self):
        """Initialize the time series utilities."""
        self.train_data = None
        self.test_data = None
        self.split_date = None
        
    def chronological_split(self, data: pd.DataFrame, test_size: float = 0.2, 
                          date_column: str = 'Date') -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split time series data chronologically.
        
        Args:
            data: DataFrame with time series data
            test_size: Fraction of data to use for testing (default 0.2)
            date_column: Name of the date column
            
        Returns:
            Tuple of (train_data, test_data)
        """
        # Ensure data is sorted by date
        data_sorted = data.sort_values(date_column).reset_index(drop=True)
        
        # Calculate split point
        split_idx = int(len(data_sorted) * (1 - test_size))
        
        # Split data
        train_data = data_sorted.iloc[:split_idx].copy()
        test_data = data_sorted.iloc[split_idx:].copy()
        
        # Store split information
        self.train_data = train_data
        self.test_data = test_data
        self.split_date = test_data[date_column].iloc[0]
        
        logger.info(f"Data split chronologically:")
        logger.info(f"  Train set: {len(train_data)} records ({train_data[date_column].min()} to {train_data[date_column].max()})")
        logger.info(f"  Test set: {len(test_data)} records ({test_data[date_column].min()} to {test_data[date_column].max()})")
        logger.info(f"  Split date: {self.split_date}")
        
        return train_data, test_data
    
    def prepare_target_series(self, data: pd.DataFrame, target_column: str = 'Close',
                            date_column: str = 'Date') -> pd.Series:
        """
        Prepare target series for forecasting.
        
        Args:
            data: DataFrame with time series data
            target_column: Name of the target column to forecast
            date_column: Name of the date column
            
        Returns:
            Series with datetime index
        """
        # Create series with datetime index
        series = pd.Series(
            data[target_column].values,
            index=pd.to_datetime(data[date_column]),
            name=target_column
        )
        
        # Remove any missing values
        series = series.dropna()
        
        logger.info(f"Prepared target series '{target_column}' with {len(series)} observations")
        logger.info(f"  Date range: {series.index.min()} to {series.index.max()}")
        logger.info(f"  Value range: ${series.min():.2f} to ${series.max():.2f}")
        
        return series
    
    def create_lstm_sequences(self, data: np.ndarray, sequence_length: int = 60,
                            forecast_horizon: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for LSTM training.
        
        Args:
            data: 1D array of time series data
            sequence_length: Number of time steps to look back
            forecast_horizon: Number of time steps to forecast ahead
            
        Returns:
            Tuple of (X, y) arrays for LSTM training
        """
        X, y = [], []
        
        for i in range(sequence_length, len(data) - forecast_horizon + 1):
            # Input sequence
            X.append(data[i - sequence_length:i])
            # Target value(s)
            if forecast_horizon == 1:
                y.append(data[i])
            else:
                y.append(data[i:i + forecast_horizon])
        
        X = np.array(X)
        y = np.array(y)
        
        # Reshape X for LSTM input (samples, time steps, features)
        X = X.reshape((X.shape[0], X.shape[1], 1))
        
        logger.info(f"Created LSTM sequences:")
        logger.info(f"  Input shape: {X.shape}")
        logger.info(f"  Output shape: {y.shape}")
        logger.info(f"  Sequence length: {sequence_length}")
        logger.info(f"  Forecast horizon: {forecast_horizon}")
        
        return X, y
    
    def scale_data(self, train_data: np.ndarray, test_data: Optional[np.ndarray] = None,
                   method: str = 'minmax') -> Tuple[np.ndarray, np.ndarray, object]:
        """
        Scale time series data for neural network training.
        
        Args:
            train_data: Training data array
            test_data: Test data array (optional)
            method: Scaling method ('minmax' or 'standard')
            
        Returns:
            Tuple of (scaled_train, scaled_test, scaler)
        """
        if method == 'minmax':
            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler()
        elif method == 'standard':
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
        else:
            raise ValueError(f"Unknown scaling method: {method}")
        
        # Fit scaler on training data
        train_data_reshaped = train_data.reshape(-1, 1)
        scaled_train = scaler.fit_transform(train_data_reshaped).flatten()
        
        # Transform test data if provided
        scaled_test = None
        if test_data is not None:
            test_data_reshaped = test_data.reshape(-1, 1)
            scaled_test = scaler.transform(test_data_reshaped).flatten()
        
        logger.info(f"Applied {method} scaling to time series data")
        logger.info(f"  Train data scale: {scaled_train.min():.4f} to {scaled_train.max():.4f}")
        if scaled_test is not None:
            logger.info(f"  Test data scale: {scaled_test.min():.4f} to {scaled_test.max():.4f}")
        
        return scaled_train, scaled_test, scaler
    
    def inverse_scale(self, scaled_data: np.ndarray, scaler: object) -> np.ndarray:
        """
        Inverse transform scaled data back to original scale.
        
        Args:
            scaled_data: Scaled data array
            scaler: Fitted scaler object
            
        Returns:
            Data in original scale
        """
        # Reshape for inverse transform
        scaled_reshaped = scaled_data.reshape(-1, 1)
        original_data = scaler.inverse_transform(scaled_reshaped).flatten()
        
        return original_data


class ForecastEvaluator:
    """
    Class for evaluating time series forecasting performance.
    """
    
    def __init__(self):
        """Initialize the forecast evaluator."""
        self.metrics_history = {}
        
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                         model_name: str = "Model") -> Dict[str, float]:
        """
        Calculate comprehensive forecasting metrics.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            model_name: Name of the model for tracking
            
        Returns:
            Dictionary of metrics
        """
        # Ensure arrays are the same length
        min_len = min(len(y_true), len(y_pred))
        y_true = y_true[:min_len]
        y_pred = y_pred[:min_len]
        
        # Calculate metrics
        mae = np.mean(np.abs(y_true - y_pred))
        mse = np.mean((y_true - y_pred) ** 2)
        rmse = np.sqrt(mse)
        
        # MAPE (handle division by zero)
        mape = np.mean(np.abs((y_true - y_pred) / np.where(y_true != 0, y_true, 1))) * 100
        
        # Additional metrics
        r2 = 1 - (np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2))
        
        # Directional accuracy (for stock prices)
        direction_true = np.diff(y_true) > 0
        direction_pred = np.diff(y_pred) > 0
        directional_accuracy = np.mean(direction_true == direction_pred) * 100
        
        metrics = {
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'MAPE': mape,
            'R²': r2,
            'Directional_Accuracy': directional_accuracy
        }
        
        # Store metrics
        self.metrics_history[model_name] = metrics
        
        logger.info(f"Metrics for {model_name}:")
        logger.info(f"  MAE: {mae:.4f}")
        logger.info(f"  RMSE: {rmse:.4f}")
        logger.info(f"  MAPE: {mape:.2f}%")
        logger.info(f"  R²: {r2:.4f}")
        logger.info(f"  Directional Accuracy: {directional_accuracy:.2f}%")
        
        return metrics
    
    def compare_models(self, metrics_dict: Dict[str, Dict[str, float]]) -> pd.DataFrame:
        """
        Compare multiple models' performance.
        
        Args:
            metrics_dict: Dictionary of model metrics
            
        Returns:
            DataFrame with comparison results
        """
        comparison_df = pd.DataFrame(metrics_dict).T
        
        # Rank models by key metrics (lower is better for error metrics)
        comparison_df['MAE_Rank'] = comparison_df['MAE'].rank()
        comparison_df['RMSE_Rank'] = comparison_df['RMSE'].rank()
        comparison_df['MAPE_Rank'] = comparison_df['MAPE'].rank()
        
        # Higher is better for these metrics
        comparison_df['R²_Rank'] = comparison_df['R²'].rank(ascending=False)
        comparison_df['Dir_Acc_Rank'] = comparison_df['Directional_Accuracy'].rank(ascending=False)
        
        # Overall ranking (simple average of ranks)
        rank_columns = ['MAE_Rank', 'RMSE_Rank', 'MAPE_Rank', 'R²_Rank', 'Dir_Acc_Rank']
        comparison_df['Overall_Rank'] = comparison_df[rank_columns].mean(axis=1)
        
        # Sort by overall ranking
        comparison_df = comparison_df.sort_values('Overall_Rank')
        
        return comparison_df
    
    def plot_predictions(self, y_true: np.ndarray, predictions_dict: Dict[str, np.ndarray],
                        dates: Optional[pd.DatetimeIndex] = None, title: str = "Model Predictions",
                        figsize: Tuple[int, int] = (15, 8)) -> None:
        """
        Plot actual vs predicted values for multiple models.
        
        Args:
            y_true: True values
            predictions_dict: Dictionary of model predictions
            dates: Datetime index for x-axis
            title: Plot title
            figsize: Figure size
        """
        plt.figure(figsize=figsize)
        
        # Use index if dates not provided
        if dates is None:
            dates = range(len(y_true))
        
        # Plot actual values
        plt.plot(dates, y_true, label='Actual', color='black', linewidth=2, alpha=0.8)
        
        # Plot predictions for each model
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
        for i, (model_name, y_pred) in enumerate(predictions_dict.items()):
            color = colors[i % len(colors)]
            plt.plot(dates[:len(y_pred)], y_pred, label=model_name, 
                    color=color, linewidth=1.5, alpha=0.7)
        
        plt.title(title, fontsize=16, fontweight='bold')
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Stock Price ($)', fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def plot_residuals(self, y_true: np.ndarray, y_pred: np.ndarray, 
                      model_name: str = "Model", figsize: Tuple[int, int] = (12, 8)) -> None:
        """
        Plot residual analysis for a model.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            model_name: Name of the model
            figsize: Figure size
        """
        residuals = y_true - y_pred
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle(f'Residual Analysis - {model_name}', fontsize=16, fontweight='bold')
        
        # Residuals vs Time
        axes[0, 0].plot(residuals)
        axes[0, 0].axhline(y=0, color='red', linestyle='--', alpha=0.7)
        axes[0, 0].set_title('Residuals vs Time')
        axes[0, 0].set_ylabel('Residuals')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Residuals vs Predicted
        axes[0, 1].scatter(y_pred, residuals, alpha=0.6)
        axes[0, 1].axhline(y=0, color='red', linestyle='--', alpha=0.7)
        axes[0, 1].set_title('Residuals vs Predicted')
        axes[0, 1].set_xlabel('Predicted Values')
        axes[0, 1].set_ylabel('Residuals')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Histogram of residuals
        axes[1, 0].hist(residuals, bins=30, alpha=0.7, density=True)
        axes[1, 0].axvline(residuals.mean(), color='red', linestyle='--', 
                          label=f'Mean: {residuals.mean():.4f}')
        axes[1, 0].set_title('Distribution of Residuals')
        axes[1, 0].set_xlabel('Residuals')
        axes[1, 0].set_ylabel('Density')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Q-Q plot
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=axes[1, 1])
        axes[1, 1].set_title('Q-Q Plot (Normal Distribution)')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()


def validate_time_series_assumptions(series: pd.Series, model_type: str = "ARIMA") -> Dict[str, any]:
    """
    Validate assumptions for time series modeling.
    
    Args:
        series: Time series data
        model_type: Type of model ("ARIMA" or "LSTM")
        
    Returns:
        Dictionary with validation results
    """
    from statsmodels.tsa.stattools import adfuller
    from statsmodels.stats.diagnostic import acorr_ljungbox
    
    results = {}
    
    # Stationarity test
    adf_result = adfuller(series.dropna())
    results['stationarity'] = {
        'adf_statistic': adf_result[0],
        'p_value': adf_result[1],
        'is_stationary': adf_result[1] < 0.05,
        'critical_values': adf_result[4]
    }
    
    # Autocorrelation test (Ljung-Box)
    if len(series) > 10:
        lb_result = acorr_ljungbox(series.dropna(), lags=10, return_df=True)
        results['autocorrelation'] = {
            'ljung_box_stat': lb_result['lb_stat'].iloc[-1],
            'p_value': lb_result['lb_pvalue'].iloc[-1],
            'has_autocorr': lb_result['lb_pvalue'].iloc[-1] < 0.05
        }
    
    # Basic statistics
    results['descriptive_stats'] = {
        'count': len(series),
        'mean': series.mean(),
        'std': series.std(),
        'min': series.min(),
        'max': series.max(),
        'skewness': series.skew(),
        'kurtosis': series.kurtosis()
    }
    
    logger.info(f"Time series validation for {model_type}:")
    logger.info(f"  Stationarity (ADF): p-value = {results['stationarity']['p_value']:.6f}")
    if 'autocorrelation' in results:
        logger.info(f"  Autocorrelation (LB): p-value = {results['autocorrelation']['p_value']:.6f}")
    
    return results


def main():
    """
    Main function to demonstrate forecasting utilities.
    """
    logger.info("Time series forecasting utilities loaded successfully!")
    logger.info("Available classes:")
    logger.info("  - TimeSeriesUtils: Data splitting and preprocessing")
    logger.info("  - ForecastEvaluator: Model evaluation and comparison")
    logger.info("Available functions:")
    logger.info("  - validate_time_series_assumptions: Check modeling assumptions")


if __name__ == "__main__":
    main()

