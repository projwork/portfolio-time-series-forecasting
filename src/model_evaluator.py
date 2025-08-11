"""
Model evaluation and comparison framework for time series forecasting.
Provides comprehensive evaluation metrics and visualization tools.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union, Any
import warnings
import logging
from datetime import datetime
from scipy import stats

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')


class ModelComparator:
    """
    Comprehensive model comparison and evaluation framework.
    """
    
    def __init__(self):
        """Initialize the model comparator."""
        self.model_results = {}
        self.predictions = {}
        self.metrics = {}
        self.true_values = None
        self.dates = None
        self.evaluation_results = None
        
    def add_model_results(self, model_name: str, predictions: np.ndarray, 
                         actual_values: Optional[np.ndarray] = None,
                         model_type: Optional[str] = None,
                         parameters: Optional[Dict] = None,
                         model_info: Optional[Dict] = None) -> None:
        """
        Add model predictions and metadata for comparison.
        
        Args:
            model_name: Name of the model
            predictions: Model predictions
            actual_values: True values for comparison
            model_type: Type of model (e.g., 'statistical', 'deep_learning', 'fallback')
            parameters: Model parameters and configuration
            model_info: Additional model information (deprecated, use parameters)
        """
        # Store predictions
        self.predictions[model_name] = predictions
        
        # Store model metadata
        self.model_results[model_name] = {
            'predictions': predictions,
            'model_type': model_type or 'unknown',
            'parameters': parameters or {},
        }
        
        # Store actual values if provided (use the first one as reference)
        if actual_values is not None and self.true_values is None:
            self.true_values = actual_values
            
        # Legacy support for model_info
        if model_info:
            self.model_results[model_name].update(model_info)
            
        logger.info(f"Added results for {model_name}: {len(predictions)} predictions, type: {model_type}")
    
    def set_true_values(self, y_true: np.ndarray, dates: Optional[pd.DatetimeIndex] = None) -> None:
        """
        Set the true values for comparison.
        
        Args:
            y_true: True values
            dates: Corresponding dates
        """
        self.true_values = y_true
        self.dates = dates
        logger.info(f"Set true values: {len(y_true)} observations")
    
    def calculate_performance_metrics(self) -> Dict[str, Dict[str, float]]:
        """
        Calculate performance metrics for all stored models.
        
        Returns:
            Dictionary of model names to metrics
        """
        if self.true_values is None:
            raise ValueError("True values must be set before calculating metrics")
        
        results = {}
        for model_name, predictions in self.predictions.items():
            results[model_name] = self.calculate_metrics(model_name, self.true_values, predictions)
        
        return results
    
    def calculate_metrics(self, model_name: str, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate comprehensive evaluation metrics.
        
        Args:
            model_name: Name of the model
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Dictionary of metrics
        """
        # Ensure arrays are same length
        min_len = min(len(y_true), len(y_pred))
        y_true = y_true[:min_len]
        y_pred = y_pred[:min_len]
        
        # Basic error metrics
        mae = np.mean(np.abs(y_true - y_pred))
        mse = np.mean((y_true - y_pred) ** 2)
        rmse = np.sqrt(mse)
        
        # Percentage errors
        mape = np.mean(np.abs((y_true - y_pred) / np.where(y_true != 0, y_true, 1))) * 100
        
        # Symmetric MAPE (better for zero values)
        smape = np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred))) * 100
        
        # R-squared
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        # Adjusted R-squared (assuming 1 predictor for time series)
        n = len(y_true)
        p = 1
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1) if n > p + 1 else r2
        
        # Directional accuracy (for financial time series)
        if len(y_true) > 1:
            actual_direction = np.diff(y_true) > 0
            pred_direction = np.diff(y_pred) > 0
            directional_accuracy = np.mean(actual_direction == pred_direction) * 100
        else:
            directional_accuracy = 0
        
        # Theil's U statistic (relative accuracy)
        theil_u = np.sqrt(np.mean((y_pred - y_true) ** 2)) / np.sqrt(np.mean(y_true ** 2)) if np.mean(y_true ** 2) != 0 else np.inf
        
        # Maximum error
        max_error = np.max(np.abs(y_true - y_pred))
        
        # Mean percentage error (for bias assessment)
        mpe = np.mean((y_true - y_pred) / np.where(y_true != 0, y_true, 1)) * 100
        
        # Median absolute error (robust to outliers)
        median_ae = np.median(np.abs(y_true - y_pred))
        
        # Mean absolute scaled error (for comparison across different scales)
        if len(y_true) > 1:
            naive_forecast_error = np.mean(np.abs(np.diff(y_true)))
            mase = mae / naive_forecast_error if naive_forecast_error != 0 else np.inf
        else:
            mase = np.inf
        
        metrics = {
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'MAPE': mape,
            'SMAPE': smape,
            'R²': r2,
            'Adjusted_R²': adj_r2,
            'Directional_Accuracy': directional_accuracy,
            'Theil_U': theil_u,
            'Max_Error': max_error,
            'MPE': mpe,
            'Median_AE': median_ae,
            'MASE': mase
        }
        
        # Store metrics
        self.metrics[model_name] = metrics
        
        return metrics
    
    def statistical_significance_test(self, model1_name: str, model2_name: str, 
                                    alpha: float = 0.05) -> Dict[str, Any]:
        """
        Perform Diebold-Mariano test for statistical significance of difference.
        
        Args:
            model1_name: Name of first model
            model2_name: Name of second model
            alpha: Significance level
            
        Returns:
            Dictionary with test results
        """
        if self.true_values is None:
            raise ValueError("True values must be set before significance testing")
            
        if model1_name not in self.predictions or model2_name not in self.predictions:
            raise ValueError("Both models must have predictions stored")
        
        y_true = self.true_values
        pred1 = self.predictions[model1_name]
        pred2 = self.predictions[model2_name]
        
        # Ensure same length
        min_len = min(len(y_true), len(pred1), len(pred2))
        y_true = y_true[:min_len]
        pred1 = pred1[:min_len]
        pred2 = pred2[:min_len]
        
        # Calculate squared errors
        err1 = (y_true - pred1) ** 2
        err2 = (y_true - pred2) ** 2
        
        # Difference in errors
        d = err1 - err2
        
        # Diebold-Mariano statistic
        d_mean = np.mean(d)
        d_var = np.var(d, ddof=1)
        
        if d_var == 0:
            dm_stat = 0
            p_value = 1.0
        else:
            dm_stat = d_mean / np.sqrt(d_var / len(d))
            p_value = 2 * (1 - stats.norm.cdf(np.abs(dm_stat)))
        
        is_significant = p_value < alpha
        better_model = model1_name if d_mean < 0 else model2_name
        
        return {
            'dm_statistic': dm_stat,
            'p_value': p_value,
            'is_significant': is_significant,
            'alpha': alpha,
            'better_model': better_model if is_significant else 'No significant difference'
        }
    
    def plot_prediction_comparison(self, dates: Optional[pd.DatetimeIndex] = None,
                                 actual_values: Optional[np.ndarray] = None,
                                 figsize: Tuple[int, int] = (15, 10)) -> None:
        """
        Plot prediction comparison for all models.
        
        Args:
            dates: Date index for x-axis
            actual_values: True values to compare against
            figsize: Figure size
        """
        if not self.predictions:
            raise ValueError("No predictions stored")
        
        y_true = actual_values if actual_values is not None else self.true_values
        if y_true is None:
            raise ValueError("No actual values available for comparison")
        
        # Use provided dates or generate index
        if dates is None:
            dates = pd.date_range(start='2024-01-01', periods=len(y_true), freq='D')
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('Model Prediction Comparison', fontsize=16, fontweight='bold')
        
        # Plot 1: All predictions vs actual
        ax = axes[0, 0]
        ax.plot(dates, y_true, label='Actual', linewidth=2, alpha=0.8)
        
        colors = plt.cm.Set1(np.linspace(0, 1, len(self.predictions)))
        for i, (model_name, predictions) in enumerate(self.predictions.items()):
            min_len = min(len(dates), len(predictions))
            ax.plot(dates[:min_len], predictions[:min_len], 
                   label=model_name, linewidth=2, alpha=0.7, color=colors[i])
        
        ax.set_title('Predictions vs Actual Values')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price ($)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Prediction errors
        ax = axes[0, 1]
        for i, (model_name, predictions) in enumerate(self.predictions.items()):
            min_len = min(len(y_true), len(predictions))
            errors = y_true[:min_len] - predictions[:min_len]
            ax.plot(dates[:min_len], errors, label=f'{model_name} Error', 
                   linewidth=1.5, alpha=0.7, color=colors[i])
        
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax.set_title('Prediction Errors Over Time')
        ax.set_xlabel('Date')
        ax.set_ylabel('Error ($)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Scatter plot of predictions vs actual
        ax = axes[1, 0]
        for i, (model_name, predictions) in enumerate(self.predictions.items()):
            min_len = min(len(y_true), len(predictions))
            ax.scatter(y_true[:min_len], predictions[:min_len], 
                      label=model_name, alpha=0.6, color=colors[i])
        
        # Perfect prediction line
        min_val, max_val = y_true.min(), y_true.max()
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='Perfect Prediction')
        
        ax.set_title('Predictions vs Actual (Scatter)')
        ax.set_xlabel('Actual Values ($)')
        ax.set_ylabel('Predicted Values ($)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Error distribution
        ax = axes[1, 1]
        for i, (model_name, predictions) in enumerate(self.predictions.items()):
            min_len = min(len(y_true), len(predictions))
            errors = y_true[:min_len] - predictions[:min_len]
            ax.hist(errors, bins=30, alpha=0.6, label=model_name, color=colors[i])
        
        ax.axvline(x=0, color='black', linestyle='--', alpha=0.5)
        ax.set_title('Error Distribution')
        ax.set_xlabel('Prediction Error ($)')
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_residual_analysis(self, figsize: Tuple[int, int] = (15, 10)) -> None:
        """
        Plot residual analysis for all models.
        
        Args:
            figsize: Figure size
        """
        if not self.predictions or self.true_values is None:
            raise ValueError("Predictions and true values must be available")
        
        n_models = len(self.predictions)
        fig, axes = plt.subplots(n_models, 3, figsize=figsize)
        if n_models == 1:
            axes = axes.reshape(1, -1)
        
        fig.suptitle('Residual Analysis by Model', fontsize=16, fontweight='bold')
        
        for i, (model_name, predictions) in enumerate(self.predictions.items()):
            min_len = min(len(self.true_values), len(predictions))
            residuals = self.true_values[:min_len] - predictions[:min_len]
            
            # Residuals vs fitted
            axes[i, 0].scatter(predictions[:min_len], residuals, alpha=0.6)
            axes[i, 0].axhline(y=0, color='red', linestyle='--', alpha=0.7)
            axes[i, 0].set_title(f'{model_name}: Residuals vs Fitted')
            axes[i, 0].set_xlabel('Fitted Values')
            axes[i, 0].set_ylabel('Residuals')
            axes[i, 0].grid(True, alpha=0.3)
            
            # Q-Q plot
            stats.probplot(residuals, dist="norm", plot=axes[i, 1])
            axes[i, 1].set_title(f'{model_name}: Q-Q Plot')
            axes[i, 1].grid(True, alpha=0.3)
            
            # Residual histogram
            axes[i, 2].hist(residuals, bins=30, alpha=0.7, edgecolor='black')
            axes[i, 2].axvline(x=0, color='red', linestyle='--', alpha=0.7)
            axes[i, 2].set_title(f'{model_name}: Residual Distribution')
            axes[i, 2].set_xlabel('Residuals')
            axes[i, 2].set_ylabel('Frequency')
            axes[i, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_performance_comparison(self, figsize: Tuple[int, int] = (12, 8)) -> None:
        """
        Plot performance metrics comparison.
        
        Args:
            figsize: Figure size
        """
        if not self.predictions or self.true_values is None:
            raise ValueError("Predictions and true values must be available")
        
        # Calculate metrics for all models
        metrics_data = []
        for model_name, predictions in self.predictions.items():
            metrics = self.calculate_metrics(model_name, self.true_values, predictions)
            metrics['Model'] = model_name
            metrics_data.append(metrics)
        
        df = pd.DataFrame(metrics_data)
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
        
        # MAE comparison
        axes[0, 0].bar(df['Model'], df['MAE'], alpha=0.7)
        axes[0, 0].set_title('Mean Absolute Error (MAE)')
        axes[0, 0].set_ylabel('MAE ($)')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].grid(True, alpha=0.3)
        
        # RMSE comparison
        axes[0, 1].bar(df['Model'], df['RMSE'], alpha=0.7, color='orange')
        axes[0, 1].set_title('Root Mean Squared Error (RMSE)')
        axes[0, 1].set_ylabel('RMSE ($)')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, alpha=0.3)
        
        # MAPE comparison
        axes[1, 0].bar(df['Model'], df['MAPE'], alpha=0.7, color='green')
        axes[1, 0].set_title('Mean Absolute Percentage Error (MAPE)')
        axes[1, 0].set_ylabel('MAPE (%)')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(True, alpha=0.3)
        
        # R² comparison
        axes[1, 1].bar(df['Model'], df['R²'], alpha=0.7, color='red')
        axes[1, 1].set_title('R-squared (R²)')
        axes[1, 1].set_ylabel('R²')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_error_distribution(self, figsize: Tuple[int, int] = (12, 8)) -> None:
        """
        Plot error distribution analysis.
        
        Args:
            figsize: Figure size
        """
        if not self.predictions or self.true_values is None:
            raise ValueError("Predictions and true values must be available")
        
        n_models = len(self.predictions)
        fig, axes = plt.subplots(2, n_models, figsize=figsize)
        if n_models == 1:
            axes = axes.reshape(-1, 1)
        
        fig.suptitle('Error Distribution Analysis', fontsize=16, fontweight='bold')
        
        for i, (model_name, predictions) in enumerate(self.predictions.items()):
            min_len = min(len(self.true_values), len(predictions))
            errors = self.true_values[:min_len] - predictions[:min_len]
            abs_errors = np.abs(errors)
            
            # Error histogram
            axes[0, i].hist(errors, bins=30, alpha=0.7, edgecolor='black')
            axes[0, i].axvline(x=0, color='red', linestyle='--', alpha=0.7)
            axes[0, i].set_title(f'{model_name}: Error Distribution')
            axes[0, i].set_xlabel('Prediction Error ($)')
            axes[0, i].set_ylabel('Frequency')
            axes[0, i].grid(True, alpha=0.3)
            
            # Box plot of absolute errors
            axes[1, i].boxplot(abs_errors)
            axes[1, i].set_title(f'{model_name}: Absolute Error Box Plot')
            axes[1, i].set_ylabel('Absolute Error ($)')
            axes[1, i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def evaluate_all_models(self) -> pd.DataFrame:
        """
        Evaluate all models and create comparison table.
        
        Returns:
            DataFrame with model comparison results
        """
        if self.true_values is None:
            raise ValueError("True values must be set before evaluation")
        
        logger.info("Evaluating all models...")
        
        # Calculate metrics for each model
        for model_name, predictions in self.predictions.items():
            self.calculate_metrics(model_name, self.true_values, predictions)
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame(self.metrics).T
        
        # Add ranking columns (lower is better for error metrics)
        error_metrics = ['MAE', 'MSE', 'RMSE', 'MAPE', 'SMAPE', 'Max_Error', 'Theil_U', 'MASE']
        for metric in error_metrics:
            if metric in comparison_df.columns:
                comparison_df[f'{metric}_Rank'] = comparison_df[metric].rank()
        
        # Higher is better for these metrics
        performance_metrics = ['R²', 'Adjusted_R²', 'Directional_Accuracy']
        for metric in performance_metrics:
            if metric in comparison_df.columns:
                comparison_df[f'{metric}_Rank'] = comparison_df[metric].rank(ascending=False)
        
        # Calculate overall ranking
        rank_columns = [col for col in comparison_df.columns if col.endswith('_Rank')]
        if rank_columns:
            comparison_df['Overall_Rank'] = comparison_df[rank_columns].mean(axis=1)
            comparison_df = comparison_df.sort_values('Overall_Rank')
        
        self.evaluation_results = comparison_df
        
        logger.info("Model evaluation completed!")
        return comparison_df
    
    def plot_predictions_comparison(self, figsize: Tuple[int, int] = (16, 10),
                                  max_points: int = 500) -> None:
        """
        Plot comparison of all model predictions.
        
        Args:
            figsize: Figure size
            max_points: Maximum number of points to plot (for performance)
        """
        if self.true_values is None:
            raise ValueError("True values must be set before plotting")
        
        plt.figure(figsize=figsize)
        
        # Use dates if available, otherwise use index
        x_axis = self.dates if self.dates is not None else range(len(self.true_values))
        
        # Downsample if too many points
        if len(self.true_values) > max_points:
            step = len(self.true_values) // max_points
            indices = range(0, len(self.true_values), step)
            y_true_plot = self.true_values[indices]
            x_axis_plot = x_axis[indices] if hasattr(x_axis, '__getitem__') else [x_axis[i] for i in indices]
        else:
            y_true_plot = self.true_values
            x_axis_plot = x_axis
            indices = range(len(self.true_values))
        
        # Plot actual values
        plt.plot(x_axis_plot, y_true_plot, label='Actual', color='black', 
                linewidth=3, alpha=0.8, zorder=10)
        
        # Plot predictions for each model
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
        for i, (model_name, predictions) in enumerate(self.predictions.items()):
            color = colors[i % len(colors)]
            
            # Downsample predictions to match
            pred_plot = predictions[indices] if len(predictions) > max_points else predictions
            pred_plot = pred_plot[:len(x_axis_plot)]  # Ensure same length
            
            plt.plot(x_axis_plot[:len(pred_plot)], pred_plot, label=model_name, 
                    color=color, linewidth=2, alpha=0.7)
        
        plt.title('Model Predictions Comparison', fontsize=16, fontweight='bold')
        plt.xlabel('Date' if self.dates is not None else 'Time', fontsize=12)
        plt.ylabel('Stock Price ($)', fontsize=12)
        plt.legend(fontsize=10, loc='best')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def plot_error_analysis(self, figsize: Tuple[int, int] = (15, 12)) -> None:
        """
        Plot detailed error analysis for all models.
        
        Args:
            figsize: Figure size
        """
        if self.true_values is None:
            raise ValueError("True values must be set before error analysis")
        
        n_models = len(self.predictions)
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('Model Error Analysis', fontsize=16, fontweight='bold')
        
        # Error distribution
        for model_name, predictions in self.predictions.items():
            min_len = min(len(self.true_values), len(predictions))
            errors = self.true_values[:min_len] - predictions[:min_len]
            
            axes[0, 0].hist(errors, alpha=0.6, label=model_name, bins=30, density=True)
        
        axes[0, 0].set_title('Error Distribution')
        axes[0, 0].set_xlabel('Prediction Error')
        axes[0, 0].set_ylabel('Density')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Error vs actual values
        for model_name, predictions in self.predictions.items():
            min_len = min(len(self.true_values), len(predictions))
            y_true_sub = self.true_values[:min_len]
            errors = y_true_sub - predictions[:min_len]
            
            axes[0, 1].scatter(y_true_sub, errors, alpha=0.6, label=model_name, s=20)
        
        axes[0, 1].axhline(y=0, color='red', linestyle='--', alpha=0.7)
        axes[0, 1].set_title('Error vs Actual Values')
        axes[0, 1].set_xlabel('Actual Values')
        axes[0, 1].set_ylabel('Prediction Error')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Absolute error over time
        x_axis = self.dates if self.dates is not None else range(len(self.true_values))
        for model_name, predictions in self.predictions.items():
            min_len = min(len(self.true_values), len(predictions))
            abs_errors = np.abs(self.true_values[:min_len] - predictions[:min_len])
            
            axes[1, 0].plot(x_axis[:min_len], abs_errors, label=model_name, alpha=0.7)
        
        axes[1, 0].set_title('Absolute Error Over Time')
        axes[1, 0].set_xlabel('Date' if self.dates is not None else 'Time')
        axes[1, 0].set_ylabel('Absolute Error')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Error metrics comparison
        if self.metrics:
            metrics_df = pd.DataFrame(self.metrics).T
            key_metrics = ['MAE', 'RMSE', 'MAPE']
            
            # Select metrics that exist
            available_metrics = [m for m in key_metrics if m in metrics_df.columns]
            if available_metrics:
                metrics_subset = metrics_df[available_metrics]
                
                x_pos = np.arange(len(metrics_subset.index))
                width = 0.8 / len(available_metrics)
                
                for i, metric in enumerate(available_metrics):
                    axes[1, 1].bar(x_pos + i * width, metrics_subset[metric], 
                                  width, label=metric, alpha=0.7)
                
                axes[1, 1].set_title('Key Metrics Comparison')
                axes[1, 1].set_xlabel('Models')
                axes[1, 1].set_ylabel('Metric Value')
                axes[1, 1].set_xticks(x_pos + width * (len(available_metrics) - 1) / 2)
                axes[1, 1].set_xticklabels(metrics_subset.index, rotation=45)
                axes[1, 1].legend()
                axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_metrics_comparison(self, figsize: Tuple[int, int] = (15, 10)) -> None:
        """
        Plot comprehensive metrics comparison.
        
        Args:
            figsize: Figure size
        """
        if not self.metrics:
            raise ValueError("Metrics must be calculated before plotting")
        
        metrics_df = pd.DataFrame(self.metrics).T
        
        # Select key metrics for visualization
        key_metrics = ['MAE', 'RMSE', 'MAPE', 'R²', 'Directional_Accuracy']
        available_metrics = [m for m in key_metrics if m in metrics_df.columns]
        
        if not available_metrics:
            logger.warning("No key metrics available for plotting")
            return
        
        n_metrics = len(available_metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=figsize)
        
        if n_metrics == 1:
            axes = [axes]
        
        fig.suptitle('Model Performance Metrics Comparison', fontsize=16, fontweight='bold')
        
        for i, metric in enumerate(available_metrics):
            values = metrics_df[metric]
            
            # Create bar plot
            bars = axes[i].bar(range(len(values)), values, alpha=0.7)
            
            # Color bars based on performance (green for best, red for worst)
            if metric in ['R²', 'Directional_Accuracy']:
                # Higher is better
                best_idx = values.idxmax()
                worst_idx = values.idxmin()
            else:
                # Lower is better
                best_idx = values.idxmin()
                worst_idx = values.idxmax()
            
            for j, bar in enumerate(bars):
                if values.index[j] == best_idx:
                    bar.set_color('green')
                elif values.index[j] == worst_idx:
                    bar.set_color('red')
                else:
                    bar.set_color('skyblue')
            
            axes[i].set_title(f'{metric}')
            axes[i].set_xticks(range(len(values)))
            axes[i].set_xticklabels(values.index, rotation=45, ha='right')
            axes[i].grid(True, alpha=0.3)
            
            # Add value labels on bars
            for j, (idx, val) in enumerate(values.items()):
                axes[i].text(j, val + val * 0.01, f'{val:.3f}', 
                           ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.show()
    
    def statistical_significance_test(self, model1: str, model2: str,
                                    test_type: str = 'paired_t') -> Dict[str, float]:
        """
        Perform statistical significance test between two models.
        
        Args:
            model1: Name of first model
            model2: Name of second model
            test_type: Type of test ('paired_t', 'wilcoxon')
            
        Returns:
            Test results
        """
        if model1 not in self.predictions or model2 not in self.predictions:
            raise ValueError("Both models must have predictions available")
        
        # Calculate errors for both models
        pred1 = self.predictions[model1]
        pred2 = self.predictions[model2]
        
        min_len = min(len(self.true_values), len(pred1), len(pred2))
        y_true_sub = self.true_values[:min_len]
        
        errors1 = np.abs(y_true_sub - pred1[:min_len])
        errors2 = np.abs(y_true_sub - pred2[:min_len])
        
        if test_type == 'paired_t':
            # Paired t-test
            statistic, p_value = stats.ttest_rel(errors1, errors2)
            test_name = "Paired t-test"
        elif test_type == 'wilcoxon':
            # Wilcoxon signed-rank test (non-parametric)
            statistic, p_value = stats.wilcoxon(errors1, errors2)
            test_name = "Wilcoxon signed-rank test"
        else:
            raise ValueError(f"Unknown test type: {test_type}")
        
        result = {
            'test_name': test_name,
            'statistic': statistic,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'model1_mean_error': np.mean(errors1),
            'model2_mean_error': np.mean(errors2)
        }
        
        logger.info(f"Statistical significance test ({test_name}):")
        logger.info(f"  {model1} vs {model2}")
        logger.info(f"  Statistic: {statistic:.4f}")
        logger.info(f"  P-value: {p_value:.6f}")
        logger.info(f"  Significant difference: {result['significant']}")
        
        return result
    
    def generate_report(self) -> str:
        """
        Generate a comprehensive comparison report.
        
        Returns:
            Formatted report string
        """
        if self.evaluation_results is None:
            self.evaluate_all_models()
        
        report = []
        report.append("="*60)
        report.append("MODEL COMPARISON REPORT")
        report.append("="*60)
        
        # Overall ranking
        report.append("\n📊 OVERALL MODEL RANKING:")
        report.append("-" * 40)
        for i, (model, row) in enumerate(self.evaluation_results.iterrows()):
            rank = i + 1
            overall_score = row.get('Overall_Rank', 'N/A')
            report.append(f"{rank}. {model} (Score: {overall_score:.2f})")
        
        # Key metrics summary
        report.append("\n📈 KEY METRICS SUMMARY:")
        report.append("-" * 40)
        
        key_metrics = ['MAE', 'RMSE', 'MAPE', 'R²', 'Directional_Accuracy']
        for metric in key_metrics:
            if metric in self.evaluation_results.columns:
                report.append(f"\n{metric}:")
                best_model = self.evaluation_results[metric].idxmin() if metric in ['MAE', 'RMSE', 'MAPE'] else self.evaluation_results[metric].idxmax()
                best_value = self.evaluation_results.loc[best_model, metric]
                report.append(f"  Best: {best_model} ({best_value:.4f})")
                
                for model in self.evaluation_results.index:
                    value = self.evaluation_results.loc[model, metric]
                    report.append(f"  {model}: {value:.4f}")
        
        # Model characteristics
        report.append("\n🔍 MODEL CHARACTERISTICS:")
        report.append("-" * 40)
        
        for model_name in self.predictions.keys():
            report.append(f"\n{model_name}:")
            if model_name in self.models and self.models[model_name]:
                for key, value in self.models[model_name].items():
                    report.append(f"  {key}: {value}")
            
            # Performance summary
            if model_name in self.metrics:
                metrics = self.metrics[model_name]
                report.append(f"  Performance Summary:")
                report.append(f"    MAE: {metrics.get('MAE', 'N/A'):.4f}")
                report.append(f"    RMSE: {metrics.get('RMSE', 'N/A'):.4f}")
                report.append(f"    MAPE: {metrics.get('MAPE', 'N/A'):.2f}%")
                report.append(f"    R²: {metrics.get('R²', 'N/A'):.4f}")
        
        # Recommendations
        report.append("\n💡 RECOMMENDATIONS:")
        report.append("-" * 40)
        
        if len(self.evaluation_results) > 0:
            best_model = self.evaluation_results.index[0]
            report.append(f"• Best overall model: {best_model}")
            
            # Check for trade-offs
            mae_best = self.evaluation_results['MAE'].idxmin() if 'MAE' in self.evaluation_results.columns else None
            r2_best = self.evaluation_results['R²'].idxmax() if 'R²' in self.evaluation_results.columns else None
            
            if mae_best != best_model:
                report.append(f"• Best accuracy (MAE): {mae_best}")
            if r2_best != best_model:
                report.append(f"• Best fit (R²): {r2_best}")
        
        report.append("\n" + "="*60)
        
        return "\n".join(report)
    
    def export_results(self, filepath: str) -> None:
        """
        Export comparison results to CSV.
        
        Args:
            filepath: Path to save CSV file
        """
        if self.evaluation_results is None:
            self.evaluate_all_models()
        
        self.evaluation_results.to_csv(filepath)
        logger.info(f"Results exported to {filepath}")


def main():
    """
    Main function to demonstrate model evaluation.
    """
    logger.info("Model Evaluator module loaded successfully!")
    logger.info("Available features:")
    logger.info("  - Comprehensive metric calculation (MAE, RMSE, MAPE, R², etc.)")
    logger.info("  - Model comparison and ranking")
    logger.info("  - Statistical significance testing")
    logger.info("  - Detailed visualization and reporting")


if __name__ == "__main__":
    main()

