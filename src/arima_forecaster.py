"""
ARIMA and SARIMA time series forecasting module.
Implements classical statistical models for stock price prediction.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict, List, Optional, Union
import warnings
import logging
from datetime import datetime

# Statistical modeling imports
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox

# Auto ARIMA for parameter optimization
try:
    from pmdarima import auto_arima
    AUTO_ARIMA_AVAILABLE = True
except (ImportError, ValueError) as e:
    AUTO_ARIMA_AVAILABLE = False
    logger = logging.getLogger(__name__)
    if "numpy.dtype size changed" in str(e):
        logger.warning("pmdarima/numpy compatibility issue detected. Manual parameter tuning will be used.")
    else:
        logger.warning("pmdarima not available. Manual parameter tuning will be used.")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')


class ARIMAForecaster:
    """
    ARIMA and SARIMA forecasting model implementation.
    """
    
    def __init__(self, seasonal: bool = False, seasonal_periods: int = 252):
        """
        Initialize ARIMA forecaster.
        
        Args:
            seasonal: Whether to use SARIMA (seasonal ARIMA)
            seasonal_periods: Number of periods in a season (252 for daily stock data = trading days in a year)
        """
        self.seasonal = seasonal
        self.seasonal_periods = seasonal_periods
        self.model = None
        self.fitted_model = None
        self.best_params = None
        self.train_data = None
        self.is_fitted = False
        
    def check_stationarity(self, series: pd.Series, alpha: float = 0.05) -> Dict[str, any]:
        """
        Check if the time series is stationary using ADF test.
        
        Args:
            series: Time series data
            alpha: Significance level for the test
            
        Returns:
            Dictionary with stationarity test results
        """
        # Perform ADF test
        adf_result = adfuller(series.dropna())
        
        result = {
            'adf_statistic': adf_result[0],
            'p_value': adf_result[1],
            'critical_values': adf_result[4],
            'is_stationary': adf_result[1] < alpha,
            'n_lags_used': adf_result[2],
            'n_observations': adf_result[3]
        }
        
        logger.info(f"Stationarity Test Results:")
        logger.info(f"  ADF Statistic: {result['adf_statistic']:.6f}")
        logger.info(f"  P-value: {result['p_value']:.6f}")
        logger.info(f"  Critical Values: {result['critical_values']}")
        logger.info(f"  Is Stationary: {result['is_stationary']}")
        
        return result
    
    def difference_series(self, series: pd.Series, d: int = 1) -> pd.Series:
        """
        Difference the time series to make it stationary.
        
        Args:
            series: Time series data
            d: Order of differencing
            
        Returns:
            Differenced series
        """
        differenced = series.copy()
        
        for i in range(d):
            differenced = differenced.diff().dropna()
            logger.info(f"Applied {i+1} order differencing. Remaining observations: {len(differenced)}")
        
        return differenced
    
    def plot_diagnostics(self, series: pd.Series, max_lags: int = 40, 
                        figsize: Tuple[int, int] = (15, 10)) -> None:
        """
        Plot ACF and PACF for parameter selection.
        
        Args:
            series: Time series data
            max_lags: Maximum number of lags to plot
            figsize: Figure size
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('Time Series Diagnostic Plots', fontsize=16, fontweight='bold')
        
        # Time series plot
        axes[0, 0].plot(series.index, series.values)
        axes[0, 0].set_title('Time Series')
        axes[0, 0].set_xlabel('Date')
        axes[0, 0].set_ylabel('Value')
        axes[0, 0].grid(True, alpha=0.3)
        
        # ACF plot
        plot_acf(series.dropna(), ax=axes[0, 1], lags=max_lags, alpha=0.05)
        axes[0, 1].set_title('Autocorrelation Function (ACF)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # PACF plot
        plot_pacf(series.dropna(), ax=axes[1, 0], lags=max_lags, alpha=0.05)
        axes[1, 0].set_title('Partial Autocorrelation Function (PACF)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Histogram
        axes[1, 1].hist(series.dropna(), bins=30, alpha=0.7, density=True)
        axes[1, 1].set_title('Distribution of Values')
        axes[1, 1].set_xlabel('Value')
        axes[1, 1].set_ylabel('Density')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def seasonal_decomposition(self, series: pd.Series, model: str = 'additive',
                             figsize: Tuple[int, int] = (15, 12)) -> None:
        """
        Perform seasonal decomposition of the time series.
        
        Args:
            series: Time series data
            model: Type of decomposition ('additive' or 'multiplicative')
            figsize: Figure size
        """
        if not self.seasonal:
            logger.warning("Seasonal decomposition requested but seasonal=False")
            return
        
        try:
            decomposition = seasonal_decompose(
                series.dropna(), 
                model=model, 
                period=self.seasonal_periods
            )
            
            fig, axes = plt.subplots(4, 1, figsize=figsize)
            fig.suptitle(f'Seasonal Decomposition ({model.title()})', fontsize=16, fontweight='bold')
            
            # Original series
            decomposition.observed.plot(ax=axes[0], title='Original')
            axes[0].grid(True, alpha=0.3)
            
            # Trend
            decomposition.trend.plot(ax=axes[1], title='Trend')
            axes[1].grid(True, alpha=0.3)
            
            # Seasonal
            decomposition.seasonal.plot(ax=axes[2], title='Seasonal')
            axes[2].grid(True, alpha=0.3)
            
            # Residual
            decomposition.resid.plot(ax=axes[3], title='Residual')
            axes[3].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            logger.error(f"Error in seasonal decomposition: {str(e)}")
    
    def auto_arima_search(self, series: pd.Series, seasonal: bool = None,
                         max_p: int = 5, max_d: int = 2, max_q: int = 5,
                         max_P: int = 2, max_D: int = 1, max_Q: int = 2) -> Dict[str, any]:
        """
        Use auto ARIMA to find optimal parameters.
        
        Args:
            series: Time series data
            seasonal: Override seasonal setting
            max_p, max_d, max_q: Maximum values for ARIMA parameters
            max_P, max_D, max_Q: Maximum values for seasonal ARIMA parameters
            
        Returns:
            Dictionary with optimal parameters and model information
        """
        if not AUTO_ARIMA_AVAILABLE:
            logger.error("pmdarima not available. Please install with: pip install pmdarima")
            return self.grid_search_arima(series, max_p, max_d, max_q)
        
        if seasonal is None:
            seasonal = self.seasonal
        
        logger.info("Starting auto ARIMA parameter search...")
        
        try:
            auto_model = auto_arima(
                series.dropna(),
                seasonal=seasonal,
                m=self.seasonal_periods if seasonal else 1,
                max_p=max_p, max_d=max_d, max_q=max_q,
                max_P=max_P, max_D=max_D, max_Q=max_Q,
                stepwise=True,
                suppress_warnings=True,
                error_action='ignore',
                trace=True
            )
            
            # Extract parameters
            order = auto_model.order
            seasonal_order = auto_model.seasonal_order if seasonal else None
            
            result = {
                'order': order,
                'seasonal_order': seasonal_order,
                'aic': auto_model.aic(),
                'bic': auto_model.bic(),
                'params': auto_model.params(),
                'fitted_model': auto_model
            }
            
            logger.info(f"Auto ARIMA Results:")
            logger.info(f"  Optimal ARIMA order: {order}")
            if seasonal_order:
                logger.info(f"  Seasonal order: {seasonal_order}")
            logger.info(f"  AIC: {result['aic']:.2f}")
            logger.info(f"  BIC: {result['bic']:.2f}")
            
            self.best_params = result
            return result
            
        except Exception as e:
            logger.error(f"Auto ARIMA failed: {str(e)}")
            logger.info("Falling back to grid search...")
            return self.grid_search_arima(series, max_p, max_d, max_q)
    
    def grid_search_arima(self, series: pd.Series, max_p: int = 3, max_d: int = 2, 
                         max_q: int = 3) -> Dict[str, any]:
        """
        Grid search for optimal ARIMA parameters.
        
        Args:
            series: Time series data
            max_p, max_d, max_q: Maximum values for parameters
            
        Returns:
            Dictionary with optimal parameters
        """
        logger.info("Starting grid search for ARIMA parameters...")
        
        best_aic = np.inf
        best_params = None
        results = []
        
        # Clean data
        clean_series = series.dropna()
        
        for p in range(max_p + 1):
            for d in range(max_d + 1):
                for q in range(max_q + 1):
                    try:
                        if self.seasonal:
                            model = SARIMAX(
                                clean_series,
                                order=(p, d, q),
                                seasonal_order=(1, 1, 1, self.seasonal_periods),
                                enforce_stationarity=False,
                                enforce_invertibility=False
                            )
                        else:
                            model = ARIMA(clean_series, order=(p, d, q))
                        
                        fitted_model = model.fit()
                        aic = fitted_model.aic
                        
                        results.append({
                            'order': (p, d, q),
                            'aic': aic,
                            'bic': fitted_model.bic,
                            'model': fitted_model
                        })
                        
                        if aic < best_aic:
                            best_aic = aic
                            best_params = (p, d, q)
                        
                        logger.debug(f"ARIMA{(p, d, q)}: AIC = {aic:.2f}")
                        
                    except Exception as e:
                        logger.debug(f"Failed to fit ARIMA{(p, d, q)}: {str(e)}")
                        continue
        
        if best_params is None:
            logger.error("Grid search failed to find any valid parameters")
            best_params = (1, 1, 1)  # Default fallback
        
        result = {
            'order': best_params,
            'seasonal_order': (1, 1, 1, self.seasonal_periods) if self.seasonal else None,
            'aic': best_aic,
            'all_results': results
        }
        
        logger.info(f"Grid Search Results:")
        logger.info(f"  Best ARIMA order: {best_params}")
        logger.info(f"  Best AIC: {best_aic:.2f}")
        logger.info(f"  Total models evaluated: {len(results)}")
        
        self.best_params = result
        return result
    
    def fit(self, series: pd.Series, order: Optional[Tuple[int, int, int]] = None,
            seasonal_order: Optional[Tuple[int, int, int, int]] = None) -> None:
        """
        Fit ARIMA/SARIMA model to the data.
        
        Args:
            series: Training time series data
            order: ARIMA order (p, d, q). If None, uses auto search
            seasonal_order: Seasonal ARIMA order (P, D, Q, s). If None, uses default
        """
        self.train_data = series.copy()
        clean_series = series.dropna()
        
        # Use provided parameters or find optimal ones
        if order is None:
            if self.best_params is None:
                self.auto_arima_search(clean_series)
            order = self.best_params['order']
            
        if self.seasonal and seasonal_order is None:
            seasonal_order = self.best_params.get('seasonal_order', (1, 1, 1, self.seasonal_periods))
        
        logger.info(f"Fitting {'SARIMA' if self.seasonal else 'ARIMA'} model...")
        logger.info(f"  Order: {order}")
        if seasonal_order:
            logger.info(f"  Seasonal order: {seasonal_order}")
        
        try:
            if self.seasonal:
                self.model = SARIMAX(
                    clean_series,
                    order=order,
                    seasonal_order=seasonal_order,
                    enforce_stationarity=False,
                    enforce_invertibility=False
                )
            else:
                self.model = ARIMA(clean_series, order=order)
            
            self.fitted_model = self.model.fit()
            self.is_fitted = True
            
            logger.info(f"Model fitted successfully!")
            logger.info(f"  AIC: {self.fitted_model.aic:.2f}")
            logger.info(f"  BIC: {self.fitted_model.bic:.2f}")
            logger.info(f"  Log Likelihood: {self.fitted_model.llf:.2f}")
            
        except Exception as e:
            logger.error(f"Failed to fit model: {str(e)}")
            raise
    
    def forecast(self, steps: int, confidence_intervals: bool = True,
                alpha: float = 0.05) -> Dict[str, np.ndarray]:
        """
        Generate forecasts using the fitted model.
        
        Args:
            steps: Number of steps to forecast
            confidence_intervals: Whether to return confidence intervals
            alpha: Significance level for confidence intervals
            
        Returns:
            Dictionary with forecasts and confidence intervals
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before forecasting")
        
        logger.info(f"Generating {steps}-step forecast...")
        
        try:
            # Generate forecast
            forecast_result = self.fitted_model.forecast(
                steps=steps,
                alpha=alpha
            )
            
            forecasts = forecast_result
            
            # Get confidence intervals if requested
            conf_int = None
            if confidence_intervals:
                get_forecast = self.fitted_model.get_forecast(steps=steps, alpha=alpha)
                conf_int = get_forecast.conf_int()
            
            result = {
                'forecasts': forecasts,
                'conf_int_lower': conf_int.iloc[:, 0].values if conf_int is not None else None,
                'conf_int_upper': conf_int.iloc[:, 1].values if conf_int is not None else None
            }
            
            logger.info(f"Forecast generated successfully!")
            logger.info(f"  Forecast range: ${forecasts.min():.2f} to ${forecasts.max():.2f}")
            
            return result
            
        except Exception as e:
            logger.error(f"Forecasting failed: {str(e)}")
            raise
    
    def predict(self, start: int, end: int) -> np.ndarray:
        """
        Generate in-sample and out-of-sample predictions.
        
        Args:
            start: Start index for prediction
            end: End index for prediction
            
        Returns:
            Array of predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        try:
            predictions = self.fitted_model.predict(start=start, end=end)
            return predictions.values
            
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise
    
    def residual_diagnostics(self, figsize: Tuple[int, int] = (15, 10)) -> Dict[str, any]:
        """
        Perform residual diagnostics on the fitted model.
        
        Args:
            figsize: Figure size for plots
            
        Returns:
            Dictionary with diagnostic test results
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before diagnostics")
        
        residuals = self.fitted_model.resid
        
        # Ljung-Box test for autocorrelation
        lb_test = acorr_ljungbox(residuals, lags=10, return_df=True)
        
        # Normality test (Jarque-Bera)
        from scipy.stats import jarque_bera
        jb_stat, jb_pvalue = jarque_bera(residuals.dropna())
        
        diagnostics = {
            'ljung_box': {
                'statistic': lb_test['lb_stat'].iloc[-1],
                'p_value': lb_test['lb_pvalue'].iloc[-1],
                'passes': lb_test['lb_pvalue'].iloc[-1] > 0.05
            },
            'jarque_bera': {
                'statistic': jb_stat,
                'p_value': jb_pvalue,
                'passes': jb_pvalue > 0.05
            },
            'residual_stats': {
                'mean': residuals.mean(),
                'std': residuals.std(),
                'skewness': residuals.skew(),
                'kurtosis': residuals.kurtosis()
            }
        }
        
        # Plot diagnostics
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('Residual Diagnostics', fontsize=16, fontweight='bold')
        
        # Residuals over time
        axes[0, 0].plot(residuals)
        axes[0, 0].axhline(y=0, color='red', linestyle='--', alpha=0.7)
        axes[0, 0].set_title('Residuals vs Time')
        axes[0, 0].set_ylabel('Residuals')
        axes[0, 0].grid(True, alpha=0.3)
        
        # ACF of residuals
        plot_acf(residuals.dropna(), ax=axes[0, 1], lags=20, alpha=0.05)
        axes[0, 1].set_title('ACF of Residuals')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Histogram of residuals
        axes[1, 0].hist(residuals.dropna(), bins=30, alpha=0.7, density=True)
        axes[1, 0].set_title('Distribution of Residuals')
        axes[1, 0].set_xlabel('Residuals')
        axes[1, 0].set_ylabel('Density')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Q-Q plot
        from scipy import stats
        stats.probplot(residuals.dropna(), dist="norm", plot=axes[1, 1])
        axes[1, 1].set_title('Q-Q Plot (Normal Distribution)')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        logger.info("Residual Diagnostics:")
        logger.info(f"  Ljung-Box test p-value: {diagnostics['ljung_box']['p_value']:.6f}")
        logger.info(f"  Jarque-Bera test p-value: {diagnostics['jarque_bera']['p_value']:.6f}")
        logger.info(f"  Residual mean: {diagnostics['residual_stats']['mean']:.6f}")
        logger.info(f"  Residual std: {diagnostics['residual_stats']['std']:.6f}")
        
        return diagnostics
    
    def get_model_summary(self) -> str:
        """
        Get detailed model summary.
        
        Returns:
            Model summary as string
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted to get summary")
        
        return str(self.fitted_model.summary())


def main():
    """
    Main function to demonstrate ARIMA forecasting.
    """
    logger.info("ARIMA Forecasting module loaded successfully!")
    logger.info("Available features:")
    logger.info("  - ARIMA and SARIMA model implementation")
    logger.info("  - Automatic parameter optimization (auto_arima)")
    logger.info("  - Grid search parameter tuning")
    logger.info("  - Comprehensive model diagnostics")
    logger.info("  - Forecasting with confidence intervals")


if __name__ == "__main__":
    main()

