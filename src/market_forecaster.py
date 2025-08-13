"""
Market Forecaster - Advanced time series forecasting for future market trends.
Provides comprehensive forecasting capabilities with confidence intervals, trend analysis, and risk assessment.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union, Any
import warnings
import logging
from datetime import datetime, timedelta
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')


class MarketForecaster:
    """
    Advanced market forecasting class for generating future predictions and trend analysis.
    """
    
    def __init__(self):
        """Initialize the market forecaster."""
        self.forecasts = {}
        self.confidence_intervals = {}
        self.trend_analysis = {}
        self.risk_metrics = {}
        self.historical_data = None
        self.forecast_dates = None
        
    def generate_arima_forecast(self, arima_model, steps: int = 180, 
                               confidence_level: float = 0.95) -> Dict[str, Any]:
        """
        Generate ARIMA forecast with confidence intervals.
        
        Args:
            arima_model: Fitted ARIMA model
            steps: Number of steps to forecast (default 180 for ~6 months)
            confidence_level: Confidence level for intervals
            
        Returns:
            Dictionary with forecast results
        """
        try:
            # Generate forecast
            forecast_result = arima_model.forecast(steps=steps, alpha=1-confidence_level)
            
            if hasattr(forecast_result, 'predicted_mean'):
                # statsmodels format
                forecast = forecast_result.predicted_mean
                conf_int = forecast_result.conf_int()
                lower_bound = conf_int.iloc[:, 0] if hasattr(conf_int, 'iloc') else conf_int[:, 0]
                upper_bound = conf_int.iloc[:, 1] if hasattr(conf_int, 'iloc') else conf_int[:, 1]
            else:
                # pmdarima format
                forecast = forecast_result
                # Generate confidence intervals using residual standard error
                residuals = arima_model.resid()
                std_error = np.std(residuals)
                z_score = stats.norm.ppf((1 + confidence_level) / 2)
                margin_error = z_score * std_error * np.sqrt(np.arange(1, steps + 1))
                lower_bound = forecast - margin_error
                upper_bound = forecast + margin_error
            
            # Create date index for forecast
            last_date = self.historical_data.index[-1] if self.historical_data is not None else pd.Timestamp.now()
            forecast_dates = pd.date_range(start=last_date + timedelta(days=1), 
                                         periods=steps, freq='D')
            
            result = {
                'forecast': pd.Series(forecast, index=forecast_dates),
                'lower_bound': pd.Series(lower_bound, index=forecast_dates),
                'upper_bound': pd.Series(upper_bound, index=forecast_dates),
                'confidence_level': confidence_level,
                'model_type': 'ARIMA',
                'forecast_horizon': steps
            }
            
            logger.info(f"Generated ARIMA forecast for {steps} days with {confidence_level:.0%} confidence")
            return result
            
        except Exception as e:
            logger.error(f"ARIMA forecast generation failed: {str(e)}")
            return {}
    
    def generate_lstm_forecast(self, lstm_model, scaler, last_sequence: np.ndarray, 
                              steps: int = 180, confidence_level: float = 0.95,
                              monte_carlo_samples: int = 100) -> Dict[str, Any]:
        """
        Generate LSTM forecast with confidence intervals using Monte Carlo simulation.
        
        Args:
            lstm_model: Trained LSTM model
            scaler: Fitted scaler for data normalization
            last_sequence: Last sequence from training data
            steps: Number of steps to forecast
            confidence_level: Confidence level for intervals
            monte_carlo_samples: Number of Monte Carlo samples for uncertainty estimation
            
        Returns:
            Dictionary with forecast results
        """
        try:
            # Multi-step forecast using recursive prediction
            forecasts = []
            current_sequence = last_sequence.copy()
            
            # Generate multiple forecast paths for uncertainty estimation
            all_paths = []
            
            for sample in range(monte_carlo_samples):
                sample_forecasts = []
                sample_sequence = current_sequence.copy()
                
                for step in range(steps):
                    # Predict next value
                    next_pred = lstm_model.predict(sample_sequence.reshape(1, -1, 1), verbose=0)
                    next_value = next_pred[0, 0]
                    
                    # Add noise for uncertainty estimation (if not first sample)
                    if sample > 0:
                        # Estimate noise from model's training performance
                        noise_std = 0.02  # Can be estimated from validation errors
                        noise = np.random.normal(0, noise_std)
                        next_value += noise
                    
                    sample_forecasts.append(next_value)
                    
                    # Update sequence for next prediction
                    sample_sequence = np.append(sample_sequence[1:], next_value)
                
                all_paths.append(sample_forecasts)
            
            # Calculate statistics from all paths
            all_paths = np.array(all_paths)
            forecast = np.mean(all_paths, axis=0)
            
            # Calculate confidence intervals
            alpha = 1 - confidence_level
            lower_percentile = (alpha / 2) * 100
            upper_percentile = (1 - alpha / 2) * 100
            
            lower_bound = np.percentile(all_paths, lower_percentile, axis=0)
            upper_bound = np.percentile(all_paths, upper_percentile, axis=0)
            
            # Inverse transform predictions
            forecast = scaler.inverse_transform(forecast.reshape(-1, 1)).flatten()
            lower_bound = scaler.inverse_transform(lower_bound.reshape(-1, 1)).flatten()
            upper_bound = scaler.inverse_transform(upper_bound.reshape(-1, 1)).flatten()
            
            # Create date index for forecast
            last_date = self.historical_data.index[-1] if self.historical_data is not None else pd.Timestamp.now()
            forecast_dates = pd.date_range(start=last_date + timedelta(days=1), 
                                         periods=steps, freq='D')
            
            result = {
                'forecast': pd.Series(forecast, index=forecast_dates),
                'lower_bound': pd.Series(lower_bound, index=forecast_dates),
                'upper_bound': pd.Series(upper_bound, index=forecast_dates),
                'confidence_level': confidence_level,
                'model_type': 'LSTM',
                'forecast_horizon': steps,
                'monte_carlo_samples': monte_carlo_samples
            }
            
            logger.info(f"Generated LSTM forecast for {steps} days with {confidence_level:.0%} confidence using {monte_carlo_samples} MC samples")
            return result
            
        except Exception as e:
            logger.error(f"LSTM forecast generation failed: {str(e)}")
            return {}
    
    def set_historical_data(self, data: pd.Series) -> None:
        """
        Set historical data for context in forecasting.
        
        Args:
            data: Historical time series data
        """
        self.historical_data = data
        logger.info(f"Historical data set: {len(data)} observations from {data.index.min()} to {data.index.max()}")
    
    def analyze_forecast_trends(self, forecast_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze trends in forecast data.
        
        Args:
            forecast_data: Forecast results dictionary
            
        Returns:
            Dictionary with trend analysis results
        """
        if not forecast_data:
            return {}
        
        forecast = forecast_data['forecast']
        
        # Overall trend analysis
        start_price = forecast.iloc[0]
        end_price = forecast.iloc[-1]
        total_return = (end_price - start_price) / start_price * 100
        
        # Monthly trend analysis
        monthly_returns = []
        monthly_volatility = []
        
        # Group by month and calculate returns
        monthly_data = forecast.resample('M').last()
        for i in range(1, len(monthly_data)):
            monthly_return = (monthly_data.iloc[i] - monthly_data.iloc[i-1]) / monthly_data.iloc[i-1] * 100
            monthly_returns.append(monthly_return)
        
        # Calculate rolling volatility
        daily_returns = forecast.pct_change().dropna() * 100
        rolling_volatility = daily_returns.rolling(window=30).std()
        
        # Identify trend direction
        if total_return > 5:
            trend_direction = "Bullish"
        elif total_return < -5:
            trend_direction = "Bearish"
        else:
            trend_direction = "Sideways"
        
        # Identify potential turning points (local minima/maxima)
        turning_points = []
        for i in range(1, len(forecast) - 1):
            if (forecast.iloc[i] > forecast.iloc[i-1] and forecast.iloc[i] > forecast.iloc[i+1]) or \
               (forecast.iloc[i] < forecast.iloc[i-1] and forecast.iloc[i] < forecast.iloc[i+1]):
                turning_points.append({
                    'date': forecast.index[i],
                    'price': forecast.iloc[i],
                    'type': 'local_max' if forecast.iloc[i] > forecast.iloc[i-1] else 'local_min'
                })
        
        # Calculate trend strength
        correlation_with_time = np.corrcoef(range(len(forecast)), forecast.values)[0, 1]
        trend_strength = abs(correlation_with_time)
        
        analysis = {
            'overall_trend': {
                'direction': trend_direction,
                'total_return': total_return,
                'start_price': start_price,
                'end_price': end_price,
                'trend_strength': trend_strength
            },
            'monthly_analysis': {
                'avg_monthly_return': np.mean(monthly_returns) if monthly_returns else 0,
                'monthly_volatility': np.std(monthly_returns) if monthly_returns else 0,
                'best_month': max(monthly_returns) if monthly_returns else 0,
                'worst_month': min(monthly_returns) if monthly_returns else 0
            },
            'volatility_analysis': {
                'avg_daily_volatility': daily_returns.std(),
                'max_daily_change': abs(daily_returns).max(),
                'volatility_trend': 'increasing' if rolling_volatility.iloc[-30:].mean() > rolling_volatility.iloc[:30].mean() else 'decreasing'
            },
            'turning_points': turning_points[:10],  # Top 10 turning points
            'forecast_period': f"{forecast.index[0].strftime('%Y-%m-%d')} to {forecast.index[-1].strftime('%Y-%m-%d')}"
        }
        
        return analysis
    
    def assess_forecast_risk(self, forecast_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Assess risk metrics for forecast.
        
        Args:
            forecast_data: Forecast results dictionary
            
        Returns:
            Dictionary with risk assessment results
        """
        if not forecast_data:
            return {}
        
        forecast = forecast_data['forecast']
        lower_bound = forecast_data['lower_bound']
        upper_bound = forecast_data['upper_bound']
        
        # Confidence interval width analysis
        ci_width = upper_bound - lower_bound
        avg_ci_width = ci_width.mean()
        ci_width_trend = 'expanding' if ci_width.iloc[-30:].mean() > ci_width.iloc[:30].mean() else 'contracting'
        
        # Relative confidence interval width
        relative_ci_width = (ci_width / forecast) * 100
        
        # Value at Risk (VaR) estimation
        forecast_returns = forecast.pct_change().dropna()
        var_95 = np.percentile(forecast_returns, 5) * 100  # 95% VaR
        var_99 = np.percentile(forecast_returns, 1) * 100  # 99% VaR
        
        # Maximum drawdown estimation
        cumulative_returns = (1 + forecast_returns).cumprod()
        rolling_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - rolling_max) / rolling_max
        max_drawdown = drawdown.min() * 100
        
        # Forecast uncertainty metrics
        uncertainty_metrics = {
            'avg_ci_width_dollars': avg_ci_width,
            'avg_ci_width_percent': relative_ci_width.mean(),
            'max_ci_width_percent': relative_ci_width.max(),
            'ci_width_trend': ci_width_trend,
            'forecast_range': {
                'min_forecast': forecast.min(),
                'max_forecast': forecast.max(),
                'min_lower_bound': lower_bound.min(),
                'max_upper_bound': upper_bound.max()
            }
        }
        
        # Risk assessment
        risk_level = 'Low'
        if relative_ci_width.mean() > 20:
            risk_level = 'High'
        elif relative_ci_width.mean() > 10:
            risk_level = 'Medium'
        
        risk_assessment = {
            'uncertainty_metrics': uncertainty_metrics,
            'risk_metrics': {
                'var_95': var_95,
                'var_99': var_99,
                'max_drawdown': max_drawdown,
                'volatility': forecast_returns.std() * np.sqrt(252) * 100  # Annualized
            },
            'risk_level': risk_level,
            'reliability_assessment': {
                'short_term': 'High' if relative_ci_width.iloc[:30].mean() < 10 else 'Medium',
                'medium_term': 'Medium' if relative_ci_width.iloc[30:90].mean() < 15 else 'Low',
                'long_term': 'Low' if relative_ci_width.iloc[90:].mean() > 20 else 'Medium'
            }
        }
        
        return risk_assessment
    
    def identify_market_opportunities(self, forecast_data: Dict[str, Any], 
                                    trend_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Identify market opportunities and risks based on forecast.
        
        Args:
            forecast_data: Forecast results dictionary
            trend_analysis: Trend analysis results
            
        Returns:
            Dictionary with opportunities and risks
        """
        forecast = forecast_data['forecast']
        lower_bound = forecast_data['lower_bound']
        upper_bound = forecast_data['upper_bound']
        
        opportunities = []
        risks = []
        
        # Price movement opportunities
        if trend_analysis['overall_trend']['total_return'] > 10:
            opportunities.append({
                'type': 'Price Appreciation',
                'description': f"Forecast suggests {trend_analysis['overall_trend']['total_return']:.1f}% price increase over forecast period",
                'confidence': 'High' if trend_analysis['overall_trend']['trend_strength'] > 0.7 else 'Medium',
                'timeframe': 'Long-term'
            })
        
        # Volatility opportunities
        if trend_analysis['volatility_analysis']['avg_daily_volatility'] > 3:
            opportunities.append({
                'type': 'Trading Volatility',
                'description': f"High volatility ({trend_analysis['volatility_analysis']['avg_daily_volatility']:.1f}%) presents trading opportunities",
                'confidence': 'Medium',
                'timeframe': 'Short-term'
            })
        
        # Identify specific buying opportunities (local minima)
        turning_points = trend_analysis['turning_points']
        buy_opportunities = [tp for tp in turning_points if tp['type'] == 'local_min']
        if buy_opportunities:
            opportunities.append({
                'type': 'Strategic Entry Points',
                'description': f"Identified {len(buy_opportunities)} potential buying opportunities at local price minima",
                'confidence': 'Medium',
                'timeframe': 'Medium-term'
            })
        
        # Risk identification
        if trend_analysis['overall_trend']['total_return'] < -10:
            risks.append({
                'type': 'Price Decline Risk',
                'description': f"Forecast suggests {abs(trend_analysis['overall_trend']['total_return']):.1f}% price decline",
                'severity': 'High',
                'probability': 'Medium'
            })
        
        # Uncertainty risk
        avg_ci_width = ((upper_bound - lower_bound) / forecast * 100).mean()
        if avg_ci_width > 20:
            risks.append({
                'type': 'High Forecast Uncertainty',
                'description': f"Wide confidence intervals ({avg_ci_width:.1f}% average width) indicate high uncertainty",
                'severity': 'Medium',
                'probability': 'High'
            })
        
        # Volatility risk
        if trend_analysis['volatility_analysis']['volatility_trend'] == 'increasing':
            risks.append({
                'type': 'Increasing Volatility',
                'description': "Forecast shows increasing volatility over time, suggesting higher risk",
                'severity': 'Medium',
                'probability': 'Medium'
            })
        
        return {
            'opportunities': opportunities,
            'risks': risks,
            'overall_sentiment': 'Bullish' if len(opportunities) > len(risks) else 'Bearish' if len(risks) > len(opportunities) else 'Neutral'
        }
    
    def visualize_forecast(self, forecast_data: Dict[str, Any], 
                          historical_lookback: int = 180,
                          figsize: Tuple[int, int] = (15, 10)) -> None:
        """
        Visualize forecast with historical data and confidence intervals.
        
        Args:
            forecast_data: Forecast results dictionary
            historical_lookback: Days of historical data to show
            figsize: Figure size
        """
        if not forecast_data:
            logger.warning("No forecast data to visualize")
            return
        
        forecast = forecast_data['forecast']
        lower_bound = forecast_data['lower_bound']
        upper_bound = forecast_data['upper_bound']
        
        # Prepare historical data for plotting
        historical_data = self.historical_data
        if historical_data is not None and historical_lookback:
            historical_data = historical_data.tail(historical_lookback)
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle(f'Tesla Stock Price Forecast - {forecast_data["model_type"]} Model', 
                     fontsize=16, fontweight='bold')
        
        # Main forecast plot
        ax = axes[0, 0]
        if historical_data is not None:
            ax.plot(historical_data.index, historical_data.values, 
                   label='Historical', color='blue', linewidth=2)
        
        ax.plot(forecast.index, forecast.values, 
               label='Forecast', color='red', linewidth=2)
        ax.fill_between(forecast.index, lower_bound.values, upper_bound.values,
                       alpha=0.3, color='red', label=f'{forecast_data["confidence_level"]:.0%} Confidence Interval')
        
        ax.set_title('Price Forecast with Confidence Intervals')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price ($)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Forecast returns distribution
        ax = axes[0, 1]
        forecast_returns = forecast.pct_change().dropna() * 100
        ax.hist(forecast_returns, bins=30, alpha=0.7, edgecolor='black')
        ax.set_title('Forecast Daily Returns Distribution')
        ax.set_xlabel('Daily Return (%)')
        ax.set_ylabel('Frequency')
        ax.grid(True, alpha=0.3)
        
        # Confidence interval width over time
        ax = axes[1, 0]
        ci_width = upper_bound - lower_bound
        relative_ci_width = (ci_width / forecast) * 100
        ax.plot(forecast.index, relative_ci_width.values, color='orange', linewidth=2)
        ax.set_title('Confidence Interval Width Over Time')
        ax.set_xlabel('Date')
        ax.set_ylabel('CI Width (% of Forecast)')
        ax.grid(True, alpha=0.3)
        
        # Price level analysis
        ax = axes[1, 1]
        price_levels = pd.DataFrame({
            'Forecast': forecast,
            'Lower Bound': lower_bound,
            'Upper Bound': upper_bound
        })
        price_levels.plot(ax=ax, alpha=0.8)
        ax.set_title('Forecast Price Levels')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price ($)')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def generate_interactive_forecast_plot(self, forecast_data: Dict[str, Any], 
                                         historical_lookback: int = 180) -> go.Figure:
        """
        Generate interactive Plotly forecast visualization.
        
        Args:
            forecast_data: Forecast results dictionary
            historical_lookback: Days of historical data to show
            
        Returns:
            Plotly figure object
        """
        if not forecast_data:
            return go.Figure()
        
        forecast = forecast_data['forecast']
        lower_bound = forecast_data['lower_bound']
        upper_bound = forecast_data['upper_bound']
        
        # Prepare historical data
        historical_data = self.historical_data
        if historical_data is not None and historical_lookback:
            historical_data = historical_data.tail(historical_lookback)
        
        fig = go.Figure()
        
        # Add historical data
        if historical_data is not None:
            fig.add_trace(go.Scatter(
                x=historical_data.index,
                y=historical_data.values,
                mode='lines',
                name='Historical Data',
                line=dict(color='blue', width=2)
            ))
        
        # Add forecast
        fig.add_trace(go.Scatter(
            x=forecast.index,
            y=forecast.values,
            mode='lines',
            name='Forecast',
            line=dict(color='red', width=2)
        ))
        
        # Add confidence interval
        fig.add_trace(go.Scatter(
            x=forecast.index,
            y=upper_bound.values,
            mode='lines',
            line=dict(width=0),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        fig.add_trace(go.Scatter(
            x=forecast.index,
            y=lower_bound.values,
            mode='lines',
            line=dict(width=0),
            fill='tonexty',
            fillcolor='rgba(255, 0, 0, 0.2)',
            name=f'{forecast_data["confidence_level"]:.0%} Confidence Interval',
            hovertemplate='Lower Bound: $%{y:.2f}<extra></extra>'
        ))
        
        fig.update_layout(
            title=f'Tesla Stock Price Forecast - {forecast_data["model_type"]} Model',
            xaxis_title='Date',
            yaxis_title='Price ($)',
            hovermode='x unified',
            template='plotly_white'
        )
        
        return fig
    
    def generate_comprehensive_report(self, forecast_data: Dict[str, Any]) -> str:
        """
        Generate comprehensive forecast analysis report.
        
        Args:
            forecast_data: Forecast results dictionary
            
        Returns:
            Formatted report string
        """
        if not forecast_data:
            return "No forecast data available for report generation."
        
        # Perform analysis
        trend_analysis = self.analyze_forecast_trends(forecast_data)
        risk_assessment = self.assess_forecast_risk(forecast_data)
        opportunities = self.identify_market_opportunities(forecast_data, trend_analysis)
        
        report = []
        report.append("=" * 80)
        report.append("TESLA STOCK PRICE FORECAST ANALYSIS REPORT")
        report.append("=" * 80)
        report.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Model Type: {forecast_data['model_type']}")
        report.append(f"Forecast Period: {trend_analysis['forecast_period']}")
        report.append(f"Confidence Level: {forecast_data['confidence_level']:.0%}")
        
        # Executive Summary
        report.append("\n🎯 EXECUTIVE SUMMARY")
        report.append("-" * 40)
        report.append(f"Overall Trend: {trend_analysis['overall_trend']['direction']}")
        report.append(f"Expected Return: {trend_analysis['overall_trend']['total_return']:.1f}%")
        report.append(f"Risk Level: {risk_assessment['risk_level']}")
        report.append(f"Market Sentiment: {opportunities['overall_sentiment']}")
        
        # Detailed Trend Analysis
        report.append("\n📈 TREND ANALYSIS")
        report.append("-" * 40)
        report.append(f"Start Price: ${trend_analysis['overall_trend']['start_price']:.2f}")
        report.append(f"End Price: ${trend_analysis['overall_trend']['end_price']:.2f}")
        report.append(f"Trend Strength: {trend_analysis['overall_trend']['trend_strength']:.2f}")
        report.append(f"Average Monthly Return: {trend_analysis['monthly_analysis']['avg_monthly_return']:.1f}%")
        report.append(f"Monthly Volatility: {trend_analysis['monthly_analysis']['monthly_volatility']:.1f}%")
        
        # Risk Assessment
        report.append("\n⚠️ RISK ASSESSMENT")
        report.append("-" * 40)
        report.append(f"Average CI Width: {risk_assessment['uncertainty_metrics']['avg_ci_width_percent']:.1f}%")
        report.append(f"95% VaR: {risk_assessment['risk_metrics']['var_95']:.2f}%")
        report.append(f"Maximum Drawdown: {risk_assessment['risk_metrics']['max_drawdown']:.1f}%")
        report.append(f"Annualized Volatility: {risk_assessment['risk_metrics']['volatility']:.1f}%")
        
        # Reliability Assessment
        report.append("\n🎯 FORECAST RELIABILITY")
        report.append("-" * 40)
        reliability = risk_assessment['reliability_assessment']
        report.append(f"Short-term (1-3 months): {reliability['short_term']}")
        report.append(f"Medium-term (3-6 months): {reliability['medium_term']}")
        report.append(f"Long-term (6+ months): {reliability['long_term']}")
        
        # Market Opportunities
        report.append("\n💰 MARKET OPPORTUNITIES")
        report.append("-" * 40)
        for i, opp in enumerate(opportunities['opportunities'], 1):
            report.append(f"{i}. {opp['type']} ({opp['confidence']} confidence)")
            report.append(f"   {opp['description']}")
        
        # Market Risks
        report.append("\n🚨 MARKET RISKS")
        report.append("-" * 40)
        for i, risk in enumerate(opportunities['risks'], 1):
            report.append(f"{i}. {risk['type']} ({risk['severity']} severity)")
            report.append(f"   {risk['description']}")
        
        # Key Insights and Recommendations
        report.append("\n💡 KEY INSIGHTS & RECOMMENDATIONS")
        report.append("-" * 40)
        
        # Generate insights based on analysis
        if trend_analysis['overall_trend']['total_return'] > 15:
            report.append("• Strong bullish trend expected - Consider long positions")
        elif trend_analysis['overall_trend']['total_return'] > 5:
            report.append("• Moderate upward trend - Cautiously optimistic outlook")
        elif trend_analysis['overall_trend']['total_return'] < -15:
            report.append("• Strong bearish trend expected - Consider defensive strategies")
        else:
            report.append("• Sideways movement expected - Range-bound trading strategies")
        
        if risk_assessment['uncertainty_metrics']['avg_ci_width_percent'] > 20:
            report.append("• High forecast uncertainty - Use shorter-term strategies")
        
        if risk_assessment['uncertainty_metrics']['ci_width_trend'] == 'expanding':
            report.append("• Increasing uncertainty over time - Confidence decreases with time horizon")
        
        report.append("\n" + "=" * 80)
        
        return "\n".join(report)


def main():
    """
    Main function to demonstrate market forecaster functionality.
    """
    logger.info("Market Forecaster module loaded successfully!")
    logger.info("Available features:")
    logger.info("  - ARIMA and LSTM forecast generation")
    logger.info("  - Confidence interval calculation")
    logger.info("  - Trend and risk analysis")
    logger.info("  - Market opportunity identification")
    logger.info("  - Interactive visualization")
    logger.info("  - Comprehensive reporting")


if __name__ == "__main__":
    main()
