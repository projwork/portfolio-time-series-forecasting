"""
Exploratory Data Analysis module for portfolio time series analysis.
Provides comprehensive visualization and analysis tools for financial data.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.tsa.stattools import adfuller
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from typing import Dict, List, Tuple, Optional
import warnings
import logging

# Configure plotting
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EDAAnalyzer:
    """
    A class to perform comprehensive exploratory data analysis on financial time series data.
    """
    
    def __init__(self, data: Dict[str, pd.DataFrame]):
        """
        Initialize the EDA analyzer with processed data.
        
        Args:
            data: Dictionary of DataFrames with asset symbols as keys
        """
        self.data = data
        self.figures = {}
        self.analysis_results = {}
        
    def plot_price_trends(self, figsize: Tuple[int, int] = (15, 10), save_path: Optional[str] = None) -> None:
        """
        Plot closing price trends for all assets.
        
        Args:
            figsize: Figure size for matplotlib plots
            save_path: Path to save the plot (optional)
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('Portfolio Assets - Price Trends Analysis', fontsize=16, fontweight='bold')
        
        # Individual price trends
        for i, (symbol, data) in enumerate(self.data.items()):
            if 'Date' not in data.columns or 'Close' not in data.columns:
                continue
                
            row, col = i // 2, i % 2
            ax = axes[row, col]
            
            ax.plot(data['Date'], data['Close'], linewidth=2, label=f'{symbol} Close Price')
            ax.set_title(f'{symbol} - Closing Price Over Time', fontweight='bold')
            ax.set_xlabel('Date')
            ax.set_ylabel('Price ($)')
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # Add trend line
            x_numeric = range(len(data))
            z = np.polyfit(x_numeric, data['Close'], 1)
            p = np.poly1d(z)
            ax.plot(data['Date'], p(x_numeric), "--", alpha=0.7, color='red', label='Trend Line')
            ax.legend()
        
        # Combined comparison plot
        ax_combined = axes[1, 1]
        for symbol, data in self.data.items():
            if 'Date' not in data.columns or 'Close' not in data.columns:
                continue
            # Normalize prices to start at 100 for comparison
            normalized_price = (data['Close'] / data['Close'].iloc[0]) * 100
            ax_combined.plot(data['Date'], normalized_price, linewidth=2, label=f'{symbol}')
        
        ax_combined.set_title('Normalized Price Comparison (Base = 100)', fontweight='bold')
        ax_combined.set_xlabel('Date')
        ax_combined.set_ylabel('Normalized Price')
        ax_combined.grid(True, alpha=0.3)
        ax_combined.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Price trends plot saved to {save_path}")
        
        plt.show()
        self.figures['price_trends'] = fig
    
    def plot_returns_analysis(self, figsize: Tuple[int, int] = (15, 12), save_path: Optional[str] = None) -> None:
        """
        Plot returns analysis including daily returns and distributions.
        
        Args:
            figsize: Figure size for matplotlib plots
            save_path: Path to save the plot (optional)
        """
        fig, axes = plt.subplots(3, len(self.data), figsize=figsize)
        fig.suptitle('Portfolio Assets - Returns Analysis', fontsize=16, fontweight='bold')
        
        for i, (symbol, data) in enumerate(self.data.items()):
            if 'Daily_Return' not in data.columns:
                continue
            
            # Daily returns time series
            axes[0, i].plot(data['Date'], data['Daily_Return'] * 100, alpha=0.7, linewidth=1)
            axes[0, i].axhline(y=0, color='red', linestyle='--', alpha=0.7)
            axes[0, i].set_title(f'{symbol} - Daily Returns (%)')
            axes[0, i].set_ylabel('Daily Return (%)')
            axes[0, i].grid(True, alpha=0.3)
            
            # Returns distribution
            axes[1, i].hist(data['Daily_Return'].dropna() * 100, bins=50, alpha=0.7, density=True)
            axes[1, i].axvline(data['Daily_Return'].mean() * 100, color='red', linestyle='--', 
                              label=f'Mean: {data["Daily_Return"].mean()*100:.3f}%')
            axes[1, i].set_title(f'{symbol} - Returns Distribution')
            axes[1, i].set_xlabel('Daily Return (%)')
            axes[1, i].set_ylabel('Density')
            axes[1, i].legend()
            axes[1, i].grid(True, alpha=0.3)
            
            # Cumulative returns
            if 'Cumulative_Return' in data.columns:
                axes[2, i].plot(data['Date'], data['Cumulative_Return'] * 100, linewidth=2)
                axes[2, i].set_title(f'{symbol} - Cumulative Returns')
                axes[2, i].set_xlabel('Date')
                axes[2, i].set_ylabel('Cumulative Return (%)')
                axes[2, i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Returns analysis plot saved to {save_path}")
        
        plt.show()
        self.figures['returns_analysis'] = fig
    
    def plot_volatility_analysis(self, window: int = 30, figsize: Tuple[int, int] = (15, 10), 
                                save_path: Optional[str] = None) -> None:
        """
        Plot volatility analysis with rolling statistics.
        
        Args:
            window: Rolling window size for volatility calculation
            figsize: Figure size for matplotlib plots
            save_path: Path to save the plot (optional)
        """
        fig, axes = plt.subplots(2, len(self.data), figsize=figsize)
        fig.suptitle(f'Portfolio Assets - Volatility Analysis (Rolling {window}-day)', 
                    fontsize=16, fontweight='bold')
        
        for i, (symbol, data) in enumerate(self.data.items()):
            if 'Daily_Return' not in data.columns:
                continue
            
            # Calculate rolling volatility
            rolling_vol = data['Daily_Return'].rolling(window=window).std() * np.sqrt(252) * 100
            rolling_mean = data['Daily_Return'].rolling(window=window).mean() * 100
            
            # Rolling volatility
            axes[0, i].plot(data['Date'], rolling_vol, linewidth=2, color='red', label='Rolling Volatility')
            axes[0, i].fill_between(data['Date'], rolling_vol, alpha=0.3, color='red')
            axes[0, i].set_title(f'{symbol} - Rolling Volatility')
            axes[0, i].set_ylabel('Annualized Volatility (%)')
            axes[0, i].grid(True, alpha=0.3)
            axes[0, i].legend()
            
            # Rolling mean vs volatility
            ax_twin = axes[0, i].twinx()
            ax_twin.plot(data['Date'], rolling_mean, linewidth=2, color='blue', alpha=0.7, label='Rolling Mean Return')
            ax_twin.set_ylabel('Rolling Mean Return (%)', color='blue')
            ax_twin.legend(loc='upper right')
            
            # Volatility distribution
            axes[1, i].hist(rolling_vol.dropna(), bins=30, alpha=0.7, density=True, color='red')
            axes[1, i].axvline(rolling_vol.mean(), color='darkred', linestyle='--', 
                              label=f'Mean: {rolling_vol.mean():.2f}%')
            axes[1, i].set_title(f'{symbol} - Volatility Distribution')
            axes[1, i].set_xlabel('Annualized Volatility (%)')
            axes[1, i].set_ylabel('Density')
            axes[1, i].legend()
            axes[1, i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Volatility analysis plot saved to {save_path}")
        
        plt.show()
        self.figures['volatility_analysis'] = fig
    
    def perform_stationarity_tests(self) -> Dict[str, Dict]:
        """
        Perform Augmented Dickey-Fuller test for stationarity.
        
        Returns:
            Dictionary containing test results for each asset and variable
        """
        stationarity_results = {}
        
        for symbol, data in self.data.items():
            logger.info(f"Performing stationarity tests for {symbol}...")
            
            stationarity_results[symbol] = {}
            
            # Test variables
            test_variables = ['Close', 'Daily_Return', 'Log_Return']
            
            for var in test_variables:
                if var not in data.columns:
                    continue
                
                # Remove missing values
                series = data[var].dropna()
                
                if len(series) < 10:  # Need sufficient data points
                    continue
                
                try:
                    # Perform ADF test
                    adf_result = adfuller(series, autolag='AIC')
                    
                    stationarity_results[symbol][var] = {
                        'adf_statistic': adf_result[0],
                        'p_value': adf_result[1],
                        'critical_values': adf_result[4],
                        'is_stationary': adf_result[1] < 0.05,  # 5% significance level
                        'interpretation': 'Stationary' if adf_result[1] < 0.05 else 'Non-stationary'
                    }
                    
                    logger.info(f"{symbol} {var}: ADF Statistic = {adf_result[0]:.4f}, "
                              f"p-value = {adf_result[1]:.4f}, "
                              f"Result: {stationarity_results[symbol][var]['interpretation']}")
                    
                except Exception as e:
                    logger.error(f"Error in stationarity test for {symbol} {var}: {str(e)}")
        
        self.analysis_results['stationarity'] = stationarity_results
        return stationarity_results
    
    def analyze_correlations(self, figsize: Tuple[int, int] = (12, 8), save_path: Optional[str] = None) -> pd.DataFrame:
        """
        Analyze correlations between assets.
        
        Args:
            figsize: Figure size for matplotlib plots
            save_path: Path to save the plot (optional)
            
        Returns:
            Correlation matrix DataFrame
        """
        # Combine closing prices
        price_data = {}
        return_data = {}
        
        for symbol, data in self.data.items():
            if 'Close' in data.columns and 'Daily_Return' in data.columns:
                price_data[f'{symbol}_Close'] = data.set_index('Date')['Close']
                return_data[f'{symbol}_Return'] = data.set_index('Date')['Daily_Return']
        
        # Create correlation matrices
        price_df = pd.DataFrame(price_data)
        return_df = pd.DataFrame(return_data)
        
        price_corr = price_df.corr()
        return_corr = return_df.corr()
        
        # Plot correlations
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        fig.suptitle('Asset Correlations Analysis', fontsize=16, fontweight='bold')
        
        # Price correlations
        sns.heatmap(price_corr, annot=True, cmap='coolwarm', center=0, 
                   square=True, ax=axes[0], cbar_kws={'label': 'Correlation'})
        axes[0].set_title('Price Correlations')
        
        # Return correlations
        sns.heatmap(return_corr, annot=True, cmap='coolwarm', center=0,
                   square=True, ax=axes[1], cbar_kws={'label': 'Correlation'})
        axes[1].set_title('Return Correlations')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Correlation analysis plot saved to {save_path}")
        
        plt.show()
        self.figures['correlations'] = fig
        
        self.analysis_results['correlations'] = {
            'price_correlations': price_corr,
            'return_correlations': return_corr
        }
        
        return return_corr
    
    def analyze_seasonality(self, figsize: Tuple[int, int] = (15, 10), save_path: Optional[str] = None) -> None:
        """
        Analyze seasonal patterns in returns.
        
        Args:
            figsize: Figure size for matplotlib plots
            save_path: Path to save the plot (optional)
        """
        fig, axes = plt.subplots(2, len(self.data), figsize=figsize)
        fig.suptitle('Seasonal Patterns Analysis', fontsize=16, fontweight='bold')
        
        for i, (symbol, data) in enumerate(self.data.items()):
            if 'Daily_Return' not in data.columns or 'Date' not in data.columns:
                continue
            
            # Extract time components
            data_copy = data.copy()
            data_copy['Month'] = pd.to_datetime(data_copy['Date']).dt.month
            data_copy['DayOfWeek'] = pd.to_datetime(data_copy['Date']).dt.dayofweek
            data_copy['Year'] = pd.to_datetime(data_copy['Date']).dt.year
            
            # Monthly seasonality
            monthly_returns = data_copy.groupby('Month')['Daily_Return'].mean() * 100
            axes[0, i].bar(range(1, 13), monthly_returns, alpha=0.7)
            axes[0, i].set_title(f'{symbol} - Average Returns by Month')
            axes[0, i].set_xlabel('Month')
            axes[0, i].set_ylabel('Average Daily Return (%)')
            axes[0, i].set_xticks(range(1, 13))
            axes[0, i].grid(True, alpha=0.3)
            
            # Day of week seasonality
            dow_returns = data_copy.groupby('DayOfWeek')['Daily_Return'].mean() * 100
            dow_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri']
            axes[1, i].bar(range(5), dow_returns[:5], alpha=0.7)
            axes[1, i].set_title(f'{symbol} - Average Returns by Day of Week')
            axes[1, i].set_xlabel('Day of Week')
            axes[1, i].set_ylabel('Average Daily Return (%)')
            axes[1, i].set_xticks(range(5))
            axes[1, i].set_xticklabels(dow_labels)
            axes[1, i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Seasonality analysis plot saved to {save_path}")
        
        plt.show()
        self.figures['seasonality'] = fig
    
    def create_interactive_dashboard(self) -> go.Figure:
        """
        Create an interactive dashboard using Plotly.
        
        Returns:
            Plotly figure object
        """
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Price Trends', 'Daily Returns', 'Volume', 'Volatility'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        colors = ['blue', 'red', 'green']
        
        for i, (symbol, data) in enumerate(self.data.items()):
            color = colors[i % len(colors)]
            
            if 'Date' not in data.columns:
                continue
            
            # Price trends
            if 'Close' in data.columns:
                fig.add_trace(
                    go.Scatter(x=data['Date'], y=data['Close'], name=f'{symbol} Price',
                              line=dict(color=color), mode='lines'),
                    row=1, col=1
                )
            
            # Daily returns
            if 'Daily_Return' in data.columns:
                fig.add_trace(
                    go.Scatter(x=data['Date'], y=data['Daily_Return']*100, name=f'{symbol} Returns',
                              line=dict(color=color), mode='lines'),
                    row=1, col=2
                )
            
            # Volume
            if 'Volume' in data.columns:
                fig.add_trace(
                    go.Scatter(x=data['Date'], y=data['Volume'], name=f'{symbol} Volume',
                              line=dict(color=color), mode='lines'),
                    row=2, col=1
                )
            
            # Volatility
            if 'Daily_Return' in data.columns:
                rolling_vol = data['Daily_Return'].rolling(window=30).std() * np.sqrt(252) * 100
                fig.add_trace(
                    go.Scatter(x=data['Date'], y=rolling_vol, name=f'{symbol} Volatility',
                              line=dict(color=color), mode='lines'),
                    row=2, col=2
                )
        
        # Update layout
        fig.update_layout(
            title_text="Portfolio Assets - Interactive Dashboard",
            title_x=0.5,
            height=800,
            showlegend=True
        )
        
        # Update y-axis labels
        fig.update_yaxes(title_text="Price ($)", row=1, col=1)
        fig.update_yaxes(title_text="Daily Return (%)", row=1, col=2)
        fig.update_yaxes(title_text="Volume", row=2, col=1)
        fig.update_yaxes(title_text="Volatility (%)", row=2, col=2)
        
        self.figures['interactive_dashboard'] = fig
        return fig
    
    def generate_summary_report(self) -> Dict[str, Dict]:
        """
        Generate a comprehensive summary report of the EDA findings.
        
        Returns:
            Dictionary containing summary statistics and insights
        """
        summary_report = {}
        
        for symbol, data in self.data.items():
            logger.info(f"Generating summary report for {symbol}...")
            
            if 'Close' not in data.columns or 'Daily_Return' not in data.columns:
                continue
            
            # Basic statistics
            close_price = data['Close']
            daily_returns = data['Daily_Return'].dropna()
            
            summary_report[symbol] = {
                'price_statistics': {
                    'current_price': close_price.iloc[-1],
                    'min_price': close_price.min(),
                    'max_price': close_price.max(),
                    'price_range': close_price.max() - close_price.min(),
                    'total_return': (close_price.iloc[-1] / close_price.iloc[0] - 1) * 100
                },
                'return_statistics': {
                    'mean_daily_return': daily_returns.mean() * 100,
                    'std_daily_return': daily_returns.std() * 100,
                    'annualized_return': daily_returns.mean() * 252 * 100,
                    'annualized_volatility': daily_returns.std() * np.sqrt(252) * 100,
                    'skewness': daily_returns.skew(),
                    'kurtosis': daily_returns.kurtosis(),
                    'min_daily_return': daily_returns.min() * 100,
                    'max_daily_return': daily_returns.max() * 100
                },
                'risk_metrics': {
                    'sharpe_ratio': (daily_returns.mean() / daily_returns.std()) * np.sqrt(252) if daily_returns.std() != 0 else 0,
                    'var_95': np.percentile(daily_returns, 5) * 100,
                    'var_99': np.percentile(daily_returns, 1) * 100,
                    'max_drawdown': self._calculate_max_drawdown(close_price)
                }
            }
        
        self.analysis_results['summary'] = summary_report
        return summary_report
    
    def _calculate_max_drawdown(self, price_series: pd.Series) -> float:
        """
        Calculate maximum drawdown for a price series.
        
        Args:
            price_series: Series of prices
            
        Returns:
            Maximum drawdown as percentage
        """
        peak = price_series.expanding().max()
        drawdown = (price_series - peak) / peak * 100
        return drawdown.min()
    
    def save_all_figures(self, base_path: str) -> None:
        """
        Save all generated figures to files.
        
        Args:
            base_path: Base path for saving figures
        """
        for name, fig in self.figures.items():
            if hasattr(fig, 'write_html'):  # Plotly figure
                fig.write_html(f"{base_path}_{name}.html")
                logger.info(f"Saved interactive {name} to {base_path}_{name}.html")
            else:  # Matplotlib figure
                fig.savefig(f"{base_path}_{name}.png", dpi=300, bbox_inches='tight')
                logger.info(f"Saved {name} to {base_path}_{name}.png")


def main():
    """
    Main function to demonstrate EDA functionality.
    """
    logger.info("EDA Analyzer module loaded successfully!")
    logger.info("Use this module in conjunction with data_fetcher.py and data_preprocessor.py for complete analysis.")


if __name__ == "__main__":
    main()