"""
Risk analysis module for portfolio time series analysis.
Calculates various risk metrics including VaR, Sharpe ratio, and other risk measures.
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union
import warnings
import logging

# Configure plotting
plt.style.use('seaborn-v0_8')
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RiskAnalyzer:
    """
    A class to perform comprehensive risk analysis on financial time series data.
    """
    
    def __init__(self, data: Dict[str, pd.DataFrame], risk_free_rate: float = 0.02):
        """
        Initialize the risk analyzer.
        
        Args:
            data: Dictionary of DataFrames with asset symbols as keys
            risk_free_rate: Annual risk-free rate (default 2%)
        """
        self.data = data
        self.risk_free_rate = risk_free_rate
        self.daily_risk_free_rate = risk_free_rate / 252  # Convert to daily
        self.risk_metrics = {}
        
    def calculate_var(self, confidence_levels: List[float] = [0.95, 0.99], 
                     method: str = 'historical') -> Dict[str, Dict]:
        """
        Calculate Value at Risk (VaR) for all assets.
        
        Args:
            confidence_levels: List of confidence levels (e.g., [0.95, 0.99])
            method: Method for VaR calculation ('historical', 'parametric', 'monte_carlo')
            
        Returns:
            Dictionary containing VaR values for each asset and confidence level
        """
        var_results = {}
        
        for symbol, data in self.data.items():
            if 'Daily_Return' not in data.columns:
                continue
                
            logger.info(f"Calculating VaR for {symbol} using {method} method...")
            
            returns = data['Daily_Return'].dropna()
            var_results[symbol] = {}
            
            for confidence in confidence_levels:
                if method == 'historical':
                    var_value = np.percentile(returns, (1 - confidence) * 100) * 100
                    
                elif method == 'parametric':
                    # Assume normal distribution
                    mean_return = returns.mean()
                    std_return = returns.std()
                    var_value = (mean_return + stats.norm.ppf(1 - confidence) * std_return) * 100
                    
                elif method == 'monte_carlo':
                    # Monte Carlo simulation
                    n_simulations = 10000
                    mean_return = returns.mean()
                    std_return = returns.std()
                    
                    simulated_returns = np.random.normal(mean_return, std_return, n_simulations)
                    var_value = np.percentile(simulated_returns, (1 - confidence) * 100) * 100
                
                var_results[symbol][f'VaR_{int(confidence*100)}%'] = var_value
                
                logger.info(f"{symbol} VaR at {confidence*100}% confidence: {var_value:.3f}%")
        
        self.risk_metrics['var'] = var_results
        return var_results
    
    def calculate_expected_shortfall(self, confidence_levels: List[float] = [0.95, 0.99]) -> Dict[str, Dict]:
        """
        Calculate Expected Shortfall (Conditional VaR) for all assets.
        
        Args:
            confidence_levels: List of confidence levels
            
        Returns:
            Dictionary containing ES values for each asset and confidence level
        """
        es_results = {}
        
        for symbol, data in self.data.items():
            if 'Daily_Return' not in data.columns:
                continue
                
            logger.info(f"Calculating Expected Shortfall for {symbol}...")
            
            returns = data['Daily_Return'].dropna()
            es_results[symbol] = {}
            
            for confidence in confidence_levels:
                # Calculate VaR threshold
                var_threshold = np.percentile(returns, (1 - confidence) * 100)
                
                # Calculate Expected Shortfall (mean of returns below VaR)
                tail_returns = returns[returns <= var_threshold]
                es_value = tail_returns.mean() * 100 if len(tail_returns) > 0 else 0
                
                es_results[symbol][f'ES_{int(confidence*100)}%'] = es_value
                
                logger.info(f"{symbol} Expected Shortfall at {confidence*100}% confidence: {es_value:.3f}%")
        
        self.risk_metrics['expected_shortfall'] = es_results
        return es_results
    
    def calculate_sharpe_ratio(self) -> Dict[str, float]:
        """
        Calculate Sharpe ratio for all assets.
        
        Returns:
            Dictionary containing Sharpe ratios for each asset
        """
        sharpe_ratios = {}
        
        for symbol, data in self.data.items():
            if 'Daily_Return' not in data.columns:
                continue
                
            logger.info(f"Calculating Sharpe ratio for {symbol}...")
            
            returns = data['Daily_Return'].dropna()
            
            # Calculate excess returns
            excess_returns = returns - self.daily_risk_free_rate
            
            # Calculate Sharpe ratio (annualized)
            if excess_returns.std() != 0:
                sharpe_ratio = (excess_returns.mean() / excess_returns.std()) * np.sqrt(252)
            else:
                sharpe_ratio = 0
            
            sharpe_ratios[symbol] = sharpe_ratio
            
            logger.info(f"{symbol} Sharpe Ratio: {sharpe_ratio:.4f}")
        
        self.risk_metrics['sharpe_ratio'] = sharpe_ratios
        return sharpe_ratios
    
    def calculate_sortino_ratio(self) -> Dict[str, float]:
        """
        Calculate Sortino ratio for all assets (focuses on downside risk).
        
        Returns:
            Dictionary containing Sortino ratios for each asset
        """
        sortino_ratios = {}
        
        for symbol, data in self.data.items():
            if 'Daily_Return' not in data.columns:
                continue
                
            logger.info(f"Calculating Sortino ratio for {symbol}...")
            
            returns = data['Daily_Return'].dropna()
            
            # Calculate excess returns
            excess_returns = returns - self.daily_risk_free_rate
            
            # Calculate downside deviation (only negative returns)
            downside_returns = excess_returns[excess_returns < 0]
            downside_deviation = downside_returns.std() if len(downside_returns) > 0 else 0
            
            # Calculate Sortino ratio (annualized)
            if downside_deviation != 0:
                sortino_ratio = (excess_returns.mean() / downside_deviation) * np.sqrt(252)
            else:
                sortino_ratio = 0
            
            sortino_ratios[symbol] = sortino_ratio
            
            logger.info(f"{symbol} Sortino Ratio: {sortino_ratio:.4f}")
        
        self.risk_metrics['sortino_ratio'] = sortino_ratios
        return sortino_ratios
    
    def calculate_maximum_drawdown(self) -> Dict[str, Dict]:
        """
        Calculate maximum drawdown and related metrics for all assets.
        
        Returns:
            Dictionary containing drawdown metrics for each asset
        """
        drawdown_results = {}
        
        for symbol, data in self.data.items():
            if 'Close' not in data.columns:
                continue
                
            logger.info(f"Calculating maximum drawdown for {symbol}...")
            
            prices = data['Close']
            
            # Calculate running maximum (peak)
            peak = prices.expanding().max()
            
            # Calculate drawdown
            drawdown = (prices - peak) / peak
            
            # Find maximum drawdown
            max_drawdown = drawdown.min()
            max_drawdown_date = data.loc[drawdown.idxmin(), 'Date'] if 'Date' in data.columns else None
            
            # Calculate drawdown duration
            drawdown_periods = (drawdown < 0).astype(int)
            drawdown_duration = self._calculate_max_duration(drawdown_periods)
            
            # Recovery time (time to reach new peak after max drawdown)
            max_dd_idx = drawdown.idxmin()
            recovery_idx = None
            for i in range(max_dd_idx + 1, len(prices)):
                if prices.iloc[i] >= peak.iloc[max_dd_idx]:
                    recovery_idx = i
                    break
            
            recovery_time = recovery_idx - max_dd_idx if recovery_idx else None
            
            drawdown_results[symbol] = {
                'max_drawdown': max_drawdown * 100,  # Convert to percentage
                'max_drawdown_date': max_drawdown_date,
                'max_drawdown_duration': drawdown_duration,
                'recovery_time': recovery_time,
                'current_drawdown': drawdown.iloc[-1] * 100
            }
            
            logger.info(f"{symbol} Maximum Drawdown: {max_drawdown * 100:.2f}%")
        
        self.risk_metrics['maximum_drawdown'] = drawdown_results
        return drawdown_results
    
    def _calculate_max_duration(self, binary_series: pd.Series) -> int:
        """
        Calculate maximum consecutive duration of 1s in a binary series.
        
        Args:
            binary_series: Series of 0s and 1s
            
        Returns:
            Maximum consecutive duration
        """
        if binary_series.sum() == 0:
            return 0
        
        # Find consecutive groups
        groups = binary_series.groupby((binary_series != binary_series.shift()).cumsum())
        consecutive_lengths = groups.sum()
        
        return consecutive_lengths.max() if len(consecutive_lengths) > 0 else 0
    
    def calculate_beta(self, market_returns: pd.Series) -> Dict[str, float]:
        """
        Calculate beta for all assets relative to market returns.
        
        Args:
            market_returns: Market return series (e.g., SPY returns)
            
        Returns:
            Dictionary containing beta values for each asset
        """
        betas = {}
        
        for symbol, data in self.data.items():
            if 'Daily_Return' not in data.columns:
                continue
                
            logger.info(f"Calculating beta for {symbol}...")
            
            returns = data['Daily_Return'].dropna()
            
            # Align the series
            aligned_data = pd.DataFrame({
                'asset': returns,
                'market': market_returns
            }).dropna()
            
            if len(aligned_data) < 30:  # Need sufficient data
                logger.warning(f"Insufficient data for beta calculation for {symbol}")
                betas[symbol] = np.nan
                continue
            
            # Calculate beta using covariance method
            covariance = aligned_data['asset'].cov(aligned_data['market'])
            market_variance = aligned_data['market'].var()
            
            beta = covariance / market_variance if market_variance != 0 else 0
            betas[symbol] = beta
            
            logger.info(f"{symbol} Beta: {beta:.4f}")
        
        self.risk_metrics['beta'] = betas
        return betas
    
    def calculate_tracking_error(self, benchmark_returns: pd.Series) -> Dict[str, float]:
        """
        Calculate tracking error for all assets relative to a benchmark.
        
        Args:
            benchmark_returns: Benchmark return series
            
        Returns:
            Dictionary containing tracking errors for each asset
        """
        tracking_errors = {}
        
        for symbol, data in self.data.items():
            if 'Daily_Return' not in data.columns:
                continue
                
            logger.info(f"Calculating tracking error for {symbol}...")
            
            returns = data['Daily_Return'].dropna()
            
            # Align the series
            aligned_data = pd.DataFrame({
                'asset': returns,
                'benchmark': benchmark_returns
            }).dropna()
            
            if len(aligned_data) < 30:
                logger.warning(f"Insufficient data for tracking error calculation for {symbol}")
                tracking_errors[symbol] = np.nan
                continue
            
            # Calculate tracking error (annualized standard deviation of excess returns)
            excess_returns = aligned_data['asset'] - aligned_data['benchmark']
            tracking_error = excess_returns.std() * np.sqrt(252) * 100  # Annualized percentage
            
            tracking_errors[symbol] = tracking_error
            
            logger.info(f"{symbol} Tracking Error: {tracking_error:.2f}%")
        
        self.risk_metrics['tracking_error'] = tracking_errors
        return tracking_errors
    
    def plot_var_analysis_simple(self, figsize: Tuple[int, int] = (15, 8), save_path: Optional[str] = None) -> None:
        """
        Plot simplified VaR analysis visualization (without time series breaches).
        
        Args:
            figsize: Figure size for matplotlib plots
            save_path: Path to save the plot (optional)
        """
        if 'var' not in self.risk_metrics:
            logger.warning("VaR analysis not performed yet. Running VaR calculation...")
            self.calculate_var()
        
        fig, axes = plt.subplots(1, len(self.data), figsize=figsize)
        if len(self.data) == 1:
            axes = [axes]
        fig.suptitle('Value at Risk (VaR) Analysis - Returns Distribution', fontsize=16, fontweight='bold')
        
        for i, (symbol, data) in enumerate(self.data.items()):
            if 'Daily_Return' not in data.columns:
                continue
            
            returns = data['Daily_Return'].dropna() * 100  # Convert to percentage
            
            # Returns distribution with VaR markers
            axes[i].hist(returns, bins=50, alpha=0.7, density=True, color='lightblue', edgecolor='black')
            
            # Add VaR lines
            var_95 = self.risk_metrics['var'][symbol]['VaR_95%']
            var_99 = self.risk_metrics['var'][symbol]['VaR_99%']
            
            axes[i].axvline(var_95, color='red', linestyle='--', linewidth=2, label=f'VaR 95%: {var_95:.2f}%')
            axes[i].axvline(var_99, color='darkred', linestyle='--', linewidth=2, label=f'VaR 99%: {var_99:.2f}%')
            axes[i].axvline(returns.mean(), color='green', linestyle='-', linewidth=2, label=f'Mean: {returns.mean():.2f}%')
            
            axes[i].set_title(f'{symbol} - Returns Distribution & VaR', fontweight='bold')
            axes[i].set_xlabel('Daily Return (%)')
            axes[i].set_ylabel('Density')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Simplified VaR analysis plot saved to {save_path}")
        
        plt.show()
    
    def plot_var_analysis(self, figsize: Tuple[int, int] = (15, 10), save_path: Optional[str] = None) -> None:
        """
        Plot VaR analysis visualization.
        
        Args:
            figsize: Figure size for matplotlib plots
            save_path: Path to save the plot (optional)
        """
        if 'var' not in self.risk_metrics:
            logger.warning("VaR analysis not performed yet. Running VaR calculation...")
            self.calculate_var()
        
        fig, axes = plt.subplots(2, len(self.data), figsize=figsize)
        fig.suptitle('Value at Risk (VaR) Analysis', fontsize=16, fontweight='bold')
        
        for i, (symbol, data) in enumerate(self.data.items()):
            if 'Daily_Return' not in data.columns:
                continue
            
            returns = data['Daily_Return'].dropna() * 100  # Convert to percentage
            
            # Returns distribution with VaR markers
            axes[0, i].hist(returns, bins=50, alpha=0.7, density=True, color='lightblue', edgecolor='black')
            
            # Add VaR lines
            var_95 = self.risk_metrics['var'][symbol]['VaR_95%']
            var_99 = self.risk_metrics['var'][symbol]['VaR_99%']
            
            axes[0, i].axvline(var_95, color='red', linestyle='--', linewidth=2, label=f'VaR 95%: {var_95:.2f}%')
            axes[0, i].axvline(var_99, color='darkred', linestyle='--', linewidth=2, label=f'VaR 99%: {var_99:.2f}%')
            axes[0, i].axvline(returns.mean(), color='green', linestyle='-', linewidth=2, label=f'Mean: {returns.mean():.2f}%')
            
            axes[0, i].set_title(f'{symbol} - Returns Distribution & VaR')
            axes[0, i].set_xlabel('Daily Return (%)')
            axes[0, i].set_ylabel('Density')
            axes[0, i].legend()
            axes[0, i].grid(True, alpha=0.3)
            
            # Time series with VaR breaches (align data lengths)
            if len(data['Date']) == len(returns):
                axes[1, i].plot(data['Date'], returns, alpha=0.7, linewidth=1, color='blue')
            else:
                # Handle mismatched lengths
                min_len = min(len(data['Date']), len(returns))
                axes[1, i].plot(data['Date'][:min_len], returns[:min_len], alpha=0.7, linewidth=1, color='blue')
            axes[1, i].axhline(var_95, color='red', linestyle='--', alpha=0.7, label='VaR 95%')
            axes[1, i].axhline(var_99, color='darkred', linestyle='--', alpha=0.7, label='VaR 99%')
            
            # Highlight VaR breaches (handle data alignment)
            try:
                # Ensure we have matching indices for data and returns
                min_len = min(len(data), len(returns))
                data_aligned = data.iloc[:min_len].copy()
                returns_aligned = returns.iloc[:min_len] if hasattr(returns, 'iloc') else returns[:min_len]
                
                var_breaches_95 = returns_aligned < var_95
                var_breaches_99 = returns_aligned < var_99
                
                if var_breaches_95.any():
                    breach_indices = var_breaches_95[var_breaches_95].index if hasattr(var_breaches_95, 'index') else np.where(var_breaches_95)[0]
                    if hasattr(breach_indices, '__len__') and len(breach_indices) > 0:
                        breach_dates = data_aligned.iloc[breach_indices]['Date'] if hasattr(breach_indices, '__iter__') else data_aligned.loc[breach_indices, 'Date']
                        breach_values = returns_aligned[var_breaches_95]
                        axes[1, i].scatter(breach_dates, breach_values, color='red', s=20, alpha=0.8, label='95% Breaches')
                
                if var_breaches_99.any():
                    breach_indices = var_breaches_99[var_breaches_99].index if hasattr(var_breaches_99, 'index') else np.where(var_breaches_99)[0]
                    if hasattr(breach_indices, '__len__') and len(breach_indices) > 0:
                        breach_dates = data_aligned.iloc[breach_indices]['Date'] if hasattr(breach_indices, '__iter__') else data_aligned.loc[breach_indices, 'Date']
                        breach_values = returns_aligned[var_breaches_99]
                        axes[1, i].scatter(breach_dates, breach_values, color='darkred', s=30, alpha=0.8, label='99% Breaches')
                        
            except Exception as e:
                logger.warning(f"Could not plot VaR breaches for {symbol}: {str(e)}")
                # Continue without breach markers
            
            axes[1, i].set_title(f'{symbol} - Returns Time Series & VaR Breaches')
            axes[1, i].set_xlabel('Date')
            axes[1, i].set_ylabel('Daily Return (%)')
            axes[1, i].legend()
            axes[1, i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"VaR analysis plot saved to {save_path}")
        
        plt.show()
    
    def plot_risk_metrics_comparison(self, figsize: Tuple[int, int] = (15, 8), save_path: Optional[str] = None) -> None:
        """
        Plot comparison of risk metrics across assets.
        
        Args:
            figsize: Figure size for matplotlib plots
            save_path: Path to save the plot (optional)
        """
        # Ensure all metrics are calculated
        if 'sharpe_ratio' not in self.risk_metrics:
            self.calculate_sharpe_ratio()
        if 'sortino_ratio' not in self.risk_metrics:
            self.calculate_sortino_ratio()
        if 'maximum_drawdown' not in self.risk_metrics:
            self.calculate_maximum_drawdown()
        if 'var' not in self.risk_metrics:
            self.calculate_var()
        
        # Prepare data for plotting
        symbols = list(self.data.keys())
        sharpe_ratios = [self.risk_metrics['sharpe_ratio'].get(symbol, 0) for symbol in symbols]
        sortino_ratios = [self.risk_metrics['sortino_ratio'].get(symbol, 0) for symbol in symbols]
        max_drawdowns = [abs(self.risk_metrics['maximum_drawdown'].get(symbol, {}).get('max_drawdown', 0)) for symbol in symbols]
        var_95 = [abs(self.risk_metrics['var'].get(symbol, {}).get('VaR_95%', 0)) for symbol in symbols]
        
        # Create comparison plots
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('Risk Metrics Comparison Across Assets', fontsize=16, fontweight='bold')
        
        # Sharpe Ratio
        bars1 = axes[0, 0].bar(symbols, sharpe_ratios, alpha=0.7, color='green')
        axes[0, 0].set_title('Sharpe Ratio Comparison')
        axes[0, 0].set_ylabel('Sharpe Ratio')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].axhline(y=0, color='red', linestyle='--', alpha=0.7)
        
        # Add value labels on bars
        for bar, value in zip(bars1, sharpe_ratios):
            axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{value:.2f}', ha='center', va='bottom')
        
        # Sortino Ratio
        bars2 = axes[0, 1].bar(symbols, sortino_ratios, alpha=0.7, color='blue')
        axes[0, 1].set_title('Sortino Ratio Comparison')
        axes[0, 1].set_ylabel('Sortino Ratio')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].axhline(y=0, color='red', linestyle='--', alpha=0.7)
        
        for bar, value in zip(bars2, sortino_ratios):
            axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{value:.2f}', ha='center', va='bottom')
        
        # Maximum Drawdown
        bars3 = axes[1, 0].bar(symbols, max_drawdowns, alpha=0.7, color='red')
        axes[1, 0].set_title('Maximum Drawdown Comparison')
        axes[1, 0].set_ylabel('Maximum Drawdown (%)')
        axes[1, 0].grid(True, alpha=0.3)
        
        for bar, value in zip(bars3, max_drawdowns):
            axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                           f'{value:.1f}%', ha='center', va='bottom')
        
        # VaR 95%
        bars4 = axes[1, 1].bar(symbols, var_95, alpha=0.7, color='orange')
        axes[1, 1].set_title('Value at Risk (95%) Comparison')
        axes[1, 1].set_ylabel('VaR 95% (%)')
        axes[1, 1].grid(True, alpha=0.3)
        
        for bar, value in zip(bars4, var_95):
            axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                           f'{value:.2f}%', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Risk metrics comparison plot saved to {save_path}")
        
        plt.show()
    
    def generate_risk_report(self) -> Dict[str, Dict]:
        """
        Generate a comprehensive risk analysis report.
        
        Returns:
            Dictionary containing all risk metrics and analysis
        """
        logger.info("Generating comprehensive risk analysis report...")
        
        # Calculate all risk metrics
        self.calculate_var()
        self.calculate_expected_shortfall()
        self.calculate_sharpe_ratio()
        self.calculate_sortino_ratio()
        self.calculate_maximum_drawdown()
        
        # Compile risk report
        risk_report = {}
        
        for symbol in self.data.keys():
            risk_report[symbol] = {
                'var_metrics': self.risk_metrics.get('var', {}).get(symbol, {}),
                'expected_shortfall': self.risk_metrics.get('expected_shortfall', {}).get(symbol, {}),
                'sharpe_ratio': self.risk_metrics.get('sharpe_ratio', {}).get(symbol, 0),
                'sortino_ratio': self.risk_metrics.get('sortino_ratio', {}).get(symbol, 0),
                'maximum_drawdown': self.risk_metrics.get('maximum_drawdown', {}).get(symbol, {}),
                'beta': self.risk_metrics.get('beta', {}).get(symbol, np.nan),
                'tracking_error': self.risk_metrics.get('tracking_error', {}).get(symbol, np.nan)
            }
        
        return risk_report
    
    def export_risk_metrics(self, filepath: str) -> None:
        """
        Export risk metrics to CSV file.
        
        Args:
            filepath: Path to save the CSV file
        """
        # Generate comprehensive report
        risk_report = self.generate_risk_report()
        
        # Convert to DataFrame for easy export
        export_data = []
        
        for symbol, metrics in risk_report.items():
            row = {'Symbol': symbol}
            
            # Flatten the nested dictionary
            for category, values in metrics.items():
                if isinstance(values, dict):
                    for key, value in values.items():
                        row[f"{category}_{key}"] = value
                else:
                    row[category] = values
            
            export_data.append(row)
        
        # Create DataFrame and save
        df = pd.DataFrame(export_data)
        df.to_csv(filepath, index=False)
        
        logger.info(f"Risk metrics exported to {filepath}")


def main():
    """
    Main function to demonstrate risk analysis functionality.
    """
    logger.info("Risk Analyzer module loaded successfully!")
    logger.info("Use this module in conjunction with other modules for comprehensive risk analysis.")


if __name__ == "__main__":
    main()