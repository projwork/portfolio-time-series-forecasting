"""
Portfolio Optimizer - Modern Portfolio Theory implementation for optimal portfolio construction.
Implements efficient frontier calculation, portfolio optimization, and performance analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union, Any
import warnings
import logging
from datetime import datetime
from scipy.optimize import minimize
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')


class PortfolioOptimizer:
    """
    Modern Portfolio Theory implementation for portfolio optimization.
    """
    
    def __init__(self, risk_free_rate: float = 0.02):
        """
        Initialize the portfolio optimizer.
        
        Args:
            risk_free_rate: Annual risk-free rate (default 2%)
        """
        self.risk_free_rate = risk_free_rate
        self.assets = []
        self.expected_returns = None
        self.covariance_matrix = None
        self.efficient_frontier = None
        self.optimal_portfolios = {}
        self.historical_data = {}
        
    def set_assets(self, asset_names: List[str]) -> None:
        """
        Set the asset names for optimization.
        
        Args:
            asset_names: List of asset symbols
        """
        self.assets = asset_names
        logger.info(f"Portfolio assets set: {', '.join(asset_names)}")
    
    def add_historical_data(self, asset_name: str, returns: pd.Series) -> None:
        """
        Add historical return data for an asset.
        
        Args:
            asset_name: Asset symbol
            returns: Historical returns series
        """
        self.historical_data[asset_name] = returns
        logger.info(f"Added historical data for {asset_name}: {len(returns)} observations")
    
    def calculate_expected_returns(self, forecast_data: Optional[Dict[str, float]] = None) -> pd.Series:
        """
        Calculate expected returns for all assets.
        
        Args:
            forecast_data: Dictionary with asset forecasted returns
            
        Returns:
            Series of expected annual returns
        """
        expected_returns = {}
        
        for asset in self.assets:
            if forecast_data and asset in forecast_data:
                # Use forecasted return
                expected_returns[asset] = forecast_data[asset]
                logger.info(f"Using forecast return for {asset}: {forecast_data[asset]:.3f}")
            else:
                # Use historical average return (annualized)
                if asset in self.historical_data:
                    daily_returns = self.historical_data[asset]
                    annual_return = daily_returns.mean() * 252  # 252 trading days
                    expected_returns[asset] = annual_return
                    logger.info(f"Using historical return for {asset}: {annual_return:.3f}")
                else:
                    logger.warning(f"No data available for {asset}, using default return of 0.05")
                    expected_returns[asset] = 0.05
        
        self.expected_returns = pd.Series(expected_returns)
        return self.expected_returns
    
    def calculate_covariance_matrix(self) -> pd.DataFrame:
        """
        Calculate the covariance matrix from historical data.
        
        Returns:
            Covariance matrix (annualized)
        """
        if not self.historical_data:
            raise ValueError("No historical data available for covariance calculation")
        
        # Combine all returns into a DataFrame
        returns_df = pd.DataFrame()
        for asset in self.assets:
            if asset in self.historical_data:
                returns_df[asset] = self.historical_data[asset]
            else:
                logger.warning(f"No historical data for {asset}, using synthetic data")
                # Generate synthetic returns with correlation
                synthetic_returns = np.random.normal(0.0005, 0.02, len(list(self.historical_data.values())[0]))
                returns_df[asset] = synthetic_returns
        
        # Calculate annualized covariance matrix
        daily_cov = returns_df.cov()
        annual_cov = daily_cov * 252  # Annualize
        
        self.covariance_matrix = annual_cov
        logger.info(f"Calculated covariance matrix for {len(self.assets)} assets")
        
        return self.covariance_matrix
    
    def portfolio_performance(self, weights: np.ndarray) -> Tuple[float, float, float]:
        """
        Calculate portfolio performance metrics.
        
        Args:
            weights: Portfolio weights
            
        Returns:
            Tuple of (expected_return, volatility, sharpe_ratio)
        """
        if self.expected_returns is None or self.covariance_matrix is None:
            raise ValueError("Expected returns and covariance matrix must be calculated first")
        
        # Portfolio expected return
        portfolio_return = np.sum(weights * self.expected_returns.values)
        
        # Portfolio volatility
        portfolio_variance = np.dot(weights.T, np.dot(self.covariance_matrix.values, weights))
        portfolio_volatility = np.sqrt(portfolio_variance)
        
        # Sharpe ratio
        sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_volatility
        
        return portfolio_return, portfolio_volatility, sharpe_ratio
    
    def negative_sharpe_ratio(self, weights: np.ndarray) -> float:
        """
        Calculate negative Sharpe ratio for optimization (minimization).
        
        Args:
            weights: Portfolio weights
            
        Returns:
            Negative Sharpe ratio
        """
        _, _, sharpe = self.portfolio_performance(weights)
        return -sharpe
    
    def portfolio_volatility(self, weights: np.ndarray) -> float:
        """
        Calculate portfolio volatility for optimization.
        
        Args:
            weights: Portfolio weights
            
        Returns:
            Portfolio volatility
        """
        _, volatility, _ = self.portfolio_performance(weights)
        return volatility
    
    def optimize_portfolio(self, objective: str = 'max_sharpe') -> Dict[str, Any]:
        """
        Optimize portfolio based on specified objective.
        
        Args:
            objective: Optimization objective ('max_sharpe' or 'min_volatility')
            
        Returns:
            Dictionary with optimization results
        """
        if self.expected_returns is None or self.covariance_matrix is None:
            raise ValueError("Expected returns and covariance matrix must be calculated first")
        
        n_assets = len(self.assets)
        
        # Constraints and bounds
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})  # Weights sum to 1
        bounds = tuple((0, 1) for _ in range(n_assets))  # No short selling
        
        # Initial guess (equal weights)
        initial_weights = np.array([1/n_assets] * n_assets)
        
        # Optimization
        if objective == 'max_sharpe':
            result = minimize(self.negative_sharpe_ratio, initial_weights,
                            method='SLSQP', bounds=bounds, constraints=constraints)
        elif objective == 'min_volatility':
            result = minimize(self.portfolio_volatility, initial_weights,
                            method='SLSQP', bounds=bounds, constraints=constraints)
        else:
            raise ValueError("Objective must be 'max_sharpe' or 'min_volatility'")
        
        if not result.success:
            logger.warning(f"Optimization did not converge: {result.message}")
        
        # Calculate performance metrics
        optimal_weights = result.x
        portfolio_return, portfolio_volatility, sharpe_ratio = self.portfolio_performance(optimal_weights)
        
        optimization_result = {
            'weights': dict(zip(self.assets, optimal_weights)),
            'expected_return': portfolio_return,
            'volatility': portfolio_volatility,
            'sharpe_ratio': sharpe_ratio,
            'optimization_success': result.success,
            'objective': objective
        }
        
        self.optimal_portfolios[objective] = optimization_result
        logger.info(f"Optimized portfolio for {objective}: Sharpe={sharpe_ratio:.3f}, Return={portfolio_return:.3f}, Vol={portfolio_volatility:.3f}")
        
        return optimization_result
    
    def generate_efficient_frontier(self, n_portfolios: int = 50) -> pd.DataFrame:
        """
        Generate the efficient frontier.
        
        Args:
            n_portfolios: Number of portfolios to generate
            
        Returns:
            DataFrame with efficient frontier portfolios
        """
        if self.expected_returns is None or self.covariance_matrix is None:
            raise ValueError("Expected returns and covariance matrix must be calculated first")
        
        n_assets = len(self.assets)
        
        # Target returns for efficient frontier
        min_return = self.expected_returns.min()
        max_return = self.expected_returns.max()
        target_returns = np.linspace(min_return, max_return, n_portfolios)
        
        efficient_portfolios = []
        
        # Constraints and bounds
        bounds = tuple((0, 1) for _ in range(n_assets))  # No short selling
        initial_weights = np.array([1/n_assets] * n_assets)
        
        for target_return in target_returns:
            # Constraints: weights sum to 1 and achieve target return
            constraints = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
                {'type': 'eq', 'fun': lambda x: np.sum(x * self.expected_returns.values) - target_return}
            ]
            
            # Minimize volatility for target return
            result = minimize(self.portfolio_volatility, initial_weights,
                            method='SLSQP', bounds=bounds, constraints=constraints)
            
            if result.success:
                weights = result.x
                portfolio_return, portfolio_volatility, sharpe_ratio = self.portfolio_performance(weights)
                
                portfolio_data = {
                    'target_return': target_return,
                    'return': portfolio_return,
                    'volatility': portfolio_volatility,
                    'sharpe_ratio': sharpe_ratio
                }
                
                # Add individual weights
                for i, asset in enumerate(self.assets):
                    portfolio_data[f'weight_{asset}'] = weights[i]
                
                efficient_portfolios.append(portfolio_data)
        
        self.efficient_frontier = pd.DataFrame(efficient_portfolios)
        logger.info(f"Generated efficient frontier with {len(self.efficient_frontier)} portfolios")
        
        return self.efficient_frontier
    
    def plot_efficient_frontier(self, figsize: Tuple[int, int] = (12, 8)) -> None:
        """
        Plot the efficient frontier with key portfolios.
        
        Args:
            figsize: Figure size
        """
        if self.efficient_frontier is None:
            logger.warning("Efficient frontier not generated yet")
            return
        
        plt.figure(figsize=figsize)
        
        # Plot efficient frontier
        plt.plot(self.efficient_frontier['volatility'], self.efficient_frontier['return'],
                'b-', linewidth=3, label='Efficient Frontier', alpha=0.8)
        
        # Plot individual assets
        for asset in self.assets:
            if asset in self.historical_data:
                asset_return = self.expected_returns[asset]
                asset_vol = np.sqrt(self.covariance_matrix.loc[asset, asset])
                plt.scatter(asset_vol, asset_return, s=100, alpha=0.8, label=f'{asset}')
        
        # Plot optimal portfolios
        if 'max_sharpe' in self.optimal_portfolios:
            max_sharpe = self.optimal_portfolios['max_sharpe']
            plt.scatter(max_sharpe['volatility'], max_sharpe['expected_return'],
                       s=200, c='red', marker='*', label='Max Sharpe Ratio', zorder=5)
        
        if 'min_volatility' in self.optimal_portfolios:
            min_vol = self.optimal_portfolios['min_volatility']
            plt.scatter(min_vol['volatility'], min_vol['expected_return'],
                       s=200, c='green', marker='D', label='Min Volatility', zorder=5)
        
        plt.xlabel('Volatility (Risk)', fontsize=12)
        plt.ylabel('Expected Return', fontsize=12)
        plt.title('Efficient Frontier and Optimal Portfolios', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def create_interactive_frontier_plot(self) -> go.Figure:
        """
        Create interactive Plotly efficient frontier plot.
        
        Returns:
            Plotly figure object
        """
        if self.efficient_frontier is None:
            return go.Figure()
        
        fig = go.Figure()
        
        # Add efficient frontier
        fig.add_trace(go.Scatter(
            x=self.efficient_frontier['volatility'],
            y=self.efficient_frontier['return'],
            mode='lines',
            name='Efficient Frontier',
            line=dict(color='blue', width=3),
            hovertemplate='Risk: %{x:.3f}<br>Return: %{y:.3f}<br>Sharpe: %{customdata:.3f}<extra></extra>',
            customdata=self.efficient_frontier['sharpe_ratio']
        ))
        
        # Add individual assets
        for asset in self.assets:
            if asset in self.historical_data:
                asset_return = self.expected_returns[asset]
                asset_vol = np.sqrt(self.covariance_matrix.loc[asset, asset])
                fig.add_trace(go.Scatter(
                    x=[asset_vol],
                    y=[asset_return],
                    mode='markers',
                    name=asset,
                    marker=dict(size=12),
                    hovertemplate=f'{asset}<br>Risk: %{{x:.3f}}<br>Return: %{{y:.3f}}<extra></extra>'
                ))
        
        # Add optimal portfolios
        if 'max_sharpe' in self.optimal_portfolios:
            max_sharpe = self.optimal_portfolios['max_sharpe']
            fig.add_trace(go.Scatter(
                x=[max_sharpe['volatility']],
                y=[max_sharpe['expected_return']],
                mode='markers',
                name='Max Sharpe Ratio',
                marker=dict(size=15, symbol='star', color='red'),
                hovertemplate='Max Sharpe Portfolio<br>Risk: %{x:.3f}<br>Return: %{y:.3f}<br>Sharpe: ' + f"{max_sharpe['sharpe_ratio']:.3f}<extra></extra>"
            ))
        
        if 'min_volatility' in self.optimal_portfolios:
            min_vol = self.optimal_portfolios['min_volatility']
            fig.add_trace(go.Scatter(
                x=[min_vol['volatility']],
                y=[min_vol['expected_return']],
                mode='markers',
                name='Min Volatility',
                marker=dict(size=15, symbol='diamond', color='green'),
                hovertemplate='Min Volatility Portfolio<br>Risk: %{x:.3f}<br>Return: %{y:.3f}<br>Sharpe: ' + f"{min_vol['sharpe_ratio']:.3f}<extra></extra>"
            ))
        
        fig.update_layout(
            title='Interactive Efficient Frontier',
            xaxis_title='Volatility (Risk)',
            yaxis_title='Expected Return',
            hovermode='closest',
            template='plotly_white'
        )
        
        return fig
    
    def analyze_portfolio_composition(self, portfolio_type: str = 'max_sharpe') -> Dict[str, Any]:
        """
        Analyze portfolio composition and characteristics.
        
        Args:
            portfolio_type: Type of portfolio to analyze
            
        Returns:
            Dictionary with portfolio analysis
        """
        if portfolio_type not in self.optimal_portfolios:
            raise ValueError(f"Portfolio type '{portfolio_type}' not found")
        
        portfolio = self.optimal_portfolios[portfolio_type]
        weights = portfolio['weights']
        
        # Portfolio concentration analysis
        max_weight = max(weights.values())
        min_weight = min(weights.values())
        weight_std = np.std(list(weights.values()))
        
        # Effective number of assets (diversification measure)
        herfindahl_index = sum(w**2 for w in weights.values())
        effective_assets = 1 / herfindahl_index if herfindahl_index > 0 else 0
        
        # Risk contribution analysis
        portfolio_weights = np.array([weights[asset] for asset in self.assets])
        cov_matrix = self.covariance_matrix.values
        
        # Marginal risk contribution
        marginal_contrib = np.dot(cov_matrix, portfolio_weights)
        risk_contrib = portfolio_weights * marginal_contrib
        total_risk = portfolio['volatility']**2
        risk_contrib_pct = risk_contrib / total_risk
        
        analysis = {
            'portfolio_type': portfolio_type,
            'performance': {
                'expected_return': portfolio['expected_return'],
                'volatility': portfolio['volatility'],
                'sharpe_ratio': portfolio['sharpe_ratio']
            },
            'composition': {
                'weights': weights,
                'max_weight': max_weight,
                'min_weight': min_weight,
                'weight_concentration': max_weight - min_weight,
                'weight_std': weight_std
            },
            'diversification': {
                'effective_assets': effective_assets,
                'herfindahl_index': herfindahl_index,
                'diversification_ratio': effective_assets / len(self.assets)
            },
            'risk_contribution': {
                'marginal_contrib': dict(zip(self.assets, marginal_contrib)),
                'risk_contrib_pct': dict(zip(self.assets, risk_contrib_pct)),
                'largest_risk_contributor': self.assets[np.argmax(risk_contrib_pct)]
            }
        }
        
        return analysis
    
    def compare_portfolios(self) -> pd.DataFrame:
        """
        Compare all optimized portfolios.
        
        Returns:
            DataFrame with portfolio comparison
        """
        if not self.optimal_portfolios:
            logger.warning("No optimized portfolios available for comparison")
            return pd.DataFrame()
        
        comparison_data = []
        
        for portfolio_type, portfolio in self.optimal_portfolios.items():
            data = {
                'Portfolio': portfolio_type.replace('_', ' ').title(),
                'Expected Return': portfolio['expected_return'],
                'Volatility': portfolio['volatility'],
                'Sharpe Ratio': portfolio['sharpe_ratio']
            }
            
            # Add weights
            for asset in self.assets:
                data[f'{asset} Weight'] = portfolio['weights'].get(asset, 0)
            
            comparison_data.append(data)
        
        comparison_df = pd.DataFrame(comparison_data)
        return comparison_df
    
    def plot_portfolio_composition(self, portfolio_type: str = 'max_sharpe', 
                                  figsize: Tuple[int, int] = (10, 6)) -> None:
        """
        Plot portfolio composition as pie chart and bar chart.
        
        Args:
            portfolio_type: Type of portfolio to plot
            figsize: Figure size
        """
        if portfolio_type not in self.optimal_portfolios:
            logger.warning(f"Portfolio type '{portfolio_type}' not found")
            return
        
        weights = self.optimal_portfolios[portfolio_type]['weights']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Pie chart
        assets = list(weights.keys())
        weight_values = list(weights.values())
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(assets)))
        wedges, texts, autotexts = ax1.pie(weight_values, labels=assets, autopct='%1.1f%%',
                                          colors=colors, startangle=90)
        ax1.set_title(f'{portfolio_type.replace("_", " ").title()} Portfolio Composition')
        
        # Bar chart
        bars = ax2.bar(assets, weight_values, color=colors, alpha=0.8)
        ax2.set_title('Portfolio Weights')
        ax2.set_ylabel('Weight')
        ax2.set_xlabel('Assets')
        
        # Add value labels on bars
        for bar, weight in zip(bars, weight_values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                    f'{weight:.1%}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()
    
    def generate_portfolio_report(self, portfolio_type: str = 'max_sharpe') -> str:
        """
        Generate comprehensive portfolio analysis report.
        
        Args:
            portfolio_type: Type of portfolio to analyze
            
        Returns:
            Formatted report string
        """
        if portfolio_type not in self.optimal_portfolios:
            return f"Portfolio type '{portfolio_type}' not found"
        
        analysis = self.analyze_portfolio_composition(portfolio_type)
        
        report = []
        report.append("=" * 70)
        report.append(f"PORTFOLIO OPTIMIZATION REPORT")
        report.append("=" * 70)
        report.append(f"Portfolio Type: {portfolio_type.replace('_', ' ').title()}")
        report.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Performance Summary
        perf = analysis['performance']
        report.append("\n📊 PERFORMANCE SUMMARY")
        report.append("-" * 40)
        report.append(f"Expected Annual Return: {perf['expected_return']:.2%}")
        report.append(f"Annual Volatility: {perf['volatility']:.2%}")
        report.append(f"Sharpe Ratio: {perf['sharpe_ratio']:.3f}")
        report.append(f"Risk-Free Rate: {self.risk_free_rate:.2%}")
        
        # Portfolio Composition
        comp = analysis['composition']
        report.append("\n🎯 PORTFOLIO COMPOSITION")
        report.append("-" * 40)
        for asset, weight in comp['weights'].items():
            report.append(f"{asset}: {weight:.1%}")
        
        report.append(f"\nConcentration Analysis:")
        report.append(f"  Largest Position: {comp['max_weight']:.1%}")
        report.append(f"  Smallest Position: {comp['min_weight']:.1%}")
        report.append(f"  Weight Concentration: {comp['weight_concentration']:.1%}")
        
        # Diversification Analysis
        div = analysis['diversification']
        report.append("\n📈 DIVERSIFICATION ANALYSIS")
        report.append("-" * 40)
        report.append(f"Effective Number of Assets: {div['effective_assets']:.2f}")
        report.append(f"Herfindahl Index: {div['herfindahl_index']:.3f}")
        report.append(f"Diversification Ratio: {div['diversification_ratio']:.1%}")
        
        # Risk Analysis
        risk = analysis['risk_contribution']
        report.append("\n⚠️ RISK CONTRIBUTION ANALYSIS")
        report.append("-" * 40)
        report.append(f"Largest Risk Contributor: {risk['largest_risk_contributor']}")
        report.append("Risk Contribution by Asset:")
        for asset, contrib in risk['risk_contrib_pct'].items():
            report.append(f"  {asset}: {contrib:.1%}")
        
        # Investment Insights
        report.append("\n💡 INVESTMENT INSIGHTS")
        report.append("-" * 40)
        
        # Generate insights based on analysis
        if perf['sharpe_ratio'] > 1.0:
            report.append("• Excellent risk-adjusted returns (Sharpe > 1.0)")
        elif perf['sharpe_ratio'] > 0.5:
            report.append("• Good risk-adjusted returns (Sharpe > 0.5)")
        else:
            report.append("• Moderate risk-adjusted returns")
        
        if div['effective_assets'] > 2.5:
            report.append("• Well-diversified portfolio")
        elif div['effective_assets'] > 1.5:
            report.append("• Moderately diversified portfolio")
        else:
            report.append("• Concentrated portfolio - consider diversification")
        
        if comp['max_weight'] > 0.6:
            report.append("• High concentration in single asset - monitor closely")
        elif comp['max_weight'] > 0.4:
            report.append("• Moderate concentration - acceptable for focused strategy")
        
        report.append("\n" + "=" * 70)
        
        return "\n".join(report)


def main():
    """
    Main function to demonstrate portfolio optimizer functionality.
    """
    logger.info("Portfolio Optimizer module loaded successfully!")
    logger.info("Available features:")
    logger.info("  - Modern Portfolio Theory implementation")
    logger.info("  - Efficient frontier generation")
    logger.info("  - Portfolio optimization (Max Sharpe, Min Volatility)")
    logger.info("  - Risk and diversification analysis")
    logger.info("  - Interactive visualization")
    logger.info("  - Comprehensive reporting")


if __name__ == "__main__":
    main()
