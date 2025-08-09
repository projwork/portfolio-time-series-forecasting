"""
Data fetching module for portfolio time series analysis.
Fetches historical financial data for TSLA, BND, and SPY using YFinance.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, date
from typing import List, Dict, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PortfolioDataFetcher:
    """
    A class to fetch and manage historical financial data for portfolio analysis.
    """
    
    def __init__(self, start_date: str = "2015-07-01", end_date: str = "2025-07-31"):
        """
        Initialize the data fetcher with date range.
        
        Args:
            start_date (str): Start date in YYYY-MM-DD format
            end_date (str): End date in YYYY-MM-DD format
        """
        self.start_date = start_date
        self.end_date = end_date
        self.assets = {
            'TSLA': 'Tesla Inc. - High-growth, high-risk stock in consumer discretionary sector',
            'BND': 'Vanguard Total Bond Market ETF - Stability and income from US investment-grade bonds',
            'SPY': 'S&P 500 ETF - Broad US market exposure with moderate risk'
        }
        self.data = {}
        
    def fetch_single_asset(self, symbol: str) -> Optional[pd.DataFrame]:
        """
        Fetch historical data for a single asset.
        
        Args:
            symbol (str): Stock/ETF symbol
            
        Returns:
            pd.DataFrame: Historical price data or None if failed
        """
        try:
            logger.info(f"Fetching data for {symbol}...")
            ticker = yf.Ticker(symbol)
            data = ticker.history(start=self.start_date, end=self.end_date)
            
            if data.empty:
                logger.warning(f"No data found for {symbol}")
                return None
                
            # Reset index to make Date a column
            data.reset_index(inplace=True)
            
            # Add symbol column for identification
            data['Symbol'] = symbol
            
            # Calculate additional metrics
            data['Daily_Return'] = data['Close'].pct_change()
            data['Volatility'] = data['Daily_Return'].rolling(window=30).std() * np.sqrt(252)
            
            logger.info(f"Successfully fetched {len(data)} records for {symbol}")
            return data
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {str(e)}")
            return None
    
    def fetch_all_assets(self) -> Dict[str, pd.DataFrame]:
        """
        Fetch historical data for all portfolio assets.
        
        Returns:
            Dict[str, pd.DataFrame]: Dictionary with asset symbols as keys and DataFrames as values
        """
        for symbol in self.assets.keys():
            asset_data = self.fetch_single_asset(symbol)
            if asset_data is not None:
                self.data[symbol] = asset_data
            
        return self.data
    
    def get_combined_data(self) -> pd.DataFrame:
        """
        Combine all asset data into a single DataFrame.
        
        Returns:
            pd.DataFrame: Combined data for all assets
        """
        if not self.data:
            logger.warning("No data available. Please fetch data first.")
            return pd.DataFrame()
            
        combined_data = pd.concat(self.data.values(), ignore_index=True)
        return combined_data
    
    def get_closing_prices(self) -> pd.DataFrame:
        """
        Get closing prices for all assets in a pivot table format.
        
        Returns:
            pd.DataFrame: Pivot table with dates as index and symbols as columns
        """
        if not self.data:
            logger.warning("No data available. Please fetch data first.")
            return pd.DataFrame()
            
        combined_data = self.get_combined_data()
        closing_prices = combined_data.pivot(index='Date', columns='Symbol', values='Close')
        return closing_prices
    
    def save_data(self, filepath: str) -> None:
        """
        Save fetched data to CSV files.
        
        Args:
            filepath (str): Base filepath for saving data
        """
        if not self.data:
            logger.warning("No data to save. Please fetch data first.")
            return
            
        for symbol, data in self.data.items():
            filename = f"{filepath}_{symbol}.csv"
            data.to_csv(filename, index=False)
            logger.info(f"Saved {symbol} data to {filename}")
            
        # Save combined closing prices
        closing_prices = self.get_closing_prices()
        closing_prices.to_csv(f"{filepath}_closing_prices.csv")
        logger.info(f"Saved combined closing prices to {filepath}_closing_prices.csv")
    
    def get_data_summary(self) -> Dict[str, Dict]:
        """
        Get summary statistics for fetched data.
        
        Returns:
            Dict[str, Dict]: Summary statistics for each asset
        """
        summary = {}
        
        for symbol, data in self.data.items():
            summary[symbol] = {
                'description': self.assets[symbol],
                'records': len(data),
                'date_range': f"{data['Date'].min().date()} to {data['Date'].max().date()}",
                'avg_close': data['Close'].mean(),
                'avg_volume': data['Volume'].mean(),
                'avg_daily_return': data['Daily_Return'].mean() * 100,  # Convert to percentage
                'avg_volatility': data['Volatility'].mean() * 100  # Convert to percentage
            }
            
        return summary


def main():
    """
    Main function to demonstrate data fetching functionality.
    """
    # Initialize fetcher
    fetcher = PortfolioDataFetcher()
    
    # Fetch all asset data
    logger.info("Starting data fetch for portfolio assets...")
    data = fetcher.fetch_all_assets()
    
    if data:
        # Print summary
        summary = fetcher.get_data_summary()
        print("\n=== Portfolio Data Summary ===")
        for symbol, stats in summary.items():
            print(f"\n{symbol}: {stats['description']}")
            print(f"  Records: {stats['records']}")
            print(f"  Date Range: {stats['date_range']}")
            print(f"  Avg Close Price: ${stats['avg_close']:.2f}")
            print(f"  Avg Daily Volume: {stats['avg_volume']:,.0f}")
            print(f"  Avg Daily Return: {stats['avg_daily_return']:.3f}%")
            print(f"  Avg Volatility: {stats['avg_volatility']:.2f}%")
        
        # Save data
        fetcher.save_data("../data/portfolio_data")
        logger.info("Data fetching completed successfully!")
    else:
        logger.error("Failed to fetch any data!")


if __name__ == "__main__":
    main()