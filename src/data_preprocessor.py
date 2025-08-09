"""
Data preprocessing module for portfolio time series analysis.
Handles data cleaning, missing values, normalization, and basic statistics.
"""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import Dict, List, Tuple, Optional, Union
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    A class to preprocess financial time series data for portfolio analysis.
    """
    
    def __init__(self):
        """Initialize the data preprocessor."""
        self.scalers = {}
        self.original_data = {}
        self.processed_data = {}
        
    def load_data(self, data: Union[Dict[str, pd.DataFrame], pd.DataFrame]) -> None:
        """
        Load data into the preprocessor.
        
        Args:
            data: Either a dictionary of DataFrames or a single DataFrame
        """
        if isinstance(data, dict):
            self.original_data = data.copy()
        else:
            self.original_data = {'combined': data.copy()}
        
        # Initialize processed data as copy of original
        self.processed_data = {k: v.copy() for k, v in self.original_data.items()}
        
    def check_data_quality(self) -> Dict[str, Dict]:
        """
        Check data quality including missing values, data types, and basic statistics.
        
        Returns:
            Dict containing quality assessment for each dataset
        """
        quality_report = {}
        
        for symbol, data in self.processed_data.items():
            logger.info(f"Checking data quality for {symbol}...")
            
            quality_report[symbol] = {
                'total_records': len(data),
                'missing_values': data.isnull().sum().to_dict(),
                'data_types': data.dtypes.to_dict(),
                'duplicate_records': data.duplicated().sum(),
                'date_range': {
                    'start': data['Date'].min() if 'Date' in data.columns else 'N/A',
                    'end': data['Date'].max() if 'Date' in data.columns else 'N/A'
                }
            }
            
            # Check for numeric columns
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                quality_report[symbol]['basic_stats'] = data[numeric_cols].describe().to_dict()
                
                # Check for outliers using IQR method
                outlier_info = {}
                for col in numeric_cols:
                    if col not in ['Volume']:  # Skip volume as it naturally has high variance
                        Q1 = data[col].quantile(0.25)
                        Q3 = data[col].quantile(0.75)
                        IQR = Q3 - Q1
                        lower_bound = Q1 - 1.5 * IQR
                        upper_bound = Q3 + 1.5 * IQR
                        outliers = data[(data[col] < lower_bound) | (data[col] > upper_bound)]
                        outlier_info[col] = len(outliers)
                
                quality_report[symbol]['outliers'] = outlier_info
        
        return quality_report
    
    def handle_missing_values(self, method: str = 'forward_fill', columns: Optional[List[str]] = None) -> None:
        """
        Handle missing values in the data.
        
        Args:
            method: Method for handling missing values ('forward_fill', 'backward_fill', 'interpolate', 'drop')
            columns: Specific columns to process (if None, process all numeric columns)
        """
        for symbol, data in self.processed_data.items():
            logger.info(f"Handling missing values for {symbol} using {method}...")
            
            # Identify columns to process
            if columns is None:
                target_columns = data.select_dtypes(include=[np.number]).columns
            else:
                target_columns = columns
            
            initial_missing = data[target_columns].isnull().sum().sum()
            
            if method == 'forward_fill':
                data[target_columns] = data[target_columns].fillna(method='ffill')
            elif method == 'backward_fill':
                data[target_columns] = data[target_columns].fillna(method='bfill')
            elif method == 'interpolate':
                data[target_columns] = data[target_columns].interpolate(method='linear')
            elif method == 'drop':
                data.dropna(subset=target_columns, inplace=True)
            else:
                logger.warning(f"Unknown method {method}. Skipping missing value handling.")
                continue
                
            final_missing = data[target_columns].isnull().sum().sum()
            logger.info(f"Missing values reduced from {initial_missing} to {final_missing}")
    
    def detect_outliers(self, method: str = 'iqr', columns: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
        """
        Detect outliers in the data.
        
        Args:
            method: Method for outlier detection ('iqr', 'z_score', 'modified_z_score')
            columns: Specific columns to analyze (if None, analyze all numeric columns)
            
        Returns:
            Dictionary of DataFrames containing outlier information for each asset
        """
        outliers = {}
        
        for symbol, data in self.processed_data.items():
            logger.info(f"Detecting outliers for {symbol} using {method}...")
            
            # Identify columns to analyze
            if columns is None:
                target_columns = data.select_dtypes(include=[np.number]).columns
                # Exclude Volume as it naturally has high variance
                target_columns = [col for col in target_columns if col != 'Volume']
            else:
                target_columns = columns
            
            outlier_indices = set()
            
            for col in target_columns:
                if method == 'iqr':
                    Q1 = data[col].quantile(0.25)
                    Q3 = data[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    col_outliers = data[(data[col] < lower_bound) | (data[col] > upper_bound)].index
                    
                elif method == 'z_score':
                    z_scores = np.abs(stats.zscore(data[col].dropna()))
                    col_outliers = data.iloc[np.where(z_scores > 3)[0]].index
                    
                elif method == 'modified_z_score':
                    median = data[col].median()
                    mad = np.median(np.abs(data[col] - median))
                    modified_z_scores = 0.6745 * (data[col] - median) / mad
                    col_outliers = data[np.abs(modified_z_scores) > 3.5].index
                
                outlier_indices.update(col_outliers)
            
            if outlier_indices:
                outliers[symbol] = data.loc[list(outlier_indices)].copy()
                logger.info(f"Found {len(outlier_indices)} outlier records for {symbol}")
            else:
                outliers[symbol] = pd.DataFrame()
                logger.info(f"No outliers found for {symbol}")
        
        return outliers
    
    def normalize_data(self, method: str = 'standard', columns: Optional[List[str]] = None) -> None:
        """
        Normalize/scale the data.
        
        Args:
            method: Normalization method ('standard', 'minmax', 'robust')
            columns: Specific columns to normalize (if None, normalize all numeric columns except Date)
        """
        for symbol, data in self.processed_data.items():
            logger.info(f"Normalizing data for {symbol} using {method} scaling...")
            
            # Identify columns to normalize
            if columns is None:
                target_columns = data.select_dtypes(include=[np.number]).columns
                # Exclude certain columns from normalization
                exclude_cols = ['Volume']  # Volume should be kept in original scale for interpretation
                target_columns = [col for col in target_columns if col not in exclude_cols]
            else:
                target_columns = columns
            
            if method == 'standard':
                scaler = StandardScaler()
            elif method == 'minmax':
                scaler = MinMaxScaler()
            else:
                logger.warning(f"Unknown normalization method {method}. Skipping normalization.")
                continue
            
            # Fit and transform the data
            data[target_columns] = scaler.fit_transform(data[target_columns])
            
            # Store scaler for potential inverse transformation
            self.scalers[f"{symbol}_{method}"] = scaler
            logger.info(f"Applied {method} scaling to {len(target_columns)} columns")
    
    def calculate_returns(self, price_column: str = 'Close') -> None:
        """
        Calculate various types of returns.
        
        Args:
            price_column: Column name containing price data
        """
        for symbol, data in self.processed_data.items():
            if price_column not in data.columns:
                logger.warning(f"Column {price_column} not found in {symbol} data")
                continue
                
            logger.info(f"Calculating returns for {symbol}...")
            
            # Simple daily returns
            data['Daily_Return'] = data[price_column].pct_change()
            
            # Log returns
            data['Log_Return'] = np.log(data[price_column] / data[price_column].shift(1))
            
            # Cumulative returns
            data['Cumulative_Return'] = (1 + data['Daily_Return']).cumprod() - 1
            
            # Rolling returns (weekly, monthly)
            data['Weekly_Return'] = data[price_column].pct_change(periods=5)  # 5 trading days
            data['Monthly_Return'] = data[price_column].pct_change(periods=21)  # ~21 trading days
    
    def add_technical_indicators(self) -> None:
        """
        Add basic technical indicators to the data.
        """
        for symbol, data in self.processed_data.items():
            if 'Close' not in data.columns:
                continue
                
            logger.info(f"Adding technical indicators for {symbol}...")
            
            # Moving averages
            data['SMA_10'] = data['Close'].rolling(window=10).mean()
            data['SMA_20'] = data['Close'].rolling(window=20).mean()
            data['SMA_50'] = data['Close'].rolling(window=50).mean()
            
            # Exponential moving averages
            data['EMA_12'] = data['Close'].ewm(span=12).mean()
            data['EMA_26'] = data['Close'].ewm(span=26).mean()
            
            # MACD
            data['MACD'] = data['EMA_12'] - data['EMA_26']
            data['MACD_Signal'] = data['MACD'].ewm(span=9).mean()
            
            # Bollinger Bands
            data['BB_Middle'] = data['Close'].rolling(window=20).mean()
            bb_std = data['Close'].rolling(window=20).std()
            data['BB_Upper'] = data['BB_Middle'] + (bb_std * 2)
            data['BB_Lower'] = data['BB_Middle'] - (bb_std * 2)
            
            # RSI (simplified version)
            delta = data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            data['RSI'] = 100 - (100 / (1 + rs))
    
    def get_processed_data(self) -> Dict[str, pd.DataFrame]:
        """
        Get the processed data.
        
        Returns:
            Dictionary of processed DataFrames
        """
        return self.processed_data
    
    def save_processed_data(self, filepath: str) -> None:
        """
        Save processed data to CSV files.
        
        Args:
            filepath: Base filepath for saving data
        """
        for symbol, data in self.processed_data.items():
            filename = f"{filepath}_processed_{symbol}.csv"
            data.to_csv(filename, index=False)
            logger.info(f"Saved processed {symbol} data to {filename}")


def main():
    """
    Main function to demonstrate preprocessing functionality.
    """
    # This would typically load data from the data fetcher
    logger.info("Data preprocessing module loaded successfully!")
    logger.info("Use this module in conjunction with data_fetcher.py for complete preprocessing pipeline.")


if __name__ == "__main__":
    main()