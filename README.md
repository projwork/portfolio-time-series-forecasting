# Portfolio Time Series Forecasting - Task 1: Data Preprocessing and EDA

This project implements comprehensive data preprocessing and exploratory data analysis for portfolio time series forecasting using historical financial data for three key assets: Tesla (TSLA), Vanguard Total Bond Market ETF (BND), and S&P 500 ETF (SPY).

## 📊 Assets Overview

- **TSLA**: High-growth, high-risk stock in consumer discretionary sector (Automobile Manufacturing)
- **BND**: Bond ETF tracking U.S. investment-grade bonds, providing stability and income
- **SPY**: ETF tracking the S&P 500 Index, offering broad U.S. market exposure

## 📅 Analysis Period

July 1, 2015 - July 31, 2025 (Historical data from YFinance)

## 🏗️ Project Structure

```
portfolio-time-series-forecast/
├── data/                    # Generated data files (created after running analysis)
├── notebooks/
│   └── portfolio_analysis.ipynb    # Main analysis notebook
├── src/
│   ├── data_fetcher.py             # YFinance data fetching module
│   ├── data_preprocessor.py        # Data cleaning and preprocessing
│   ├── eda_analyzer.py             # Exploratory data analysis
│   └── risk_analyzer.py            # Risk metrics and analysis
├── scripts/
├── tests/
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## 🚀 Getting Started

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Analysis

Open and run the Jupyter notebook:

```bash
jupyter notebook notebooks/portfolio_analysis.ipynb
```

### 3. Alternative: Use Modules Directly

```python
from src.data_fetcher import PortfolioDataFetcher
from src.data_preprocessor import DataPreprocessor
from src.eda_analyzer import EDAAnalyzer
from src.risk_analyzer import RiskAnalyzer

# Fetch data
fetcher = PortfolioDataFetcher(start_date="2015-07-01", end_date="2025-07-31")
raw_data = fetcher.fetch_all_assets()

# Preprocess data
preprocessor = DataPreprocessor()
preprocessor.load_data(raw_data)
processed_data = preprocessor.get_processed_data()

# Perform EDA
eda_analyzer = EDAAnalyzer(processed_data)
eda_analyzer.plot_price_trends()

# Risk analysis
risk_analyzer = RiskAnalyzer(processed_data)
risk_report = risk_analyzer.generate_risk_report()
```

## 📋 Features Implemented

### ✅ Task 1 - Complete Implementation:

1. **Data Acquisition**

   - YFinance integration for historical data fetching
   - Automated data retrieval for TSLA, BND, SPY
   - Data validation and quality checks

2. **Data Preprocessing**

   - Missing value handling (forward fill, interpolation)
   - Data type validation and cleaning
   - Outlier detection using IQR and Z-score methods
   - Technical indicator calculation (SMA, EMA, MACD, Bollinger Bands, RSI)

3. **Exploratory Data Analysis**

   - Price trend visualization and analysis
   - Returns distribution and statistical analysis
   - Volatility analysis with rolling statistics
   - Correlation analysis between assets
   - Seasonality pattern detection

4. **Stationarity Testing**

   - Augmented Dickey-Fuller (ADF) tests
   - Statistical significance evaluation
   - Implications for time series modeling

5. **Risk Analysis**
   - Value at Risk (VaR) calculation (95%, 99% confidence levels)
   - Expected Shortfall (Conditional VaR)
   - Sharpe and Sortino ratio calculations
   - Maximum drawdown analysis
   - Risk metrics visualization and comparison

## 📊 Key Insights

### Asset Characteristics:

- **TSLA**: High volatility, high potential returns, suitable for aggressive portfolios
- **BND**: Low volatility, stable returns, provides diversification benefits
- **SPY**: Moderate risk-return profile, broad market exposure

### Statistical Properties:

- Price levels are non-stationary (require differencing for modeling)
- Returns are stationary (suitable for ARIMA models)
- Volatility clustering present in all assets

### Risk Profile:

- TSLA shows highest VaR and maximum drawdown
- BND provides portfolio stability with lowest volatility
- Correlation analysis supports diversification benefits

## 🛠️ Technical Implementation

### Modular Design:

- **Separation of Concerns**: Each module handles specific functionality
- **Reusable Components**: Modules can be used independently
- **Comprehensive Testing**: Built-in data validation and error handling
- **Extensible Architecture**: Easy to add new assets or analysis methods

### Data Quality Assurance:

- Automated missing value detection and handling
- Outlier identification and analysis
- Data type validation
- Statistical significance testing

### Visualization:

- Professional matplotlib/seaborn plotting
- Interactive Plotly dashboards (optional)
- Comprehensive risk visualization
- Time series trend analysis

## 📈 Next Steps

This analysis provides the foundation for:

1. Time series forecasting model development (ARIMA, LSTM, etc.)
2. Portfolio optimization strategies
3. Risk management framework implementation
4. Trading strategy backtesting
5. Real-time monitoring and alerting systems

## 📦 Dependencies

See `requirements.txt` for complete list. Key dependencies:

- `pandas`: Data manipulation and analysis
- `numpy`: Numerical computing
- `matplotlib/seaborn`: Visualization
- `yfinance`: Financial data retrieval
- `scipy`: Statistical functions
- `statsmodels`: Time series analysis
- `scikit-learn`: Machine learning utilities

## 🤝 Contributing

This is an educational project demonstrating financial time series analysis best practices. Feel free to extend the analysis or adapt for other assets.

## ⚠️ Disclaimer

This project is for educational and research purposes only. It does not constitute financial advice. Past performance does not guarantee future results.
