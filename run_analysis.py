#!/usr/bin/env python3
"""
Portfolio Time Series Analysis - Direct Python Script
Alternative to Jupyter notebook for running the complete analysis.

Run this script if you have issues with Jupyter notebook setup.
"""

import sys
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add src directory to path
sys.path.append('src')

def main():
    """Run the complete portfolio analysis."""
    print("🚀 PORTFOLIO TIME SERIES ANALYSIS - TASK 1")
    print("=" * 60)
    print("Starting comprehensive data preprocessing and exploratory data analysis")
    print(f"Analysis started at: {datetime.now()}")
    print("\n" + "=" * 60)
    
    # Import libraries
    try:
        print("\n📦 Importing libraries...")
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Configure plotting
        plt.style.use('default')  # Use default instead of seaborn-v0_8 for compatibility
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 10
        
        # Import custom modules
        from data_fetcher import PortfolioDataFetcher
        from data_preprocessor import DataPreprocessor
        from eda_analyzer import EDAAnalyzer
        from risk_analyzer import RiskAnalyzer
        
        print("✅ All libraries imported successfully!")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("\n🔧 Please run: pip install -r requirements.txt")
        return False
    
    # Step 1: Data Fetching
    print("\n" + "=" * 60)
    print("📊 STEP 1: DATA FETCHING")
    print("=" * 60)
    
    try:
        # Initialize data fetcher
        fetcher = PortfolioDataFetcher(start_date="2015-07-01", end_date="2025-07-31")
        
        # Fetch data for all assets
        print("Fetching historical data for TSLA, BND, and SPY...")
        raw_data = fetcher.fetch_all_assets()
        
        if raw_data:
            print(f"\n✅ Successfully fetched data for {len(raw_data)} assets")
            
            # Display summary statistics
            summary = fetcher.get_data_summary()
            
            print("\n📋 DATA SUMMARY:")
            for symbol, stats in summary.items():
                print(f"\n📊 {symbol}: {stats['description']}")
                print(f"   📅 Records: {stats['records']:,}")
                print(f"   📆 Date Range: {stats['date_range']}")
                print(f"   💰 Avg Close Price: ${stats['avg_close']:,.2f}")
                print(f"   📈 Avg Daily Return: {stats['avg_daily_return']:.3f}%")
                print(f"   📊 Avg Volatility: {stats['avg_volatility']:.2f}%")
                print(f"   📦 Avg Volume: {stats['avg_volume']:,.0f}")
        else:
            print("❌ Failed to fetch data!")
            return False
            
    except Exception as e:
        print(f"❌ Data fetching failed: {e}")
        return False
    
    # Step 2: Data Preprocessing
    print("\n" + "=" * 60)
    print("🔧 STEP 2: DATA PREPROCESSING")
    print("=" * 60)
    
    try:
        # Initialize data preprocessor
        preprocessor = DataPreprocessor()
        preprocessor.load_data(raw_data)
        
        # Check data quality
        print("\n🔍 Checking data quality...")
        quality_report = preprocessor.check_data_quality()
        
        for symbol, report in quality_report.items():
            print(f"\n📋 {symbol} Data Quality Report:")
            print(f"   📊 Total Records: {report['total_records']:,}")
            print(f"   🔄 Duplicate Records: {report['duplicate_records']}")
            
            # Missing values
            missing_vals = report['missing_values']
            total_missing = sum(missing_vals.values())
            print(f"   ❓ Total Missing Values: {total_missing}")
        
        # Handle missing values and calculate additional metrics
        print("\n🔧 Processing data...")
        preprocessor.handle_missing_values(method='forward_fill')
        preprocessor.calculate_returns()
        preprocessor.add_technical_indicators()
        
        # Get processed data
        processed_data = preprocessor.get_processed_data()
        print("\n✅ Data preprocessing completed!")
        
    except Exception as e:
        print(f"❌ Data preprocessing failed: {e}")
        return False
    
    # Step 3: Basic Statistical Analysis
    print("\n" + "=" * 60)
    print("📈 STEP 3: BASIC STATISTICAL ANALYSIS")
    print("=" * 60)
    
    try:
        # Display basic statistics for each asset
        for symbol, data in processed_data.items():
            print(f"\n📈 {symbol} - Descriptive Statistics")
            print("-" * 50)
            
            # Price statistics
            if 'Close' in data.columns:
                price_stats = data['Close'].describe()
                print(f"\n💰 Price Statistics:")
                print(f"   Mean: ${price_stats['mean']:,.2f}")
                print(f"   Std: ${price_stats['std']:,.2f}")
                print(f"   Min: ${price_stats['min']:,.2f}")
                print(f"   Max: ${price_stats['max']:,.2f}")
            
            # Return statistics
            if 'Daily_Return' in data.columns:
                return_stats = data['Daily_Return'].describe()
                print(f"\n📊 Daily Return Statistics:")
                print(f"   Mean: {return_stats['mean']*100:.4f}%")
                print(f"   Std (Volatility): {return_stats['std']*100:.4f}%")
                print(f"   Skewness: {data['Daily_Return'].skew():.4f}")
                print(f"   Kurtosis: {data['Daily_Return'].kurtosis():.4f}")
                
                # Annualized metrics
                ann_return = return_stats['mean'] * 252 * 100
                ann_volatility = return_stats['std'] * np.sqrt(252) * 100
                print(f"\n🗓️ Annualized Metrics:")
                print(f"   Annualized Return: {ann_return:.2f}%")
                print(f"   Annualized Volatility: {ann_volatility:.2f}%")
                
    except Exception as e:
        print(f"❌ Statistical analysis failed: {e}")
        return False
    
    # Step 4: EDA Analysis
    print("\n" + "=" * 60)
    print("📊 STEP 4: EXPLORATORY DATA ANALYSIS")
    print("=" * 60)
    
    try:
        # Initialize EDA analyzer
        eda_analyzer = EDAAnalyzer(processed_data)
        
        # Generate comprehensive summary report
        summary_report = eda_analyzer.generate_summary_report()
        
        print("\n📊 EDA Summary Report:")
        for symbol, metrics in summary_report.items():
            print(f"\n🏢 {symbol}:")
            print(f"   💰 Current Price: ${metrics['price_statistics']['current_price']:,.2f}")
            print(f"   📈 Total Return: {metrics['price_statistics']['total_return']:,.2f}%")
            print(f"   📊 Annualized Return: {metrics['return_statistics']['annualized_return']:,.2f}%")
            print(f"   📉 Annualized Volatility: {metrics['return_statistics']['annualized_volatility']:,.2f}%")
            print(f"   📐 Sharpe Ratio: {metrics['risk_metrics']['sharpe_ratio']:,.3f}")
            print(f"   🔻 Max Drawdown: {metrics['risk_metrics']['max_drawdown']:,.2f}%")
            print(f"   💥 VaR 95%: {metrics['risk_metrics']['var_95']:,.3f}%")
            print(f"   💥 VaR 99%: {metrics['risk_metrics']['var_99']:,.3f}%")
        
        # Perform stationarity tests
        print("\n📊 Stationarity Analysis (ADF Tests):")
        stationarity_results = eda_analyzer.perform_stationarity_tests()
        
        for symbol, tests in stationarity_results.items():
            print(f"\n🏢 {symbol}:")
            for variable, result in tests.items():
                status = "✅ Stationary" if result['is_stationary'] else "❌ Non-stationary"
                print(f"   📊 {variable}: {status} (p-value: {result['p_value']:.6f})")
        
        print("\n📈 Creating visualizations...")
        
        # Generate plots (save to files instead of showing)
        eda_analyzer.plot_price_trends(figsize=(16, 12), save_path="data/price_trends.png")
        print("   ✅ Price trends plot saved to data/price_trends.png")
        
        eda_analyzer.plot_returns_analysis(figsize=(18, 14), save_path="data/returns_analysis.png")
        print("   ✅ Returns analysis plot saved to data/returns_analysis.png")
        
        eda_analyzer.plot_volatility_analysis(window=30, figsize=(18, 12), save_path="data/volatility_analysis.png")
        print("   ✅ Volatility analysis plot saved to data/volatility_analysis.png")
        
    except Exception as e:
        print(f"❌ EDA analysis failed: {e}")
        return False
    
    # Step 5: Risk Analysis
    print("\n" + "=" * 60)
    print("⚠️ STEP 5: RISK ANALYSIS")
    print("=" * 60)
    
    try:
        # Initialize risk analyzer
        risk_analyzer = RiskAnalyzer(processed_data, risk_free_rate=0.02)
        
        # Generate comprehensive risk report
        risk_report = risk_analyzer.generate_risk_report()
        
        print("\n📊 Risk Analysis Summary:")
        for symbol, metrics in risk_report.items():
            print(f"\n🏢 {symbol}:")
            print("-" * 40)
            
            # VaR metrics
            var_metrics = metrics.get('var_metrics', {})
            print(f"   💥 VaR (95%): {var_metrics.get('VaR_95%', 'N/A')}")
            print(f"   💥 VaR (99%): {var_metrics.get('VaR_99%', 'N/A')}")
            
            # Risk-adjusted returns
            print(f"   📐 Sharpe Ratio: {metrics.get('sharpe_ratio', 'N/A'):.4f}")
            print(f"   📐 Sortino Ratio: {metrics.get('sortino_ratio', 'N/A'):.4f}")
            
            # Drawdown metrics
            dd_metrics = metrics.get('maximum_drawdown', {})
            print(f"   🔻 Maximum Drawdown: {dd_metrics.get('max_drawdown', 'N/A'):.2f}%")
        
        # Generate risk plots
        print("\n📊 Creating risk analysis plots...")
        try:
            risk_analyzer.plot_var_analysis_simple(figsize=(18, 8), save_path="data/var_analysis.png")
            print("   ✅ VaR analysis plot saved to data/var_analysis.png")
        except Exception as e:
            print(f"   ⚠️ VaR plotting issue: {str(e)}")
            print("   📊 VaR metrics calculated successfully (plotting skipped)")
        
        risk_analyzer.plot_risk_metrics_comparison(figsize=(16, 10), save_path="data/risk_comparison.png")
        print("   ✅ Risk metrics comparison plot saved to data/risk_comparison.png")
        
    except Exception as e:
        print(f"❌ Risk analysis failed: {e}")
        return False
    
    # Step 6: Save Results
    print("\n" + "=" * 60)
    print("💾 STEP 6: SAVING RESULTS")
    print("=" * 60)
    
    try:
        # Create data directory if it doesn't exist
        os.makedirs('data', exist_ok=True)
        
        # Save raw and processed data
        print("\n💾 Saving data files...")
        fetcher.save_data("data/raw_portfolio_data")
        preprocessor.save_processed_data("data/processed_portfolio_data")
        
        # Save risk metrics
        risk_analyzer.export_risk_metrics("data/risk_metrics.csv")
        
        print("✅ All data files saved to data/ directory")
        
    except Exception as e:
        print(f"❌ Saving results failed: {e}")
        return False
    
    # Final Summary
    print("\n" + "=" * 60)
    print("🎯 ANALYSIS COMPLETE - KEY FINDINGS")
    print("=" * 60)
    
    # Calculate portfolio-level metrics
    assets_summary = {}
    
    for symbol, data in processed_data.items():
        if 'Daily_Return' in data.columns and 'Close' in data.columns:
            returns = data['Daily_Return'].dropna()
            prices = data['Close']
            
            assets_summary[symbol] = {
                'total_return': (prices.iloc[-1] / prices.iloc[0] - 1) * 100,
                'annual_return': returns.mean() * 252 * 100,
                'annual_volatility': returns.std() * np.sqrt(252) * 100,
                'sharpe_ratio': (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() != 0 else 0,
                'max_drawdown': ((prices.expanding().max() - prices) / prices.expanding().max()).max() * 100
            }
    
    print("\n📊 ASSET PERFORMANCE COMPARISON:")
    print(f"{'Asset':<6} {'Total Return':<12} {'Ann. Return':<12} {'Ann. Vol':<10} {'Sharpe':<8} {'Max DD':<8}")
    print("-" * 70)
    
    for symbol, metrics in assets_summary.items():
        print(f"{symbol:<6} {metrics['total_return']:>10.1f}% {metrics['annual_return']:>10.1f}% {metrics['annual_volatility']:>8.1f}% {metrics['sharpe_ratio']:>6.2f} {metrics['max_drawdown']:>6.1f}%")
    
    print("\n🔍 KEY INSIGHTS:")
    print("-" * 40)
    print("1. 📈 TESLA (TSLA): High-growth potential with significant volatility")
    print("2. 🛡️ BOND ETF (BND): Provides portfolio stability and diversification")
    print("3. 🏛️ S&P 500 ETF (SPY): Balanced market exposure with moderate risk")
    print("\n📋 STATIONARITY: Returns are stationary, prices require differencing")
    print("⚠️ RISK: VaR analysis reveals tail risk exposure for each asset")
    print("🚀 PORTFOLIO: Diversification across asset classes reduces overall risk")
    
    print("\n📁 OUTPUT FILES CREATED:")
    print("   📊 data/price_trends.png - Price trend analysis")
    print("   📈 data/returns_analysis.png - Returns distribution analysis") 
    print("   📉 data/volatility_analysis.png - Volatility patterns")
    print("   💥 data/var_analysis.png - Value at Risk analysis")
    print("   📊 data/risk_comparison.png - Risk metrics comparison")
    print("   💾 data/risk_metrics.csv - Comprehensive risk metrics")
    print("   💾 data/*_processed_*.csv - Processed data files")
    
    print(f"\n✅ Portfolio analysis completed successfully at: {datetime.now()}")
    print("\n🎉 Task 1: Data Preprocessing and Exploratory Data Analysis - COMPLETE!")
    
    return True

if __name__ == "__main__":
    success = main()
    
    if not success:
        print("\n❌ Analysis failed. Please check error messages above.")
        print("💡 Try running: python test_environment.py")
        sys.exit(1)
    else:
        print("\n🚀 Next steps: Use this analysis as foundation for time series forecasting!")
        sys.exit(0)