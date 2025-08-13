"""
Trend Analyzer - Advanced trend detection and pattern analysis for time series forecasting.
Provides comprehensive trend analysis, pattern recognition, and anomaly detection capabilities.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union, Any
import warnings
import logging
from datetime import datetime, timedelta
from scipy import stats, signal
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')


class TrendAnalyzer:
    """
    Advanced trend analysis class for detecting patterns, trends, and anomalies in time series data.
    """
    
    def __init__(self):
        """Initialize the trend analyzer."""
        self.trend_data = {}
        self.pattern_analysis = {}
        self.anomaly_detection = {}
        self.seasonal_analysis = {}
        
    def detect_trend_direction(self, data: pd.Series, window_sizes: List[int] = [30, 60, 90]) -> Dict[str, Any]:
        """
        Detect trend direction using multiple time windows.
        
        Args:
            data: Time series data
            window_sizes: List of window sizes for trend analysis
            
        Returns:
            Dictionary with trend direction analysis
        """
        trends = {}
        
        for window in window_sizes:
            if len(data) < window:
                continue
                
            # Calculate rolling means and slopes
            rolling_mean = data.rolling(window=window).mean()
            
            # Calculate trend slope using linear regression
            slopes = []
            for i in range(window, len(data)):
                y = data.iloc[i-window+1:i+1].values
                x = np.arange(len(y))
                if len(y) > 1:
                    slope, _, r_value, p_value, _ = stats.linregress(x, y)
                    slopes.append({
                        'slope': slope,
                        'r_squared': r_value**2,
                        'p_value': p_value,
                        'date': data.index[i]
                    })
            
            # Classify trend strength
            recent_slopes = [s['slope'] for s in slopes[-10:]]  # Last 10 observations
            avg_slope = np.mean(recent_slopes) if recent_slopes else 0
            avg_r_squared = np.mean([s['r_squared'] for s in slopes[-10:]]) if slopes else 0
            
            if avg_slope > 0.5:
                direction = "Strong Uptrend"
            elif avg_slope > 0.1:
                direction = "Weak Uptrend"
            elif avg_slope < -0.5:
                direction = "Strong Downtrend"
            elif avg_slope < -0.1:
                direction = "Weak Downtrend"
            else:
                direction = "Sideways"
            
            trends[f"{window}_day"] = {
                'direction': direction,
                'slope': avg_slope,
                'strength': avg_r_squared,
                'confidence': 'High' if avg_r_squared > 0.7 else 'Medium' if avg_r_squared > 0.4 else 'Low',
                'all_slopes': slopes
            }
        
        # Overall trend consensus
        directions = [trends[key]['direction'] for key in trends.keys()]
        uptrend_count = sum(1 for d in directions if 'Uptrend' in d)
        downtrend_count = sum(1 for d in directions if 'Downtrend' in d)
        
        if uptrend_count > downtrend_count:
            consensus = "Bullish"
        elif downtrend_count > uptrend_count:
            consensus = "Bearish"
        else:
            consensus = "Neutral"
        
        return {
            'individual_trends': trends,
            'consensus': consensus,
            'trend_consistency': max(uptrend_count, downtrend_count) / len(directions) if directions else 0
        }
    
    def identify_support_resistance_levels(self, data: pd.Series, 
                                         min_touches: int = 3,
                                         tolerance: float = 0.02) -> Dict[str, Any]:
        """
        Identify support and resistance levels in the data.
        
        Args:
            data: Time series data
            min_touches: Minimum number of touches to confirm level
            tolerance: Tolerance for level identification (as fraction)
            
        Returns:
            Dictionary with support and resistance levels
        """
        # Find local minima and maxima
        peaks, _ = signal.find_peaks(data.values, prominence=data.std() * 0.5)
        troughs, _ = signal.find_peaks(-data.values, prominence=data.std() * 0.5)
        
        # Combine and sort all critical points
        critical_points = []
        for peak in peaks:
            critical_points.append({
                'index': peak,
                'price': data.iloc[peak],
                'date': data.index[peak],
                'type': 'resistance'
            })
        
        for trough in troughs:
            critical_points.append({
                'index': trough,
                'price': data.iloc[trough],
                'date': data.index[trough],
                'type': 'support'
            })
        
        # Group similar price levels
        support_levels = []
        resistance_levels = []
        
        for point_type in ['support', 'resistance']:
            points = [p for p in critical_points if p['type'] == point_type]
            if not points:
                continue
            
            # Cluster similar price levels
            prices = [p['price'] for p in points]
            price_range = max(prices) - min(prices)
            tolerance_abs = price_range * tolerance
            
            levels = []
            for price in prices:
                # Find similar prices within tolerance
                similar_prices = [p for p in prices if abs(p - price) <= tolerance_abs]
                if len(similar_prices) >= min_touches:
                    level_price = np.mean(similar_prices)
                    level_strength = len(similar_prices)
                    
                    # Check if this level is already recorded
                    if not any(abs(l['price'] - level_price) <= tolerance_abs for l in levels):
                        # Find dates of touches
                        level_touches = [p for p in points if abs(p['price'] - level_price) <= tolerance_abs]
                        
                        levels.append({
                            'price': level_price,
                            'strength': level_strength,
                            'touches': len(level_touches),
                            'touch_dates': [t['date'] for t in level_touches],
                            'last_touch': max([t['date'] for t in level_touches]),
                            'confidence': 'High' if level_strength >= 5 else 'Medium' if level_strength >= 3 else 'Low'
                        })
            
            if point_type == 'support':
                support_levels = sorted(levels, key=lambda x: x['price'], reverse=True)[:5]  # Top 5 support levels
            else:
                resistance_levels = sorted(levels, key=lambda x: x['price'])[:5]  # Top 5 resistance levels
        
        # Current price relative to levels
        current_price = data.iloc[-1]
        nearest_support = max([s['price'] for s in support_levels if s['price'] < current_price], default=None)
        nearest_resistance = min([r['price'] for r in resistance_levels if r['price'] > current_price], default=None)
        
        return {
            'support_levels': support_levels,
            'resistance_levels': resistance_levels,
            'current_price': current_price,
            'nearest_support': nearest_support,
            'nearest_resistance': nearest_resistance,
            'support_distance': ((current_price - nearest_support) / current_price * 100) if nearest_support else None,
            'resistance_distance': ((nearest_resistance - current_price) / current_price * 100) if nearest_resistance else None
        }
    
    def detect_chart_patterns(self, data: pd.Series, lookback_window: int = 60) -> Dict[str, Any]:
        """
        Detect common chart patterns in the data.
        
        Args:
            data: Time series data
            lookback_window: Window size for pattern detection
            
        Returns:
            Dictionary with detected patterns
        """
        patterns = []
        
        if len(data) < lookback_window:
            return {'patterns': patterns, 'pattern_count': 0}
        
        # Use recent data for pattern detection
        recent_data = data.tail(lookback_window)
        prices = recent_data.values
        
        # Double Top Pattern
        peaks, _ = signal.find_peaks(prices, prominence=np.std(prices) * 0.3)
        if len(peaks) >= 2:
            # Check for double top (two similar peaks)
            for i in range(len(peaks) - 1):
                peak1_price = prices[peaks[i]]
                peak2_price = prices[peaks[i + 1]]
                
                if abs(peak1_price - peak2_price) / max(peak1_price, peak2_price) < 0.05:  # 5% tolerance
                    patterns.append({
                        'type': 'Double Top',
                        'signal': 'Bearish',
                        'confidence': 'Medium',
                        'start_date': recent_data.index[peaks[i]],
                        'end_date': recent_data.index[peaks[i + 1]],
                        'description': f"Double top pattern detected at ${peak1_price:.2f} and ${peak2_price:.2f}"
                    })
        
        # Double Bottom Pattern
        troughs, _ = signal.find_peaks(-prices, prominence=np.std(prices) * 0.3)
        if len(troughs) >= 2:
            # Check for double bottom (two similar troughs)
            for i in range(len(troughs) - 1):
                trough1_price = prices[troughs[i]]
                trough2_price = prices[troughs[i + 1]]
                
                if abs(trough1_price - trough2_price) / max(trough1_price, trough2_price) < 0.05:  # 5% tolerance
                    patterns.append({
                        'type': 'Double Bottom',
                        'signal': 'Bullish',
                        'confidence': 'Medium',
                        'start_date': recent_data.index[troughs[i]],
                        'end_date': recent_data.index[troughs[i + 1]],
                        'description': f"Double bottom pattern detected at ${trough1_price:.2f} and ${trough2_price:.2f}"
                    })
        
        # Triangle Patterns (simplified detection)
        # Calculate trend lines for highs and lows
        highs = pd.Series(prices).rolling(window=5).max()
        lows = pd.Series(prices).rolling(window=5).min()
        
        # Check for converging trend lines (triangle)
        if len(highs.dropna()) > 10 and len(lows.dropna()) > 10:
            high_slope = np.polyfit(range(len(highs.dropna())), highs.dropna().values, 1)[0]
            low_slope = np.polyfit(range(len(lows.dropna())), lows.dropna().values, 1)[0]
            
            # Ascending triangle
            if abs(high_slope) < 0.1 and low_slope > 0.1:
                patterns.append({
                    'type': 'Ascending Triangle',
                    'signal': 'Bullish',
                    'confidence': 'Low',
                    'start_date': recent_data.index[0],
                    'end_date': recent_data.index[-1],
                    'description': "Ascending triangle pattern suggests potential upward breakout"
                })
            
            # Descending triangle
            elif abs(low_slope) < 0.1 and high_slope < -0.1:
                patterns.append({
                    'type': 'Descending Triangle',
                    'signal': 'Bearish',
                    'confidence': 'Low',
                    'start_date': recent_data.index[0],
                    'end_date': recent_data.index[-1],
                    'description': "Descending triangle pattern suggests potential downward breakout"
                })
        
        # Head and Shoulders (simplified)
        if len(peaks) >= 3:
            for i in range(len(peaks) - 2):
                left_shoulder = prices[peaks[i]]
                head = prices[peaks[i + 1]]
                right_shoulder = prices[peaks[i + 2]]
                
                # Check if middle peak is higher (head) and shoulders are similar
                if (head > left_shoulder and head > right_shoulder and 
                    abs(left_shoulder - right_shoulder) / max(left_shoulder, right_shoulder) < 0.1):
                    patterns.append({
                        'type': 'Head and Shoulders',
                        'signal': 'Bearish',
                        'confidence': 'Medium',
                        'start_date': recent_data.index[peaks[i]],
                        'end_date': recent_data.index[peaks[i + 2]],
                        'description': f"Head and shoulders pattern: Left ${left_shoulder:.2f}, Head ${head:.2f}, Right ${right_shoulder:.2f}"
                    })
        
        return {
            'patterns': patterns,
            'pattern_count': len(patterns),
            'bullish_signals': len([p for p in patterns if p['signal'] == 'Bullish']),
            'bearish_signals': len([p for p in patterns if p['signal'] == 'Bearish'])
        }
    
    def analyze_price_momentum(self, data: pd.Series, periods: List[int] = [14, 30, 60]) -> Dict[str, Any]:
        """
        Analyze price momentum using various indicators.
        
        Args:
            data: Time series data
            periods: List of periods for momentum calculation
            
        Returns:
            Dictionary with momentum analysis
        """
        momentum_indicators = {}
        
        for period in periods:
            if len(data) < period:
                continue
            
            # Rate of Change (ROC)
            roc = ((data - data.shift(period)) / data.shift(period) * 100).dropna()
            
            # Relative Strength Index (RSI) - simplified
            delta = data.diff()
            gain = delta.where(delta > 0, 0).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            # Momentum (simple price change)
            momentum = data - data.shift(period)
            
            # Current values
            current_roc = roc.iloc[-1] if len(roc) > 0 else 0
            current_rsi = rsi.iloc[-1] if len(rsi.dropna()) > 0 else 50
            current_momentum = momentum.iloc[-1] if len(momentum.dropna()) > 0 else 0
            
            # Momentum classification
            if current_rsi > 70:
                rsi_signal = "Overbought"
            elif current_rsi < 30:
                rsi_signal = "Oversold"
            else:
                rsi_signal = "Neutral"
            
            if current_roc > 5:
                roc_signal = "Strong Positive"
            elif current_roc > 0:
                roc_signal = "Positive"
            elif current_roc < -5:
                roc_signal = "Strong Negative"
            else:
                roc_signal = "Negative"
            
            momentum_indicators[f"{period}_period"] = {
                'roc': current_roc,
                'rsi': current_rsi,
                'momentum': current_momentum,
                'roc_signal': roc_signal,
                'rsi_signal': rsi_signal,
                'overall_momentum': 'Bullish' if current_roc > 0 and current_rsi < 70 else 'Bearish' if current_roc < 0 and current_rsi > 30 else 'Neutral'
            }
        
        # Overall momentum consensus
        bullish_count = sum(1 for indicators in momentum_indicators.values() if indicators['overall_momentum'] == 'Bullish')
        bearish_count = sum(1 for indicators in momentum_indicators.values() if indicators['overall_momentum'] == 'Bearish')
        
        if bullish_count > bearish_count:
            consensus = "Bullish Momentum"
        elif bearish_count > bullish_count:
            consensus = "Bearish Momentum"
        else:
            consensus = "Mixed Momentum"
        
        return {
            'individual_indicators': momentum_indicators,
            'consensus': consensus,
            'momentum_strength': abs(bullish_count - bearish_count) / len(momentum_indicators) if momentum_indicators else 0
        }
    
    def detect_volatility_patterns(self, data: pd.Series, window: int = 30) -> Dict[str, Any]:
        """
        Analyze volatility patterns and clustering.
        
        Args:
            data: Time series data
            window: Rolling window for volatility calculation
            
        Returns:
            Dictionary with volatility analysis
        """
        # Calculate returns and volatility
        returns = data.pct_change().dropna()
        rolling_vol = returns.rolling(window=window).std() * np.sqrt(252) * 100  # Annualized volatility
        
        # Volatility clustering detection
        high_vol_threshold = rolling_vol.quantile(0.75)
        low_vol_threshold = rolling_vol.quantile(0.25)
        
        # Identify volatility regimes
        volatility_regimes = []
        current_regime = None
        regime_start = None
        
        for date, vol in rolling_vol.dropna().items():
            if vol > high_vol_threshold:
                new_regime = 'High'
            elif vol < low_vol_threshold:
                new_regime = 'Low'
            else:
                new_regime = 'Medium'
            
            if new_regime != current_regime:
                if current_regime is not None:
                    volatility_regimes.append({
                        'regime': current_regime,
                        'start_date': regime_start,
                        'end_date': date,
                        'duration': (date - regime_start).days
                    })
                current_regime = new_regime
                regime_start = date
        
        # Add the last regime
        if current_regime is not None and regime_start is not None:
            volatility_regimes.append({
                'regime': current_regime,
                'start_date': regime_start,
                'end_date': rolling_vol.index[-1],
                'duration': (rolling_vol.index[-1] - regime_start).days
            })
        
        # Current volatility analysis
        current_vol = rolling_vol.iloc[-1] if len(rolling_vol.dropna()) > 0 else 0
        vol_percentile = (rolling_vol <= current_vol).mean() * 100
        
        # Volatility trend
        vol_trend_period = min(30, len(rolling_vol.dropna()) // 2)
        if vol_trend_period > 0:
            recent_vol = rolling_vol.dropna().tail(vol_trend_period).mean()
            older_vol = rolling_vol.dropna().head(vol_trend_period).mean()
            vol_trend = "Increasing" if recent_vol > older_vol else "Decreasing"
        else:
            vol_trend = "Stable"
        
        return {
            'current_volatility': current_vol,
            'volatility_percentile': vol_percentile,
            'volatility_trend': vol_trend,
            'volatility_regimes': volatility_regimes[-10:],  # Last 10 regimes
            'current_regime': volatility_regimes[-1]['regime'] if volatility_regimes else 'Unknown',
            'regime_stability': len([r for r in volatility_regimes if r['duration'] > 30]) / len(volatility_regimes) if volatility_regimes else 0
        }
    
    def generate_trend_report(self, data: pd.Series) -> str:
        """
        Generate comprehensive trend analysis report.
        
        Args:
            data: Time series data
            
        Returns:
            Formatted trend analysis report
        """
        # Perform all analyses
        trend_direction = self.detect_trend_direction(data)
        support_resistance = self.identify_support_resistance_levels(data)
        chart_patterns = self.detect_chart_patterns(data)
        momentum = self.analyze_price_momentum(data)
        volatility = self.detect_volatility_patterns(data)
        
        report = []
        report.append("=" * 70)
        report.append("COMPREHENSIVE TREND ANALYSIS REPORT")
        report.append("=" * 70)
        report.append(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Data Period: {data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}")
        report.append(f"Current Price: ${data.iloc[-1]:.2f}")
        
        # Trend Direction Analysis
        report.append("\n📈 TREND DIRECTION ANALYSIS")
        report.append("-" * 50)
        report.append(f"Overall Consensus: {trend_direction['consensus']}")
        report.append(f"Trend Consistency: {trend_direction['trend_consistency']:.1%}")
        
        for timeframe, trend_info in trend_direction['individual_trends'].items():
            report.append(f"\n{timeframe.replace('_', ' ').title()} Trend:")
            report.append(f"  Direction: {trend_info['direction']}")
            report.append(f"  Strength: {trend_info['strength']:.2f}")
            report.append(f"  Confidence: {trend_info['confidence']}")
        
        # Support and Resistance Levels
        report.append("\n🎯 SUPPORT & RESISTANCE LEVELS")
        report.append("-" * 50)
        if support_resistance['nearest_support']:
            report.append(f"Nearest Support: ${support_resistance['nearest_support']:.2f} ({support_resistance['support_distance']:.1f}% below current)")
        if support_resistance['nearest_resistance']:
            report.append(f"Nearest Resistance: ${support_resistance['nearest_resistance']:.2f} ({support_resistance['resistance_distance']:.1f}% above current)")
        
        report.append(f"\nKey Support Levels:")
        for i, level in enumerate(support_resistance['support_levels'][:3], 1):
            report.append(f"  {i}. ${level['price']:.2f} (Strength: {level['confidence']}, Touches: {level['touches']})")
        
        report.append(f"\nKey Resistance Levels:")
        for i, level in enumerate(support_resistance['resistance_levels'][:3], 1):
            report.append(f"  {i}. ${level['price']:.2f} (Strength: {level['confidence']}, Touches: {level['touches']})")
        
        # Chart Patterns
        report.append("\n📊 CHART PATTERNS")
        report.append("-" * 50)
        if chart_patterns['pattern_count'] > 0:
            report.append(f"Patterns Detected: {chart_patterns['pattern_count']}")
            report.append(f"Bullish Signals: {chart_patterns['bullish_signals']}")
            report.append(f"Bearish Signals: {chart_patterns['bearish_signals']}")
            
            for pattern in chart_patterns['patterns'][:5]:  # Top 5 patterns
                report.append(f"\n• {pattern['type']} ({pattern['signal']})")
                report.append(f"  {pattern['description']}")
                report.append(f"  Confidence: {pattern['confidence']}")
        else:
            report.append("No significant chart patterns detected")
        
        # Momentum Analysis
        report.append("\n⚡ MOMENTUM ANALYSIS")
        report.append("-" * 50)
        report.append(f"Momentum Consensus: {momentum['consensus']}")
        report.append(f"Momentum Strength: {momentum['momentum_strength']:.2f}")
        
        for timeframe, indicators in momentum['individual_indicators'].items():
            report.append(f"\n{timeframe.replace('_', ' ').title()}:")
            report.append(f"  RSI: {indicators['rsi']:.1f} ({indicators['rsi_signal']})")
            report.append(f"  ROC: {indicators['roc']:.1f}% ({indicators['roc_signal']})")
            report.append(f"  Overall: {indicators['overall_momentum']}")
        
        # Volatility Analysis
        report.append("\n📊 VOLATILITY ANALYSIS")
        report.append("-" * 50)
        report.append(f"Current Volatility: {volatility['current_volatility']:.1f}%")
        report.append(f"Volatility Percentile: {volatility['volatility_percentile']:.0f}%")
        report.append(f"Volatility Trend: {volatility['volatility_trend']}")
        report.append(f"Current Regime: {volatility['current_regime']}")
        
        # Summary and Outlook
        report.append("\n🎯 SUMMARY & OUTLOOK")
        report.append("-" * 50)
        
        # Generate overall assessment
        bullish_factors = []
        bearish_factors = []
        
        if trend_direction['consensus'] == 'Bullish':
            bullish_factors.append("Bullish trend consensus")
        elif trend_direction['consensus'] == 'Bearish':
            bearish_factors.append("Bearish trend consensus")
        
        if momentum['consensus'] == 'Bullish Momentum':
            bullish_factors.append("Positive momentum")
        elif momentum['consensus'] == 'Bearish Momentum':
            bearish_factors.append("Negative momentum")
        
        if chart_patterns['bullish_signals'] > chart_patterns['bearish_signals']:
            bullish_factors.append("Bullish chart patterns")
        elif chart_patterns['bearish_signals'] > chart_patterns['bullish_signals']:
            bearish_factors.append("Bearish chart patterns")
        
        report.append("Bullish Factors:")
        for factor in bullish_factors:
            report.append(f"  • {factor}")
        
        report.append("\nBearish Factors:")
        for factor in bearish_factors:
            report.append(f"  • {factor}")
        
        # Overall outlook
        if len(bullish_factors) > len(bearish_factors):
            outlook = "BULLISH"
        elif len(bearish_factors) > len(bullish_factors):
            outlook = "BEARISH"
        else:
            outlook = "NEUTRAL"
        
        report.append(f"\nOverall Technical Outlook: {outlook}")
        
        report.append("\n" + "=" * 70)
        
        return "\n".join(report)


def main():
    """
    Main function to demonstrate trend analyzer functionality.
    """
    logger.info("Trend Analyzer module loaded successfully!")
    logger.info("Available features:")
    logger.info("  - Multi-timeframe trend detection")
    logger.info("  - Support and resistance level identification")
    logger.info("  - Chart pattern recognition")
    logger.info("  - Momentum analysis")
    logger.info("  - Volatility pattern detection")
    logger.info("  - Comprehensive trend reporting")


if __name__ == "__main__":
    main()
