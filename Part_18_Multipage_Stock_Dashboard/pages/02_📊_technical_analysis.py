"""
Technical Analysis Page
======================

Advanced technical analysis with multiple indicators, signals, and market insights.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

# Import configuration and utilities
from config.settings import STREAMLIT_CONFIG, CHART_CONFIG
from config.stock_symbols import get_stock_info, TIME_PERIODS, TECHNICAL_INDICATORS
from utils.data_manager import data_manager
from utils.chart_components import chart_components

# Configure page
st.set_page_config(**STREAMLIT_CONFIG)

class TechnicalAnalyzer:
    """Advanced technical analysis calculations"""
    
    def __init__(self):
        self.colors = CHART_CONFIG['colors']
    
    def calculate_rsi(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_macd(self, df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
        """Calculate MACD indicator"""
        ema_fast = df['close'].ewm(span=fast).mean()
        ema_slow = df['close'].ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line
        
        return {
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram
        }
    
    def calculate_bollinger_bands(self, df: pd.DataFrame, period: int = 20, std_dev: float = 2) -> Dict[str, pd.Series]:
        """Calculate Bollinger Bands"""
        sma = df['close'].rolling(window=period).mean()
        std = df['close'].rolling(window=period).std()
        
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        
        return {
            'upper': upper_band,
            'middle': sma,
            'lower': lower_band,
            'bandwidth': (upper_band - lower_band) / sma * 100,
            'position': (df['close'] - lower_band) / (upper_band - lower_band) * 100
        }
    
    def calculate_stochastic(self, df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> Dict[str, pd.Series]:
        """Calculate Stochastic Oscillator"""
        lowest_low = df['low'].rolling(window=k_period).min()
        highest_high = df['high'].rolling(window=k_period).max()
        
        k_percent = ((df['close'] - lowest_low) / (highest_high - lowest_low)) * 100
        d_percent = k_percent.rolling(window=d_period).mean()
        
        return {
            'k': k_percent,
            'd': d_percent
        }
    
    def calculate_williams_r(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Williams %R"""
        highest_high = df['high'].rolling(window=period).max()
        lowest_low = df['low'].rolling(window=period).min()
        
        williams_r = ((highest_high - df['close']) / (highest_high - lowest_low)) * -100
        return williams_r
    
    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift())
        low_close = abs(df['low'] - df['close'].shift())
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()
        return atr
    
    def detect_signals(self, df: pd.DataFrame, indicators: Dict) -> List[Dict]:
        """Detect buy/sell signals based on multiple indicators"""
        signals = []
        
        # RSI signals
        if 'rsi' in indicators:
            rsi = indicators['rsi']
            oversold = rsi < 30
            overbought = rsi > 70
            
            for i in range(1, len(rsi)):
                if oversold.iloc[i-1] and not oversold.iloc[i] and rsi.iloc[i] < 40:
                    signals.append({
                        'date': rsi.index[i],
                        'type': 'BUY',
                        'indicator': 'RSI',
                        'strength': 'Medium',
                        'reason': f'RSI oversold reversal ({rsi.iloc[i]:.1f})',
                        'price': df['close'].iloc[i]
                    })
                elif overbought.iloc[i-1] and not overbought.iloc[i] and rsi.iloc[i] > 60:
                    signals.append({
                        'date': rsi.index[i],
                        'type': 'SELL',
                        'indicator': 'RSI',
                        'strength': 'Medium',
                        'reason': f'RSI overbought reversal ({rsi.iloc[i]:.1f})',
                        'price': df['close'].iloc[i]
                    })
        
        # MACD signals
        if 'macd' in indicators:
            macd = indicators['macd']['macd']
            signal = indicators['macd']['signal']
            
            for i in range(1, len(macd)):
                if macd.iloc[i-1] <= signal.iloc[i-1] and macd.iloc[i] > signal.iloc[i]:
                    signals.append({
                        'date': macd.index[i],
                        'type': 'BUY',
                        'indicator': 'MACD',
                        'strength': 'Strong',
                        'reason': 'MACD bullish crossover',
                        'price': df['close'].iloc[i]
                    })
                elif macd.iloc[i-1] >= signal.iloc[i-1] and macd.iloc[i] < signal.iloc[i]:
                    signals.append({
                        'date': macd.index[i],
                        'type': 'SELL',
                        'indicator': 'MACD',
                        'strength': 'Strong',
                        'reason': 'MACD bearish crossover',
                        'price': df['close'].iloc[i]
                    })
        
        # Bollinger Bands signals
        if 'bollinger' in indicators:
            bb = indicators['bollinger']
            close = df['close']
            
            for i in range(1, len(close)):
                if close.iloc[i-1] <= bb['lower'].iloc[i-1] and close.iloc[i] > bb['lower'].iloc[i]:
                    signals.append({
                        'date': close.index[i],
                        'type': 'BUY',
                        'indicator': 'Bollinger Bands',
                        'strength': 'Medium',
                        'reason': 'Price bounce from lower band',
                        'price': close.iloc[i]
                    })
                elif close.iloc[i-1] >= bb['upper'].iloc[i-1] and close.iloc[i] < bb['upper'].iloc[i]:
                    signals.append({
                        'date': close.index[i],
                        'type': 'SELL',
                        'indicator': 'Bollinger Bands',
                        'strength': 'Medium',
                        'reason': 'Price rejection from upper band',
                        'price': close.iloc[i]
                    })
        
        # Sort signals by date and return recent ones
        signals.sort(key=lambda x: x['date'], reverse=True)
        return signals[:20]  # Return last 20 signals
    
    def analyze_trend(self, df: pd.DataFrame) -> Dict:
        """Analyze overall market trend"""
        if len(df) < 50:
            return {'trend': 'Insufficient Data', 'strength': 0, 'description': 'Not enough data for analysis'}
        
        # Use multiple moving averages for trend analysis
        sma_20 = df['close'].rolling(20).mean()
        sma_50 = df['close'].rolling(50).mean()
        
        current_price = df['close'].iloc[-1]
        sma_20_current = sma_20.iloc[-1]
        sma_50_current = sma_50.iloc[-1]
        
        # Calculate trend strength
        price_vs_sma20 = (current_price - sma_20_current) / sma_20_current * 100
        sma20_vs_sma50 = (sma_20_current - sma_50_current) / sma_50_current * 100
        
        # Trend determination
        if current_price > sma_20_current > sma_50_current:
            if price_vs_sma20 > 5:
                trend = 'Strong Bullish'
                strength = 8
            elif price_vs_sma20 > 2:
                trend = 'Bullish'
                strength = 6
            else:
                trend = 'Weak Bullish'
                strength = 4
        elif current_price < sma_20_current < sma_50_current:
            if price_vs_sma20 < -5:
                trend = 'Strong Bearish'
                strength = 2
            elif price_vs_sma20 < -2:
                trend = 'Bearish'
                strength = 3
            else:
                trend = 'Weak Bearish'
                strength = 4
        else:
            trend = 'Sideways'
            strength = 5
        
        return {
            'trend': trend,
            'strength': strength,
            'description': f'Price is {price_vs_sma20:.1f}% from 20-day MA, MA20 is {sma20_vs_sma50:.1f}% from MA50'
        }
    
    def calculate_support_resistance(self, df: pd.DataFrame, window: int = 20) -> Dict:
        """Calculate support and resistance levels"""
        if len(df) < window * 2:
            return {'support': [], 'resistance': []}
        
        # Find local minima and maxima
        highs = df['high'].rolling(window=window, center=True).max() == df['high']
        lows = df['low'].rolling(window=window, center=True).min() == df['low']
        
        resistance_levels = df.loc[highs, 'high'].dropna().tail(5).tolist()
        support_levels = df.loc[lows, 'low'].dropna().tail(5).tolist()
        
        return {
            'resistance': sorted(resistance_levels, reverse=True),
            'support': sorted(support_levels, reverse=True)
        }

# Initialize analyzer
analyzer = TechnicalAnalyzer()

def get_current_selection():
    """Get current stock and period selection from session state"""
    current_stock = st.session_state.get('selected_stock', 'AAPL')
    current_period = st.session_state.get('selected_period', '1Y')
    return current_stock, current_period

def display_technical_header(symbol: str):
    """Display technical analysis header"""
    stock_info = get_stock_info(symbol)
    
    st.title(f"📊 Technical Analysis - {stock_info.get('name', symbol)} ({symbol})")
    
    # Current price and basic info
    try:
        metrics = data_manager.get_stock_metrics(symbol)
        if metrics:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Current Price",
                    f"${metrics['current_price']:.2f}",
                    f"{metrics['price_change']:+.2f} ({metrics['price_change_pct']:+.2f}%)"
                )
            
            with col2:
                st.metric("Volume", f"{metrics['volume']:,}")
            
            with col3:
                # Add volatility calculation
                df = data_manager.load_stock_data(symbol)
                if not df.empty and len(df) > 20:
                    returns = df['close'].pct_change().dropna()
                    volatility = returns.tail(20).std() * np.sqrt(252) * 100
                    st.metric("Volatility (20d)", f"{volatility:.1f}%")
            
            with col4:
                # Trend analysis
                trend_analysis = analyzer.analyze_trend(df)
                trend_color = "🟢" if "Bullish" in trend_analysis['trend'] else "🔴" if "Bearish" in trend_analysis['trend'] else "🟡"
                st.metric("Trend", f"{trend_color} {trend_analysis['trend']}")
    
    except Exception as e:
        st.error(f"Error loading header metrics: {e}")

def create_indicator_controls():
    """Create sidebar controls for technical indicators"""
    st.sidebar.subheader("📊 Technical Indicators")
    
    # RSI Settings
    st.sidebar.markdown("**RSI Settings**")
    rsi_enabled = st.sidebar.checkbox("Enable RSI", value=True)
    rsi_period = st.sidebar.slider("RSI Period", 10, 30, 14) if rsi_enabled else 14
    
    # MACD Settings
    st.sidebar.markdown("**MACD Settings**")
    macd_enabled = st.sidebar.checkbox("Enable MACD", value=True)
    if macd_enabled:
        col1, col2 = st.sidebar.columns(2)
        with col1:
            macd_fast = st.slider("Fast", 8, 16, 12)
            macd_signal = st.slider("Signal", 7, 11, 9)
        with col2:
            macd_slow = st.slider("Slow", 20, 32, 26)
    else:
        macd_fast, macd_slow, macd_signal = 12, 26, 9
    
    # Bollinger Bands Settings
    st.sidebar.markdown("**Bollinger Bands Settings**")
    bb_enabled = st.sidebar.checkbox("Enable Bollinger Bands", value=True)
    if bb_enabled:
        bb_period = st.sidebar.slider("BB Period", 15, 25, 20)
        bb_std = st.sidebar.slider("Standard Deviation", 1.5, 3.0, 2.0, 0.1)
    else:
        bb_period, bb_std = 20, 2.0
    
    # Stochastic Settings
    st.sidebar.markdown("**Stochastic Settings**")
    stoch_enabled = st.sidebar.checkbox("Enable Stochastic", value=True)
    if stoch_enabled:
        stoch_k = st.sidebar.slider("K Period", 10, 20, 14)
        stoch_d = st.sidebar.slider("D Period", 2, 5, 3)
    else:
        stoch_k, stoch_d = 14, 3
    
    # Williams %R Settings
    st.sidebar.markdown("**Williams %R Settings**")
    williams_enabled = st.sidebar.checkbox("Enable Williams %R", value=False)
    williams_period = st.sidebar.slider("Williams Period", 10, 20, 14) if williams_enabled else 14
    
    # Additional Moving Averages
    st.sidebar.markdown("**Additional Moving Averages**")
    ma_enabled = st.sidebar.checkbox("Enable Additional MAs", value=True)
    if ma_enabled:
        ma_periods = st.sidebar.multiselect(
            "MA Periods",
            [10, 20, 50, 100, 200],
            default=[20, 50]
        )
    else:
        ma_periods = []
    
    return {
        'rsi': {'enabled': rsi_enabled, 'period': rsi_period},
        'macd': {'enabled': macd_enabled, 'fast': macd_fast, 'slow': macd_slow, 'signal': macd_signal},
        'bollinger': {'enabled': bb_enabled, 'period': bb_period, 'std': bb_std},
        'stochastic': {'enabled': stoch_enabled, 'k': stoch_k, 'd': stoch_d},
        'williams': {'enabled': williams_enabled, 'period': williams_period},
        'ma': {'enabled': ma_enabled, 'periods': ma_periods}
    }

def calculate_all_indicators(df: pd.DataFrame, settings: Dict) -> Dict:
    """Calculate all selected technical indicators"""
    indicators = {}
    
    try:
        # RSI
        if settings['rsi']['enabled']:
            indicators['rsi'] = analyzer.calculate_rsi(df, settings['rsi']['period'])
        
        # MACD
        if settings['macd']['enabled']:
            indicators['macd'] = analyzer.calculate_macd(
                df, 
                settings['macd']['fast'], 
                settings['macd']['slow'], 
                settings['macd']['signal']
            )
        
        # Bollinger Bands
        if settings['bollinger']['enabled']:
            indicators['bollinger'] = analyzer.calculate_bollinger_bands(
                df, 
                settings['bollinger']['period'], 
                settings['bollinger']['std']
            )
        
        # Stochastic
        if settings['stochastic']['enabled']:
            indicators['stochastic'] = analyzer.calculate_stochastic(
                df, 
                settings['stochastic']['k'], 
                settings['stochastic']['d']
            )
        
        # Williams %R
        if settings['williams']['enabled']:
            indicators['williams_r'] = analyzer.calculate_williams_r(df, settings['williams']['period'])
        
        # Additional Moving Averages
        if settings['ma']['enabled']:
            indicators['moving_averages'] = {}
            for period in settings['ma']['periods']:
                indicators['moving_averages'][f'ma_{period}'] = df['close'].rolling(period).mean()
        
        # ATR (always calculate for volatility analysis)
        indicators['atr'] = analyzer.calculate_atr(df)
        
    except Exception as e:
        st.error(f"Error calculating indicators: {e}")
    
    return indicators

def create_technical_chart(df: pd.DataFrame, indicators: Dict, settings: Dict, symbol: str) -> go.Figure:
    """Create comprehensive technical analysis chart"""
    
    # Determine number of subplots needed
    subplot_count = 1  # Main price chart
    subplot_titles = ["Price & Volume"]
    
    if settings['rsi']['enabled']:
        subplot_count += 1
        subplot_titles.append("RSI")
    
    if settings['macd']['enabled']:
        subplot_count += 1
        subplot_titles.append("MACD")
    
    if settings['stochastic']['enabled'] or settings['williams']['enabled']:
        subplot_count += 1
        title = []
        if settings['stochastic']['enabled']:
            title.append("Stochastic")
        if settings['williams']['enabled']:
            title.append("Williams %R")
        subplot_titles.append(" & ".join(title))
    
    # Create subplots
    row_heights = [0.6] + [0.4/(subplot_count-1)]*(subplot_count-1) if subplot_count > 1 else [1.0]
    
    fig = make_subplots(
        rows=subplot_count,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=subplot_titles,
        row_heights=row_heights
    )
    
    # Main candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name="Price",
            increasing_line_color=CHART_CONFIG['colors']['increasing'],
            decreasing_line_color=CHART_CONFIG['colors']['decreasing']
        ),
        row=1, col=1
    )
    
    # Add Bollinger Bands
    if 'bollinger' in indicators:
        bb = indicators['bollinger']
        fig.add_trace(
            go.Scatter(
                x=df.index, y=bb['upper'],
                name='BB Upper',
                line=dict(color='rgba(173, 204, 255, 0.8)', width=1),
                fill=None
            ),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=df.index, y=bb['lower'],
                name='BB Lower',
                line=dict(color='rgba(173, 204, 255, 0.8)', width=1),
                fill='tonexty',
                fillcolor='rgba(173, 204, 255, 0.1)'
            ),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=df.index, y=bb['middle'],
                name='BB Middle',
                line=dict(color='rgba(173, 204, 255, 0.8)', width=1, dash='dash')
            ),
            row=1, col=1
        )
    
    # Add Moving Averages
    if 'moving_averages' in indicators:
        colors = ['orange', 'purple', 'green', 'red', 'brown']
        for i, (name, ma) in enumerate(indicators['moving_averages'].items()):
            period = name.split('_')[1]
            fig.add_trace(
                go.Scatter(
                    x=df.index, y=ma,
                    name=f'MA {period}',
                    line=dict(color=colors[i % len(colors)], width=2)
                ),
                row=1, col=1
            )
    
    current_row = 2
    
    # RSI subplot
    if 'rsi' in indicators:
        rsi = indicators['rsi']
        fig.add_trace(
            go.Scatter(
                x=df.index, y=rsi,
                name='RSI',
                line=dict(color=CHART_CONFIG['colors']['rsi'], width=2)
            ),
            row=current_row, col=1
        )
        
        # Add RSI reference lines
        fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5, row=current_row, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5, row=current_row, col=1)
        fig.add_hline(y=50, line_dash="dash", line_color="gray", opacity=0.3, row=current_row, col=1)
        
        fig.update_yaxes(range=[0, 100], row=current_row, col=1)
        current_row += 1
    
    # MACD subplot
    if 'macd' in indicators:
        macd_data = indicators['macd']
        
        fig.add_trace(
            go.Scatter(
                x=df.index, y=macd_data['macd'],
                name='MACD',
                line=dict(color=CHART_CONFIG['colors']['macd'], width=2)
            ),
            row=current_row, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=df.index, y=macd_data['signal'],
                name='Signal',
                line=dict(color=CHART_CONFIG['colors']['signal'], width=2)
            ),
            row=current_row, col=1
        )
        
        # MACD Histogram
        colors = ['green' if val >= 0 else 'red' for val in macd_data['histogram']]
        fig.add_trace(
            go.Bar(
                x=df.index, y=macd_data['histogram'],
                name='Histogram',
                marker_color=colors,
                opacity=0.6
            ),
            row=current_row, col=1
        )
        
        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.3, row=current_row, col=1)
        current_row += 1
    
    # Stochastic/Williams %R subplot
    if settings['stochastic']['enabled'] or settings['williams']['enabled']:
        if 'stochastic' in indicators:
            stoch = indicators['stochastic']
            fig.add_trace(
                go.Scatter(
                    x=df.index, y=stoch['k'],
                    name='%K',
                    line=dict(color='blue', width=2)
                ),
                row=current_row, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=df.index, y=stoch['d'],
                    name='%D',
                    line=dict(color='red', width=2)
                ),
                row=current_row, col=1
            )
        
        if 'williams_r' in indicators:
            fig.add_trace(
                go.Scatter(
                    x=df.index, y=indicators['williams_r'],
                    name='Williams %R',
                    line=dict(color='purple', width=2)
                ),
                row=current_row, col=1
            )
        
        # Add reference lines
        fig.add_hline(y=80, line_dash="dash", line_color="red", opacity=0.5, row=current_row, col=1)
        fig.add_hline(y=20, line_dash="dash", line_color="green", opacity=0.5, row=current_row, col=1)
        fig.add_hline(y=50, line_dash="dash", line_color="gray", opacity=0.3, row=current_row, col=1)
        
        if settings['williams']['enabled']:
            fig.update_yaxes(range=[-100, 0], row=current_row, col=1)
        else:
            fig.update_yaxes(range=[0, 100], row=current_row, col=1)
    
    # Update layout
    fig.update_layout(
        title=f"{symbol} - Technical Analysis",
        template=CHART_CONFIG['theme'],
        height=800,
        showlegend=True,
        xaxis_rangeslider_visible=False
    )
    
    # Update x-axes
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
    
    # Update y-axes
    for i in range(1, subplot_count + 1):
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)', row=i, col=1)
    
    return fig

def display_signals_analysis(df: pd.DataFrame, indicators: Dict, symbol: str):
    """Display trading signals and analysis"""
    
    st.subheader("🎯 Trading Signals & Analysis")
    
    try:
        # Detect signals
        signals = analyzer.detect_signals(df, indicators)
        
        if signals:
            st.markdown("### 📊 Recent Signals")
            
            # Create signals dataframe
            signals_df = pd.DataFrame(signals)
            signals_df['date'] = pd.to_datetime(signals_df['date']).dt.strftime('%Y-%m-%d')
            
            # Color code signals
            def style_signal_type(val):
                if val == 'BUY':
                    return 'background-color: rgba(0, 255, 136, 0.2); color: green; font-weight: bold'
                elif val == 'SELL':
                    return 'background-color: rgba(255, 107, 107, 0.2); color: red; font-weight: bold'
                return ''
            
            def style_strength(val):
                if val == 'Strong':
                    return 'font-weight: bold; color: darkblue'
                elif val == 'Medium':
                    return 'color: orange'
                return 'color: gray'
            
            styled_df = signals_df.style.applymap(style_signal_type, subset=['type']).applymap(style_strength, subset=['strength'])
            
            st.dataframe(styled_df, use_container_width=True)
            
            # Signal summary
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                buy_signals = len([s for s in signals if s['type'] == 'BUY'])
                st.metric("Buy Signals", buy_signals)
            
            with col2:
                sell_signals = len([s for s in signals if s['type'] == 'SELL'])
                st.metric("Sell Signals", sell_signals)
            
            with col3:
                strong_signals = len([s for s in signals if s['strength'] == 'Strong'])
                st.metric("Strong Signals", strong_signals)
            
            with col4:
                recent_signal = signals[0] if signals else None
                if recent_signal:
                    signal_color = "🟢" if recent_signal['type'] == 'BUY' else "🔴"
                    st.metric("Latest Signal", f"{signal_color} {recent_signal['type']}")
        
        else:
            st.info("No recent trading signals detected with current settings.")
        
        # Market analysis
        st.markdown("### 📈 Market Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Trend analysis
            trend_analysis = analyzer.analyze_trend(df)
            
            st.markdown("**🔍 Trend Analysis**")
            trend_color = "🟢" if "Bullish" in trend_analysis['trend'] else "🔴" if "Bearish" in trend_analysis['trend'] else "🟡"
            
            st.info(f"""
            **Current Trend:** {trend_color} {trend_analysis['trend']}  
            **Strength:** {trend_analysis['strength']}/10  
            **Analysis:** {trend_analysis['description']}
            """)
        
        with col2:
            # Support and Resistance
            sr_levels = analyzer.calculate_support_resistance(df)
            
            st.markdown("**📊 Support & Resistance**")
            
            if sr_levels['resistance']:
                st.write("**Resistance Levels:**")
                for level in sr_levels['resistance'][:3]:
                    st.write(f"• ${level:.2f}")
            
            if sr_levels['support']:
                st.write("**Support Levels:**")
                for level in sr_levels['support'][:3]:
                    st.write(f"• ${level:.2f}")
        
        # Momentum analysis
        if 'rsi' in indicators:
            st.markdown("### 🚀 Momentum Analysis")
            
            rsi_current = indicators['rsi'].iloc[-1]
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                rsi_status = "Overbought" if rsi_current > 70 else "Oversold" if rsi_current < 30 else "Neutral"
                rsi_color = "🔴" if rsi_current > 70 else "🟢" if rsi_current < 30 else "🟡"
                st.metric("RSI Status", f"{rsi_color} {rsi_status}", f"RSI: {rsi_current:.1f}")
            
            with col2:
                if 'macd' in indicators:
                    macd_current = indicators['macd']['macd'].iloc[-1]
                    signal_current = indicators['macd']['signal'].iloc[-1]
                    macd_status = "Bullish" if macd_current > signal_current else "Bearish"
                    macd_color = "🟢" if macd_current > signal_current else "🔴"
                    st.metric("MACD Status", f"{macd_color} {macd_status}")
            
            with col3:
                if 'bollinger' in indicators:
                    bb_position = indicators['bollinger']['position'].iloc[-1]
                    if bb_position > 80:
                        bb_status = "Near Upper Band"
                        bb_color = "🔴"
                    elif bb_position < 20:
                        bb_status = "Near Lower Band"
                        bb_color = "🟢"
                    else:
                        bb_status = "Middle Range"
                        bb_color = "🟡"
                    st.metric("BB Position", f"{bb_color} {bb_status}", f"{bb_position:.1f}%")
    
    except Exception as e:
        st.error(f"Error in signals analysis: {e}")

def display_indicator_summary(indicators: Dict):
    """Display summary of indicator values"""
    
    st.subheader("📊 Current Indicator Values")
    
    try:
        cols = st.columns(4)
        col_idx = 0
        
        # RSI
        if 'rsi' in indicators:
            with cols[col_idx % 4]:
                rsi_val = indicators['rsi'].iloc[-1]
                rsi_status = "Overbought" if rsi_val > 70 else "Oversold" if rsi_val < 30 else "Neutral"
                st.metric("RSI (14)", f"{rsi_val:.1f}", rsi_status)
                col_idx += 1
        
        # MACD
        if 'macd' in indicators:
            with cols[col_idx % 4]:
                macd_val = indicators['macd']['macd'].iloc[-1]
                st.metric("MACD", f"{macd_val:.3f}")
                col_idx += 1
        
        # Stochastic
        if 'stochastic' in indicators:
            with cols[col_idx % 4]:
                k_val = indicators['stochastic']['k'].iloc[-1]
                st.metric("Stochastic %K", f"{k_val:.1f}")
                col_idx += 1
        
        # Williams %R
        if 'williams_r' in indicators:
            with cols[col_idx % 4]:
                williams_val = indicators['williams_r'].iloc[-1]
                st.metric("Williams %R", f"{williams_val:.1f}")
                col_idx += 1
        
        # ATR
        if 'atr' in indicators:
            with cols[col_idx % 4]:
                atr_val = indicators['atr'].iloc[-1]
                st.metric("ATR (14)", f"${atr_val:.2f}")
                col_idx += 1
        
        # Bollinger Bands position
        if 'bollinger' in indicators:
            with cols[col_idx % 4]:
                bb_pos = indicators['bollinger']['position'].iloc[-1]
                st.metric("BB Position", f"{bb_pos:.1f}%")
                col_idx += 1
    
    except Exception as e:
        st.error(f"Error displaying indicator summary: {e}")

def main():
    """Main function for technical analysis page"""
    
    # Get current selection
    current_stock, current_period = get_current_selection()
    
    # Display header
    display_technical_header(current_stock)
    
    # Create indicator controls
    settings = create_indicator_controls()
    
    # Load data
    try:
        df = data_manager.load_stock_data(current_stock)
        
        if df.empty:
            st.error("No data available for analysis")
            return
        
        # Filter by period
        filtered_df = data_manager.filter_data_by_period(df, current_period)
        
        if len(filtered_df) < 50:
            st.warning("Limited data for selected period. Some indicators may not be accurate.")
        
        # Calculate indicators
        indicators = calculate_all_indicators(filtered_df, settings)
        
        # Display indicator summary
        display_indicator_summary(indicators)
        
        st.markdown("---")
        
        # Create and display technical chart
        st.subheader("📈 Technical Analysis Chart")
        
        chart = create_technical_chart(filtered_df, indicators, settings, current_stock)
        st.plotly_chart(chart, use_container_width=True)
        
        st.markdown("---")
        
        # Display signals and analysis
        display_signals_analysis(filtered_df, indicators, current_stock)
        
        # Educational tooltips
        st.markdown("---")
        st.markdown("### 📚 Indicator Guide")
        
        with st.expander("📖 Technical Indicators Explained", expanded=False):
            st.markdown("""
            **RSI (Relative Strength Index):** 
            - Momentum oscillator ranging from 0-100
            - Above 70: Overbought (potential sell signal)
            - Below 30: Oversold (potential buy signal)
            
            **MACD (Moving Average Convergence Divergence):**
            - Trend-following momentum indicator
            - MACD line crossing above signal line: Bullish
            - MACD line crossing below signal line: Bearish
            
            **Bollinger Bands:**
            - Volatility indicator with upper and lower bands
            - Price near upper band: Potentially overbought
            - Price near lower band: Potentially oversold
            
            **Stochastic Oscillator:**
            - Momentum indicator comparing closing price to price range
            - Above 80: Overbought
            - Below 20: Oversold
            
            **Williams %R:**
            - Momentum indicator showing overbought/oversold levels
            - Above -20: Overbought
            - Below -80: Oversold
            """)
        
        # Page footer
        st.info("""
        💡 **Technical Analysis Tips:**  
        • Use multiple indicators for confirmation
        • Consider overall market trend
        • Combine with fundamental analysis
        • Never rely on a single signal
        """)
    
    except Exception as e:
        st.error(f"Error loading technical analysis: {e}")
        st.info("Please try refreshing the page or selecting a different stock.")

if __name__ == "__main__":
    main()