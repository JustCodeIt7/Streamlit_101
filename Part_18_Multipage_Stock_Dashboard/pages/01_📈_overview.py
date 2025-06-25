"""
Stock Overview Page
==================

Comprehensive stock overview with key metrics, candlestick charts, and performance analysis.
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

# Import configuration and utilities
from config.settings import STREAMLIT_CONFIG
from config.stock_symbols import get_stock_info, TIME_PERIODS, is_predefined_stock
from utils.data_manager import data_manager
from utils.chart_components import chart_components

# Configure page
st.set_page_config(**STREAMLIT_CONFIG)

def get_current_selection():
    """Get current stock and period selection from session state"""
    current_stock = st.session_state.get('selected_stock', 'AAPL')
    current_period = st.session_state.get('selected_period', '1Y')
    use_real_data = st.session_state.get('use_real_data', False)
    return current_stock, current_period, use_real_data

def display_stock_header(symbol: str, use_real_data: bool = False):
    """Display stock header with company information"""
    
    # Get company info using the data manager (handles both predefined and custom stocks)
    try:
        company_info = data_manager.get_company_info(symbol, use_real_data=use_real_data)
        
        company_name = company_info.get('name', symbol)
        st.title(f"📈 {company_name} ({symbol})")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Show data source indicator
            data_source = "Real-Time Data" if use_real_data or not is_predefined_stock(symbol) else "Sample Data"
            
            st.markdown(f"""
            **Sector:** {company_info.get('sector', 'N/A')}
            **Industry:** {company_info.get('industry', 'N/A')}
            **Exchange:** {company_info.get('exchange', 'N/A')}
            **Data Source:** {data_source}
            """)
            
            # Show description if available
            description = company_info.get('description', '')
            if description and description != 'N/A':
                # Truncate long descriptions
                if len(description) > 200:
                    description = description[:200] + "..."
                st.caption(description)
        
        with col2:
            market_cap = company_info.get('market_cap', 0)
            if market_cap > 0:
                if market_cap >= 1e12:
                    cap_str = f"${market_cap/1e12:.1f}T"
                elif market_cap >= 1e9:
                    cap_str = f"${market_cap/1e9:.1f}B"
                else:
                    cap_str = f"${market_cap/1e6:.1f}M"
                
                st.metric("Market Cap", cap_str)
            
            # Show additional metrics for real data
            if use_real_data or not is_predefined_stock(symbol):
                pe_ratio = company_info.get('pe_ratio')
                if pe_ratio:
                    st.metric("P/E Ratio", f"{pe_ratio:.2f}")
                
                dividend_yield = company_info.get('dividend_yield')
                if dividend_yield:
                    st.metric("Dividend Yield", f"{dividend_yield*100:.2f}%")
    
    except Exception as e:
        st.title(f"📈 Stock Overview - {symbol}")
        st.error(f"Unable to load company information: {e}")

def display_key_metrics(symbol: str, use_real_data: bool = False):
    """Display key stock metrics in a metrics row"""
    
    try:
        metrics = data_manager.get_stock_metrics(symbol, use_real_data=use_real_data)
        print(metrics)
        if not metrics:
            st.warning("Unable to load stock metrics")
            return
        
        st.subheader("📊 Key Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Current Price",
                f"${metrics['current_price']:.2f}",
                f"{metrics['price_change']:+.2f} ({metrics['price_change_pct']:+.2f}%)"
            )
        
        with col2:
            st.metric(
                "Volume",
                f"{metrics['volume']:,}",
                f"Avg: {metrics['avg_volume']:,.0f}"
            )
        
        with col3:
            st.metric(
                "52W High",
                f"${metrics['high_52w']:.2f}"
            )
        
        with col4:
            st.metric(
                "52W Low",
                f"${metrics['low_52w']:.2f}"
            )
        
        # Additional metrics row
        col5, col6, col7, col8 = st.columns(4)
        
        with col5:
            # Calculate day range
            df = data_manager.load_stock_data(symbol, use_real_data=use_real_data)
            if not df.empty:
                latest = df.iloc[-1]
                day_range = f"${latest['low']:.2f} - ${latest['high']:.2f}"
                st.metric("Day Range", day_range)
        
        with col6:
            # Market cap (if available)
            market_cap = metrics.get('market_cap', 0)
            if market_cap > 0:
                if market_cap >= 1e12:
                    cap_str = f"${market_cap/1e12:.1f}T"
                elif market_cap >= 1e9:
                    cap_str = f"${market_cap/1e9:.1f}B"
                else:
                    cap_str = f"${market_cap/1e6:.1f}M"
                st.metric("Market Cap", cap_str)
        
        with col7:
            # Calculate volatility (20-day)
            if not df.empty and len(df) > 20:
                returns = df['close'].pct_change().dropna()
                volatility = returns.tail(20).std() * np.sqrt(252) * 100  # Annualized
                st.metric("Volatility (20d)", f"{volatility:.1f}%")
        
        with col8:
            # Average True Range (ATR) - if available
            if not df.empty and len(df) > 14:
                high_low = df['high'] - df['low']
                high_close = abs(df['high'] - df['close'].shift())
                low_close = abs(df['low'] - df['close'].shift())
                true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
                atr = true_range.rolling(14).mean().iloc[-1]
                st.metric("ATR (14)", f"${atr:.2f}")
        
        # Show last updated time for real data
        if use_real_data or not is_predefined_stock(symbol):
            last_updated = metrics.get('last_updated', 'Unknown')
            st.caption(f"📅 Last updated: {last_updated}")
        
    except Exception as e:
        st.error(f"Error loading metrics: {e}")
        if not is_predefined_stock(symbol):
            st.info("💡 This might be due to an invalid ticker symbol or network issues. Please verify the symbol and try again.")

def display_price_chart(symbol: str, period: str, use_real_data: bool = False):
    """Display interactive candlestick chart with volume"""
    
    st.subheader("📈 Price Chart")
    
    try:
        # Load data
        df = data_manager.load_stock_data(symbol, use_real_data=use_real_data)
        
        if df.empty:
            st.error("No data available for chart")
            if not is_predefined_stock(symbol):
                st.info("💡 Please check if the ticker symbol is correct and try again.")
            return
        
        # Filter by period
        filtered_df = data_manager.filter_data_by_period(df, period)
        
        if filtered_df.empty:
            st.warning("No data available for selected period")
            return
        
        # Chart options
        chart_col1, chart_col2 = st.columns([3, 1])
        
        with chart_col2:
            show_volume = st.checkbox("Show Volume", value=True)
            show_indicators = st.checkbox("Show Moving Averages", value=True)
            
            # Indicator selection
            indicators = []
            if show_indicators:
                if st.checkbox("SMA 20", value=True):
                    indicators.append('sma_20')
                if st.checkbox("SMA 50", value=True):
                    indicators.append('sma_50')
                if st.checkbox("EMA 12"):
                    indicators.append('ema_12')
        
        with chart_col1:
            # Create chart
            data_source_label = "Real-Time" if use_real_data or not is_predefined_stock(symbol) else "Sample"
            chart_title = f"{symbol} - {TIME_PERIODS[period]['label']} ({data_source_label} Data)"
            
            fig = chart_components.create_candlestick_chart(
                df=filtered_df,
                title=chart_title,
                show_volume=show_volume,
                indicators=indicators if show_indicators else None
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Chart insights
        if len(filtered_df) > 1:
            st.markdown("### 📊 Chart Insights")
            
            insight_col1, insight_col2 = st.columns(2)
            
            with insight_col1:
                # Price movement analysis
                start_price = filtered_df.iloc[0]['close']
                end_price = filtered_df.iloc[-1]['close']
                total_return = ((end_price - start_price) / start_price) * 100
                
                st.info(f"""
                **Period Performance**
                Total Return: {total_return:+.2f}%
                Start Price: ${start_price:.2f}
                End Price: ${end_price:.2f}
                """)
            
            with insight_col2:
                # Volatility analysis
                returns = filtered_df['close'].pct_change().dropna()
                volatility = returns.std() * np.sqrt(252) * 100
                max_drawdown = ((filtered_df['close'] / filtered_df['close'].expanding().max()) - 1).min() * 100
                
                st.info(f"""
                **Risk Metrics**
                Volatility: {volatility:.1f}%
                Max Drawdown: {max_drawdown:.1f}%
                Sharpe Ratio: {(total_return / volatility * np.sqrt(252/len(filtered_df))):.2f}
                """)
    
    except Exception as e:
        st.error(f"Error creating chart: {e}")
        if not is_predefined_stock(symbol):
            st.info("💡 This might be due to an invalid ticker symbol or network issues.")

def display_performance_summary(symbol: str, use_real_data: bool = False):
    """Display performance summary for different time periods"""
    
    st.subheader("📈 Performance Summary")
    
    try:
        df = data_manager.load_stock_data(symbol, use_real_data=use_real_data)
        print(df)
        
        if df.empty:
            st.warning("No data available for performance analysis")
            return
        
        # Calculate returns for different periods
        periods_data = []
        
        for period_key, period_info in TIME_PERIODS.items():
            period_df = data_manager.filter_data_by_period(df, period_key)
            
            if len(period_df) > 1:
                start_price = period_df.iloc[0]['close']
                end_price = period_df.iloc[-1]['close']
                return_pct = ((end_price - start_price) / start_price) * 100
                
                # Calculate volatility for the period
                returns = period_df['close'].pct_change().dropna()
                volatility = returns.std() * np.sqrt(252) * 100 if len(returns) > 1 else 0
                
                periods_data.append({
                    'Period': period_info['label'],
                    'Return (%)': f"{return_pct:+.2f}%",
                    'Volatility (%)': f"{volatility:.1f}%",
                    'Start Price': f"${start_price:.2f}",
                    'End Price': f"${end_price:.2f}"
                })
        
        if periods_data:
            performance_df = pd.DataFrame(periods_data)
            st.dataframe(performance_df, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error calculating performance: {e}")

def display_recent_news(symbol: str):
    """Display recent news and sentiment for the stock"""
    
    st.subheader("📰 Recent News & Sentiment")
    
    try:
        news_data = data_manager.load_news_data(symbol)
        
        if not news_data:
            st.info("No recent news available")
            return
        
        # Show top 5 recent articles
        recent_news = news_data[:5]
        
        for article in recent_news:
            with st.expander(f"📄 {article['headline']}", expanded=False):
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.write(f"**Source:** {article['source']}")
                    st.write(f"**Date:** {article['date']}")
                    if 'url' in article:
                        st.write(f"[Read more]({article['url']})")
                
                with col2:
                    sentiment = article.get('sentiment_score', 0)
                    confidence = article.get('confidence', 0)
                    
                    # Color code sentiment
                    if sentiment > 0.2:
                        sentiment_color = "🟢"
                        sentiment_text = "Positive"
                    elif sentiment < -0.2:
                        sentiment_color = "🔴"
                        sentiment_text = "Negative"
                    else:
                        sentiment_color = "🟡"
                        sentiment_text = "Neutral"
                    
                    st.metric(
                        "Sentiment",
                        f"{sentiment_color} {sentiment_text}",
                        f"Score: {sentiment:.2f}"
                    )
                    st.caption(f"Confidence: {confidence:.1%}")
        
        # Overall sentiment summary
        st.markdown("### 📊 Sentiment Summary")
        
        sentiments = [article.get('sentiment_score', 0) for article in news_data]
        avg_sentiment = np.mean(sentiments)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            positive_count = sum(1 for s in sentiments if s > 0.2)
            st.metric("Positive Articles", positive_count)
        
        with col2:
            negative_count = sum(1 for s in sentiments if s < -0.2)
            st.metric("Negative Articles", negative_count)
        
        with col3:
            st.metric(
                "Average Sentiment",
                f"{avg_sentiment:.2f}",
                "Overall sentiment score"
            )
    
    except Exception as e:
        st.error(f"Error loading news: {e}")

def main():
    """Main function for the overview page"""
    
    # Get current selection
    current_stock, current_period, use_real_data = get_current_selection()
    
    # Display stock header
    display_stock_header(current_stock, use_real_data)
    
    # Display key metrics
    display_key_metrics(current_stock, use_real_data)
    
    st.markdown("---")
    
    # Display price chart
    display_price_chart(current_stock, current_period, use_real_data)
    
    st.markdown("---")
    
    # Two column layout for additional content
    left_col, right_col = st.columns([3, 2])
    
    with left_col:
        display_performance_summary(current_stock, use_real_data)
    
    with right_col:
        # Only show news for predefined stocks (sample news data)
        if is_predefined_stock(current_stock) and not use_real_data:
            display_recent_news(current_stock)
        else:
            st.subheader("📰 News & Sentiment")
            st.info("""
            📈 **Real-Time News Coming Soon**
            
            News sentiment analysis is currently available for predefined stocks with sample data.
            Real-time news integration for custom stocks is planned for future updates.
            """)
    
    # Page footer
    st.markdown("---")
    
    # Updated navigation tips
    data_source = "real-time" if use_real_data or not is_predefined_stock(current_stock) else "sample"
    
    st.info(f"""
    💡 **Navigation Tips:**
    • Use the sidebar to switch stocks, time periods, and data sources
    • Currently using {data_source} data for {current_stock}
    • Charts are interactive - zoom, pan, and hover for details
    • Explore other pages for detailed technical and fundamental analysis
    """)
    
    # Track page visit
    if 'page_visits' in st.session_state:
        st.session_state.page_visits['overview'] = st.session_state.page_visits.get('overview', 0) + 1

if __name__ == "__main__":
    main()