"""
Advanced Multi-Page Streamlit Stock Analysis Dashboard
=====================================================

Main application entry point with navigation and global state management.
Optimized for YouTube demonstration with realistic sample data.

Author: Streamlit Tutorial Series
Created: 2024
"""

import streamlit as st
import sys
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent
sys.path.append(str(PROJECT_ROOT))

# Import configuration and utilities
from config.settings import STREAMLIT_CONFIG, ensure_directories
from config.stock_symbols import (
    get_all_symbols, get_stock_names, DEFAULT_SYMBOL, 
    TIME_PERIODS, DEFAULT_TIME_PERIOD
)
from utils.data_manager import data_manager

# Configure Streamlit page
st.set_page_config(**STREAMLIT_CONFIG)

# Ensure required directories exist
ensure_directories()

def initialize_session_state():
    """Initialize session state variables"""
    
    # Stock selection state
    if 'selected_stock' not in st.session_state:
        st.session_state.selected_stock = DEFAULT_SYMBOL
    
    # Time period state
    if 'selected_period' not in st.session_state:
        st.session_state.selected_period = DEFAULT_TIME_PERIOD
    
    # Page visit tracking for analytics
    if 'page_visits' not in st.session_state:
        st.session_state.page_visits = {
            'overview': 0,
            'technical': 0,
            'financial': 0,
            'sentiment': 0,
            'prediction': 0
        }
    
    # Cache management
    if 'cache_info' not in st.session_state:
        st.session_state.cache_info = {
            'last_cleared': None,
            'auto_clear_enabled': True
        }

def create_sidebar_navigation():
    """Create sidebar with navigation and stock selector"""
    
    st.sidebar.title("📊 Stock Analysis Dashboard")
    st.sidebar.markdown("---")
    
    # Stock Selector
    st.sidebar.subheader("🏢 Stock Selection")
    
    stock_names = get_stock_names()
    stock_options = [f"{symbol} - {name}" for symbol, name in stock_names.items()]
    
    # Find current selection index
    current_selection = f"{st.session_state.selected_stock} - {stock_names[st.session_state.selected_stock]}"
    current_index = stock_options.index(current_selection) if current_selection in stock_options else 0
    
    selected_option = st.sidebar.selectbox(
        "Choose Stock:",
        stock_options,
        index=current_index,
        key="stock_selector"
    )
    
    # Extract symbol from selection
    selected_symbol = selected_option.split(" - ")[0]
    
    # Update session state if changed
    if selected_symbol != st.session_state.selected_stock:
        st.session_state.selected_stock = selected_symbol
        st.rerun()
    
    # Time Period Selector
    st.sidebar.subheader("📅 Time Period")
    
    period_options = {period: info['label'] for period, info in TIME_PERIODS.items()}
    
    selected_period = st.sidebar.selectbox(
        "Select Period:",
        list(period_options.keys()),
        format_func=lambda x: period_options[x],
        index=list(period_options.keys()).index(st.session_state.selected_period),
        key="period_selector"
    )
    
    if selected_period != st.session_state.selected_period:
        st.session_state.selected_period = selected_period
        st.rerun()
    
    st.sidebar.markdown("---")
    
    # Navigation Links
    st.sidebar.subheader("📍 Dashboard Pages")
    
    # Overview Page
    st.sidebar.page_link(
        "pages/01_📈_overview.py",
        label="📈 Stock Overview",
        icon="📈"
    )
    
    # Technical Analysis Page
    st.sidebar.page_link(
        "pages/02_🔧_technical_analysis.py",
        label="🔧 Technical Analysis",
        icon="🔧"
    )
    
    # Financial Statements Page
    st.sidebar.page_link(
        "pages/03_💰_financial_statements.py",
        label="💰 Financial Statements",
        icon="💰"
    )
    
    # Sentiment Analysis Page
    st.sidebar.page_link(
        "pages/04_📰_sentiment_analysis.py",
        label="📰 Sentiment Analysis",
        icon="📰"
    )
    
    # Price Prediction Page
    st.sidebar.page_link(
        "pages/05_🤖_price_prediction.py",
        label="🤖 Price Prediction",
        icon="🤖"
    )
    
    st.sidebar.markdown("---")
    
    # Current Selection Summary
    st.sidebar.subheader("📋 Current Selection")
    st.sidebar.info(f"""
    **Stock:** {st.session_state.selected_stock}  
    **Period:** {TIME_PERIODS[st.session_state.selected_period]['label']}
    """)
    
    # Cache Management
    st.sidebar.subheader("🔧 Cache Management")
    
    if st.sidebar.button("🗑️ Clear Cache", help="Clear all cached data to free memory"):
        data_manager.clear_cache()
        st.session_state.cache_info['last_cleared'] = st.timestamp()
    
    # Performance Info
    cache_stats = st.cache_data.get_stats()
    if cache_stats:
        st.sidebar.caption(f"📊 Cache Entries: {len(cache_stats)}")

def display_welcome_message():
    """Display welcome message and dashboard overview"""
    
    st.title("📊 Advanced Stock Analysis Dashboard")
    st.markdown("**Professional financial analysis with interactive visualizations**")
    
    # Welcome message
    st.markdown("""
    Welcome to the **Advanced Stock Analysis Dashboard**! This comprehensive financial analysis tool 
    provides professional-grade insights into stock performance, technical indicators, financial health, 
    market sentiment, and price predictions.
    
    ### 🚀 Key Features:
    - **📈 Stock Overview**: Real-time metrics, candlestick charts, and performance analysis
    - **🔧 Technical Analysis**: 15+ technical indicators with customizable parameters
    - **💰 Financial Statements**: Comprehensive financial health analysis and ratios
    - **📰 Sentiment Analysis**: News sentiment tracking and market impact correlation
    - **🤖 Price Prediction**: Machine learning models for price forecasting
    """)
    
    # Quick stats about current stock
    current_stock = st.session_state.selected_stock
    
    try:
        metrics = data_manager.get_stock_metrics(current_stock)
        
        if metrics:
            st.subheader(f"📋 Quick Stats - {current_stock}")
            
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
    
    except Exception as e:
        st.warning("Loading stock data... Please wait a moment.")
    
    # Navigation guidance
    st.markdown("---")
    st.markdown("""
    ### 🧭 Getting Started:
    
    1. **Select a stock** from the sidebar (AAPL, MSFT, GOOGL, TSLA, SPY)
    2. **Choose a time period** for analysis
    3. **Navigate to different pages** using the sidebar links
    4. **Explore interactive charts** and customize indicators
    
    **💡 Pro Tip**: Start with the **📈 Stock Overview** page for a comprehensive view, 
    then dive deeper into specific analysis areas!
    """)
    
    # Demo information
    st.info("""
    🎬 **YouTube Demo Mode**: This dashboard uses high-quality sample data for consistent 
    demonstration purposes. All charts and analysis are based on realistic financial patterns 
    and correlations for educational value.
    """)

def display_footer():
    """Display footer with additional information"""
    
    st.markdown("---")
    
    footer_col1, footer_col2, footer_col3 = st.columns(3)
    
    with footer_col1:
        st.markdown("""
        ### 📚 Learning Resources
        - [Streamlit Documentation](https://docs.streamlit.io)
        - [Plotly Financial Charts](https://plotly.com/python/candlestick-charts/)
        - [Technical Analysis Guide](https://www.investopedia.com/technical-analysis-4689657)
        """)
    
    with footer_col2:
        st.markdown("""
        ### 🔗 Quick Navigation
        - [📈 Overview](pages/01_📈_overview.py)
        - [🔧 Technical Analysis](pages/02_🔧_technical_analysis.py)
        - [💰 Financial Statements](pages/03_💰_financial_statements.py)
        """)
    
    with footer_col3:
        st.markdown("""
        ### ⚡ Performance
        - Multi-level caching for speed
        - Optimized chart rendering
        - Sample data for consistency
        """)
    
    # Credits
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p><strong>Advanced Stock Analysis Dashboard</strong></p>
        <p>Built with ❤️ using Streamlit, Plotly, and advanced financial analysis libraries</p>
        <p><em>Created for educational purposes and YouTube demonstration</em></p>
    </div>
    """, unsafe_allow_html=True)

def main():
    """Main application function"""
    
    # Initialize session state
    initialize_session_state()
    
    # Create sidebar navigation
    create_sidebar_navigation()
    
    # Display main content
    display_welcome_message()
    
    # Display footer
    display_footer()
    
    # Track page visit
    st.session_state.page_visits['overview'] += 1

if __name__ == "__main__":
    main()