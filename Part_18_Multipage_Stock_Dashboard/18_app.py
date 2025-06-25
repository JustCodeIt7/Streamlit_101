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
    get_all_symbols,
    get_stock_names,
    DEFAULT_SYMBOL,
    TIME_PERIODS,
    DEFAULT_TIME_PERIOD,
    is_predefined_stock,
    get_stock_display_name,
    validate_ticker_format,
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
    
    # Custom ticker input state
    if 'custom_ticker' not in st.session_state:
        st.session_state.custom_ticker = ""
    
    # Data source preference
    if 'use_real_data' not in st.session_state:
        st.session_state.use_real_data = False
    
    # Stock selection mode (predefined vs custom)
    if 'stock_selection_mode' not in st.session_state:
        st.session_state.stock_selection_mode = "predefined"
    
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
    
    # Stock Selection Mode
    st.sidebar.subheader("🏢 Stock Selection")
    
    # Selection mode toggle
    selection_mode = st.sidebar.radio(
        "Choose Selection Mode:",
        ["Predefined Stocks", "Custom Ticker"],
        index=0 if st.session_state.stock_selection_mode == "predefined" else 1,
        key="selection_mode_radio"
    )
    
    # Update session state for selection mode
    new_mode = "predefined" if selection_mode == "Predefined Stocks" else "custom"
    if new_mode != st.session_state.stock_selection_mode:
        st.session_state.stock_selection_mode = new_mode
        st.rerun()
    
    # Handle stock selection based on mode
    if st.session_state.stock_selection_mode == "predefined":
        _handle_predefined_stock_selection()
    else:
        _handle_custom_ticker_input()
    
    # Data Source Selection
    st.sidebar.markdown("---")
    st.sidebar.subheader("📡 Data Source")
    
    use_real_data = st.sidebar.checkbox(
        "Use Real-Time Data",
        value=st.session_state.use_real_data,
        help="Fetch live data from Yahoo Finance (may be slower)",
        key="real_data_checkbox"
    )
    
    if use_real_data != st.session_state.use_real_data:
        st.session_state.use_real_data = use_real_data
        st.rerun()
    
    # Show data source info
    if st.session_state.use_real_data:
        st.sidebar.info("📈 Using real-time data from Yahoo Finance")
    else:
        if is_predefined_stock(st.session_state.selected_stock):
            st.sidebar.info("📊 Using high-quality sample data")
        else:
            st.sidebar.warning("⚠️ Custom stocks require real-time data")

def _handle_predefined_stock_selection():
    """Handle predefined stock selection"""
    stock_names = get_stock_names()
    stock_options = [f"{symbol} - {name}" for symbol, name in stock_names.items()]
    
    # Find current selection index
    if is_predefined_stock(st.session_state.selected_stock):
        current_selection = f"{st.session_state.selected_stock} - {stock_names[st.session_state.selected_stock]}"
        current_index = stock_options.index(current_selection) if current_selection in stock_options else 0
    else:
        current_index = 0
        st.session_state.selected_stock = DEFAULT_SYMBOL
    
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

def _handle_custom_ticker_input():
    """Handle custom ticker input"""
    
    # Custom ticker input
    custom_ticker = st.sidebar.text_input(
        "Enter Stock Ticker:",
        value=st.session_state.custom_ticker,
        placeholder="e.g., NVDA, AMD, META",
        help="Enter any valid stock ticker symbol",
        key="custom_ticker_input"
    ).upper().strip()
    
    # Validate and load button
    col1, col2 = st.sidebar.columns([3, 1])
    
    with col1:
        load_button = st.button("Load Stock", type="primary", use_container_width=True)
    
    with col2:
        if st.button("ℹ️", help="Ticker format: 1-5 letters only"):
            st.sidebar.info("""
            **Ticker Format:**
            - 1-5 letters only
            - Examples: AAPL, MSFT, GOOGL
            - No spaces or special characters
            """)
    
    # Handle ticker loading
    if load_button and custom_ticker:
        if validate_ticker_format(custom_ticker):
            with st.sidebar.spinner(f"Validating {custom_ticker}..."):
                is_valid, message = data_manager.validate_stock_symbol(custom_ticker)
                
                if is_valid:
                    st.session_state.custom_ticker = custom_ticker
                    st.session_state.selected_stock = custom_ticker
                    st.session_state.use_real_data = True  # Force real data for custom stocks
                    st.sidebar.success(f"✅ Loaded {custom_ticker}")
                    st.rerun()
                else:
                    st.sidebar.error(f"❌ {message}")
        else:
            st.sidebar.error("❌ Invalid ticker format. Use 1-5 letters only.")
    
    # Show current custom stock if loaded
    if st.session_state.stock_selection_mode == "custom" and st.session_state.selected_stock:
        if not is_predefined_stock(st.session_state.selected_stock):
            st.sidebar.success(f"📈 Current: {st.session_state.selected_stock}")
            
            # Add remove button
            if st.sidebar.button("🗑️ Clear Custom Stock"):
                st.session_state.selected_stock = DEFAULT_SYMBOL
                st.session_state.custom_ticker = ""
                st.session_state.stock_selection_mode = "predefined"
                st.session_state.use_real_data = False
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
        "pages/02_📊_technical_analysis.py",
        label="📊 Technical Analysis",
        icon="📊"
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
    
    # Get display name for current stock
    if is_predefined_stock(st.session_state.selected_stock):
        stock_display = f"{st.session_state.selected_stock} - {get_stock_names()[st.session_state.selected_stock]}"
    else:
        stock_display = get_stock_display_name(st.session_state.selected_stock)
    
    data_source = "Real-Time" if st.session_state.use_real_data else "Sample Data"
    
    st.sidebar.info(f"""
    **Stock:** {stock_display}
    **Period:** {TIME_PERIODS[st.session_state.selected_period]['label']}
    **Data:** {data_source}
    """)
    
    # Cache Management
    st.sidebar.subheader("🔧 Cache Management")
    
    if st.sidebar.button("🗑️ Clear Cache", help="Clear all cached data to free memory"):
        data_manager.clear_cache()
    


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
        # Get stock metrics with appropriate data source
        metrics = data_manager.get_stock_metrics(current_stock, use_real_data=st.session_state.use_real_data)
        
        if metrics:
            # Display company name
            company_info = data_manager.get_company_info(current_stock, use_real_data=st.session_state.use_real_data)
            company_name = company_info.get('name', current_stock)
            
            st.subheader(f"📋 Quick Stats - {company_name}")
            
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
            
            # Show additional info for custom stocks
            if not is_predefined_stock(current_stock):
                st.info(f"""
                **Company:** {company_name}
                **Sector:** {company_info.get('sector', 'Unknown')}
                **Exchange:** {company_info.get('exchange', 'Unknown')}
                **Last Updated:** {metrics.get('last_updated', 'Unknown')}
                """)
    
    except Exception as e:
        if not is_predefined_stock(current_stock):
            st.error(f"Unable to load data for {current_stock}. Please check the ticker symbol and try again.")
        else:
            st.warning("Loading stock data... Please wait a moment.")
    
    # Navigation guidance
    st.markdown("---")
    st.markdown("""
    ### 🧭 Getting Started:
    
    1. **Select a stock** from the sidebar (AAPL, MSFT, GOOGL, TSLA, SPY)
    2. **Choose a time period** for analysis
    3. **Navigate to different pages** using the sidebar links
    4. **Explore interactive charts** and customize indicators
    

    """)


def main():
    """Main application function"""
    
    # Initialize session state
    initialize_session_state()
    
    # Create sidebar navigation
    create_sidebar_navigation()
    
    # Display main content
    display_welcome_message()

    # Track page visit
    st.session_state.page_visits['overview'] += 1

if __name__ == "__main__":
    main()