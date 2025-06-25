"""
Application settings and configuration
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SAMPLE_STOCKS_DIR = DATA_DIR / "sample_stocks"
NEWS_SAMPLES_DIR = DATA_DIR / "news_samples"
FINANCIAL_STATEMENTS_DIR = DATA_DIR / "financial_statements"

# Streamlit configuration
STREAMLIT_CONFIG = {
    'page_title': 'Advanced Stock Analysis Dashboard',
    'page_icon': '📊',
    'layout': 'wide',
    'initial_sidebar_state': 'expanded'
}

# Cache configuration
CACHE_CONFIG = {
    'data_ttl': 3600,  # 1 hour for data cache
    'indicators_ttl': 1800,  # 30 minutes for indicators
    'charts_ttl': 600,  # 10 minutes for charts
    'max_entries_data': 10,
    'max_entries_indicators': 50,
    'max_entries_charts': 20,
}

# Performance settings
PERFORMANCE_CONFIG = {
    'max_chart_points': 1000,  # Maximum points to display in charts
    'chunk_size': 252,  # Trading days per year for data chunking
    'memory_limit_mb': 500,  # Memory limit before cache clearing
    'slow_operation_threshold': 2.0,  # Seconds to trigger warning
}

# Chart configuration
CHART_CONFIG = {
    'theme': 'plotly_dark',
    'height': 600,
    'colors': {
        'increasing': '#00ff88',
        'decreasing': '#ff6b6b',
        'volume': '#1f77b4',
        'sma': '#ff7f0e',
        'ema': '#2ca02c',
        'rsi': '#d62728',
        'macd': '#9467bd',
        'signal': '#8c564b',
        'bollinger_upper': '#e377c2',
        'bollinger_lower': '#7f7f7f',
    },
    'candlestick': {
        'increasing_line_color': '#00ff88',
        'decreasing_line_color': '#ff6b6b',
        'increasing_fillcolor': '#00ff88',
        'decreasing_fillcolor': '#ff6b6b',
    }
}

# Data generation settings
DATA_GENERATION_CONFIG = {
    'start_date': '2022-01-01',
    'end_date': '2024-12-31',
    'initial_prices': {
        'AAPL': 150.0,
        'MSFT': 300.0,
        'GOOGL': 2500.0,
        'TSLA': 800.0,
        'SPY': 400.0,
    },
    'volatility': {
        'AAPL': 0.25,
        'MSFT': 0.22,
        'GOOGL': 0.28,
        'TSLA': 0.45,
        'SPY': 0.15,
    },
    'drift': {
        'AAPL': 0.12,
        'MSFT': 0.10,
        'GOOGL': 0.08,
        'TSLA': 0.15,
        'SPY': 0.08,
    }
}

# ML Model configuration
ML_CONFIG = {
    'arima': {
        'max_p': 5,
        'max_d': 2,
        'max_q': 5,
        'seasonal': True,
    },
    'prophet': {
        'daily_seasonality': True,
        'weekly_seasonality': True,
        'yearly_seasonality': True,
        'interval_width': 0.95,
    },
    'linear_regression': {
        'features': ['RSI', 'MACD', 'BB_position', 'Volume_MA'],
        'lookback_window': 20,
    }
}

# News and sentiment configuration
SENTIMENT_CONFIG = {
    'news_sources': [
        'Reuters', 'Bloomberg', 'CNBC', 'MarketWatch', 'Yahoo Finance',
        'The Wall Street Journal', 'Financial Times', 'Seeking Alpha'
    ],
    'sentiment_categories': ['Very Negative', 'Negative', 'Neutral', 'Positive', 'Very Positive'],
    'articles_per_stock': 50,
    'sentiment_window': 30,  # Days for sentiment analysis
}

# Financial ratios configuration
FINANCIAL_RATIOS = {
    'profitability': [
        'ROE', 'ROA', 'Gross Margin', 'Net Margin', 'Operating Margin'
    ],
    'liquidity': [
        'Current Ratio', 'Quick Ratio', 'Cash Ratio'
    ],
    'leverage': [
        'Debt-to-Equity', 'Debt-to-Assets', 'Interest Coverage'
    ],
    'efficiency': [
        'Asset Turnover', 'Inventory Turnover', 'Receivables Turnover'
    ],
    'valuation': [
        'P/E Ratio', 'P/B Ratio', 'P/S Ratio', 'PEG Ratio', 'EV/EBITDA'
    ]
}

# Error handling configuration
ERROR_CONFIG = {
    'max_retries': 3,
    'retry_delay': 1.0,
    'fallback_enabled': True,
    'logging_level': 'INFO',
}

# Development settings
DEBUG_CONFIG = {
    'show_performance_metrics': False,
    'enable_profiling': False,
    'mock_data_enabled': True,
    'verbose_logging': False,
}

def get_config_value(section: str, key: str, default=None):
    """Get configuration value with fallback"""
    config_sections = {
        'streamlit': STREAMLIT_CONFIG,
        'cache': CACHE_CONFIG,
        'performance': PERFORMANCE_CONFIG,
        'chart': CHART_CONFIG,
        'data_generation': DATA_GENERATION_CONFIG,
        'ml': ML_CONFIG,
        'sentiment': SENTIMENT_CONFIG,
        'error': ERROR_CONFIG,
        'debug': DEBUG_CONFIG,
    }
    
    section_config = config_sections.get(section, {})
    return section_config.get(key, default)

def ensure_directories():
    """Ensure all required directories exist"""
    directories = [
        DATA_DIR,
        SAMPLE_STOCKS_DIR,
        NEWS_SAMPLES_DIR,
        FINANCIAL_STATEMENTS_DIR,
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)