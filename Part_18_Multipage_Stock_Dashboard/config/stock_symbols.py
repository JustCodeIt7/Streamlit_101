"""
Stock symbols configuration and metadata for the dashboard
"""

# Stock symbols and their metadata
STOCK_SYMBOLS = {
    'AAPL': {
        'name': 'Apple Inc.',
        'sector': 'Technology',
        'industry': 'Consumer Electronics',
        'market_cap': 2800000000000,  # $2.8T
        'description': 'Technology company known for iPhone, iPad, Mac, and services'
    },
    'MSFT': {
        'name': 'Microsoft Corporation',
        'sector': 'Technology', 
        'industry': 'Software',
        'market_cap': 2400000000000,  # $2.4T
        'description': 'Cloud computing, productivity software, and enterprise solutions'
    },
    'GOOGL': {
        'name': 'Alphabet Inc.',
        'sector': 'Technology',
        'industry': 'Internet Services',
        'market_cap': 1600000000000,  # $1.6T
        'description': 'Search engine, cloud computing, and digital advertising'
    },
    'TSLA': {
        'name': 'Tesla, Inc.',
        'sector': 'Consumer Discretionary',
        'industry': 'Electric Vehicles',
        'market_cap': 800000000000,  # $800B
        'description': 'Electric vehicles, energy storage, and solar panels'
    },
    'SPY': {
        'name': 'SPDR S&P 500 ETF',
        'sector': 'Financial Services',
        'industry': 'Exchange Traded Fund',
        'market_cap': 400000000000,  # $400B
        'description': 'ETF tracking the S&P 500 index'
    }
}

# Default stock symbol for initial load
DEFAULT_SYMBOL = 'AAPL'

# Available time periods for charts
TIME_PERIODS = {
    '1D': {'days': 1, 'label': '1 Day'},
    '1W': {'days': 7, 'label': '1 Week'},
    '1M': {'days': 30, 'label': '1 Month'},
    '3M': {'days': 90, 'label': '3 Months'},
    '6M': {'days': 180, 'label': '6 Months'},
    '1Y': {'days': 252, 'label': '1 Year'},
    '2Y': {'days': 504, 'label': '2 Years'}
}

# Default time period
DEFAULT_TIME_PERIOD = '1Y'

# Technical indicators configuration
TECHNICAL_INDICATORS = {
    'trend': {
        'SMA_20': {'name': 'Simple Moving Average (20)', 'params': {'window': 20}},
        'SMA_50': {'name': 'Simple Moving Average (50)', 'params': {'window': 50}},
        'SMA_200': {'name': 'Simple Moving Average (200)', 'params': {'window': 200}},
        'EMA_12': {'name': 'Exponential Moving Average (12)', 'params': {'span': 12}},
        'EMA_26': {'name': 'Exponential Moving Average (26)', 'params': {'span': 26}},
    },
    'momentum': {
        'RSI': {'name': 'Relative Strength Index', 'params': {'window': 14}},
        'MACD': {'name': 'MACD (12,26,9)', 'params': {'fast': 12, 'slow': 26, 'signal': 9}},
        'Stochastic': {'name': 'Stochastic Oscillator', 'params': {'k_window': 14, 'd_window': 3}},
    },
    'volatility': {
        'BB': {'name': 'Bollinger Bands (20,2)', 'params': {'window': 20, 'std': 2}},
        'ATR': {'name': 'Average True Range', 'params': {'window': 14}},
    },
    'volume': {
        'OBV': {'name': 'On-Balance Volume', 'params': {}},
        'VWAP': {'name': 'Volume Weighted Average Price', 'params': {}},
    }
}

# Default selected indicators
DEFAULT_INDICATORS = ['SMA_20', 'RSI', 'MACD', 'BB']

def get_stock_info(symbol: str) -> dict:
    """Get stock information by symbol"""
    return STOCK_SYMBOLS.get(symbol, {})

def get_all_symbols() -> list:
    """Get all available stock symbols"""
    return list(STOCK_SYMBOLS.keys())

def get_stock_names() -> dict:
    """Get mapping of symbols to company names"""
    return {symbol: info['name'] for symbol, info in STOCK_SYMBOLS.items()}

def is_predefined_stock(symbol: str) -> bool:
    """Check if symbol is one of the predefined stocks"""
    return symbol.upper() in STOCK_SYMBOLS

def get_stock_display_name(symbol: str) -> str:
    """Get display name for a stock symbol"""
    if is_predefined_stock(symbol):
        return STOCK_SYMBOLS[symbol.upper()]['name']
    return f"{symbol.upper()} - Custom Stock"

def validate_ticker_format(ticker: str) -> bool:
    """Validate ticker format (basic validation)"""
    if not ticker:
        return False
    
    # Remove whitespace and convert to uppercase
    ticker = ticker.strip().upper()
    
    # Basic format validation (1-5 characters, letters only)
    if len(ticker) < 1 or len(ticker) > 5:
        return False
    
    if not ticker.isalpha():
        return False
        
    return True