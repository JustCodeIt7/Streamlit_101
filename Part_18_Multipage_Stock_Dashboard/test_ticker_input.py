"""
Test script for ticker input functionality
Run this to test the new features before using the full dashboard
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.append(str(PROJECT_ROOT))

from utils.real_data_fetcher import real_data_fetcher
from utils.data_manager import data_manager
from config.stock_symbols import validate_ticker_format

def test_ticker_validation():
    """Test ticker format validation"""
    print("Testing ticker format validation...")
    
    test_cases = [
        ("AAPL", True),
        ("MSFT", True),
        ("NVDA", True),
        ("GOOGL", True),
        ("", False),
        ("TOOLONG", False),
        ("123", False),
        ("AA$", False),
        ("aa", True),  # Should be converted to uppercase
    ]
    
    for ticker, expected in test_cases:
        result = validate_ticker_format(ticker)
        status = "✅" if result == expected else "❌"
        print(f"{status} {ticker:8} -> {result:5} (expected: {expected})")

def test_real_data_fetching():
    """Test real data fetching for a known ticker"""
    print("\nTesting real data fetching...")
    
    test_ticker = "AAPL"
    
    try:
        # Test validation
        is_valid, message = real_data_fetcher.validate_symbol(test_ticker)
        print(f"Validation for {test_ticker}: {is_valid} - {message}")
        
        if is_valid:
            # Test company info
            print("Fetching company info...")
            company_info = real_data_fetcher.fetch_company_info(test_ticker)
            print(f"Company: {company_info.get('name', 'Unknown')}")
            print(f"Sector: {company_info.get('sector', 'Unknown')}")
            
            # Test stock data (small sample)
            print("Fetching stock data...")
            df = real_data_fetcher.fetch_stock_data(test_ticker, period="5d")
            print(f"Data shape: {df.shape}")
            print(f"Latest close: ${df.iloc[-1]['close']:.2f}")
            
            # Test quote
            print("Fetching current quote...")
            quote = real_data_fetcher.fetch_current_quote(test_ticker)
            print(f"Current price: ${quote['current_price']:.2f}")
            print(f"Change: {quote['price_change']:+.2f} ({quote['price_change_pct']:+.2f}%)")
            
        print("✅ Real data fetching test completed successfully")
        
    except Exception as e:
        print(f"❌ Error during real data test: {e}")

def test_data_manager_integration():
    """Test the integrated data manager functionality"""
    print("\nTesting data manager integration...")
    
    test_ticker = "AAPL"
    
    try:
        # Test validation through data manager
        is_valid, message = data_manager.validate_stock_symbol(test_ticker)
        print(f"Data manager validation: {is_valid} - {message}")
        
        # Test loading real data
        print("Loading real data through data manager...")
        df = data_manager.load_stock_data(test_ticker, use_real_data=True)
        print(f"Loaded data shape: {df.shape}")
        
        # Test metrics
        print("Getting stock metrics...")
        metrics = data_manager.get_stock_metrics(test_ticker, use_real_data=True)
        print(f"Current price: ${metrics['current_price']:.2f}")
        
        # Test company info
        print("Getting company info...")
        company_info = data_manager.get_company_info(test_ticker, use_real_data=True)
        print(f"Company: {company_info['name']}")
        
        print("✅ Data manager integration test completed successfully")
        
    except Exception as e:
        print(f"❌ Error during data manager test: {e}")

if __name__ == "__main__":
    print("🧪 Testing Ticker Input Functionality")
    print("=" * 50)
    
    test_ticker_validation()
    test_real_data_fetching()
    test_data_manager_integration()
    
    print("\n" + "=" * 50)
    print("🎉 Testing completed!")
    print("\nYou can now run the dashboard with:")
    print("streamlit run 18_app.py")