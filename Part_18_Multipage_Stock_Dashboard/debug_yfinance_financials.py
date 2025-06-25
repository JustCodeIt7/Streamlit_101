"""
Debug script to investigate yfinance financial data structure
Run this to understand the exact format of financial statements from yfinance
"""

import yfinance as yf
import pandas as pd
import sys
from pathlib import Path

def debug_yfinance_financials(symbol: str = "AAPL"):
    """
    Debug yfinance financial statements structure
    Based on yfinance documentation: https://ranaroussi.github.io/yfinance/reference/yfinance.financials.html
    """
    print(f"🔍 Debugging yfinance financial data structure for {symbol}")
    print("=" * 60)
    
    try:
        # Create ticker object
        ticker = yf.Ticker(symbol)
        
        print(f"\n📊 1. INCOME STATEMENT")
        print("-" * 30)
        
        # Test different income statement methods
        try:
            # Annual income statement
            income_stmt = ticker.income_stmt
            print(f"✅ Annual income_stmt shape: {income_stmt.shape}")
            print(f"   Columns (dates): {list(income_stmt.columns)}")
            print(f"   Index (financial items): {list(income_stmt.index[:10])}...")  # First 10 items
            
            # Check if 'cost_of_revenue' exists
            if 'Cost Of Revenue' in income_stmt.index:
                print("✅ 'Cost Of Revenue' found in income statement")
            else:
                print("❌ 'Cost Of Revenue' NOT found in income statement")
                print(f"   Available revenue-related items: {[item for item in income_stmt.index if 'cost' in item.lower() or 'revenue' in item.lower()]}")
            
        except Exception as e:
            print(f"❌ Error getting annual income statement: {e}")
        
        try:
            # Quarterly income statement
            quarterly_income = ticker.quarterly_income_stmt
            print(f"✅ Quarterly income_stmt shape: {quarterly_income.shape}")
            print(f"   Columns (dates): {list(quarterly_income.columns)}")
            
        except Exception as e:
            print(f"❌ Error getting quarterly income statement: {e}")
        
        try:
            # TTM income statement
            ttm_income = ticker.ttm_income_stmt
            print(f"✅ TTM income_stmt shape: {ttm_income.shape}")
            
        except Exception as e:
            print(f"❌ Error getting TTM income statement: {e}")
        
        print(f"\n⚖️ 2. BALANCE SHEET")
        print("-" * 30)
        
        try:
            balance_sheet = ticker.balance_sheet
            print(f"✅ Balance sheet shape: {balance_sheet.shape}")
            print(f"   Columns (dates): {list(balance_sheet.columns)}")
            print(f"   Index (balance sheet items): {list(balance_sheet.index[:10])}...")
            
        except Exception as e:
            print(f"❌ Error getting balance sheet: {e}")
        
        print(f"\n💵 3. CASH FLOW")
        print("-" * 30)
        
        try:
            cashflow = ticker.cashflow
            print(f"✅ Cash flow shape: {cashflow.shape}")
            print(f"   Columns (dates): {list(cashflow.columns)}")
            print(f"   Index (cash flow items): {list(cashflow.index[:10])}...")
            
        except Exception as e:
            print(f"❌ Error getting cash flow: {e}")
        
        print(f"\n🔍 4. DATA STRUCTURE ANALYSIS")
        print("-" * 30)
        
        # Analyze the structure more deeply
        if 'income_stmt' in locals():
            print(f"Income statement data type: {type(income_stmt)}")
            print(f"Index name: {income_stmt.index.name}")
            print(f"Columns name: {income_stmt.columns.name}")
            
            # Check for period-related information
            print(f"Column data types: {income_stmt.dtypes}")
            
            # Sample data from first column
            first_col = income_stmt.columns[0]
            print(f"\nSample data from {first_col}:")
            sample_data = income_stmt[first_col].dropna().head()
            for idx, val in sample_data.items():
                print(f"   {idx}: {val}")
        
        print(f"\n📋 5. EXPECTED VS ACTUAL STRUCTURE")
        print("-" * 30)
        
        print("Expected structure (from sample data):")
        print("- DataFrame with 'period' column")
        print("- 'cost_of_revenue' field")
        print("- Row-based structure with period as column")
        
        print("\nActual yfinance structure:")
        print("- DataFrame with dates as columns")
        print("- Financial items as index")
        print("- Column-based structure with dates as columns")
        print("- Financial items like 'Cost Of Revenue' as row indices")
        
        return ticker
        
    except Exception as e:
        print(f"❌ Critical error: {e}")
        return None

def test_data_conversion(symbol: str = "AAPL"):
    """Test converting yfinance data to expected format"""
    print(f"\n🔄 TESTING DATA CONVERSION for {symbol}")
    print("=" * 60)
    
    try:
        ticker = yf.Ticker(symbol)
        income_stmt = ticker.income_stmt
        
        if income_stmt.empty:
            print("❌ No income statement data available")
            return
        
        # Convert from yfinance format to expected format
        print("Converting yfinance format to expected format...")
        
        # Transpose so dates become rows and financial items become columns
        converted_df = income_stmt.T
        converted_df.reset_index(inplace=True)
        converted_df.rename(columns={'index': 'period'}, inplace=True)
        
        print(f"✅ Converted DataFrame shape: {converted_df.shape}")
        print(f"   Columns: {list(converted_df.columns[:10])}...")  # First 10 columns
        
        # Check for cost of revenue equivalent
        cost_revenue_cols = [col for col in converted_df.columns if 'cost' in col.lower() and 'revenue' in col.lower()]
        print(f"   Cost of revenue columns: {cost_revenue_cols}")
        
        # Rename columns to match expected format
        column_mapping = {
            'Total Revenue': 'revenue',
            'Cost Of Revenue': 'cost_of_revenue',
            'Gross Profit': 'gross_profit',
            'Operating Income': 'operating_income',
            'Net Income': 'net_income'
        }
        
        # Apply mapping
        for old_name, new_name in column_mapping.items():
            if old_name in converted_df.columns:
                converted_df.rename(columns={old_name: new_name}, inplace=True)
                print(f"   ✅ Mapped '{old_name}' → '{new_name}'")
            else:
                print(f"   ❌ Column '{old_name}' not found")
        
        print(f"\nFinal columns after mapping: {list(converted_df.columns[:10])}...")
        
        # Show sample data
        print(f"\nSample converted data:")
        if 'period' in converted_df.columns:
            for i, row in converted_df.head(3).iterrows():
                print(f"   Period: {row['period']}")
                if 'revenue' in converted_df.columns:
                    print(f"     Revenue: {row.get('revenue', 'N/A')}")
                if 'cost_of_revenue' in converted_df.columns:
                    print(f"     Cost of Revenue: {row.get('cost_of_revenue', 'N/A')}")
        
        return converted_df
        
    except Exception as e:
        print(f"❌ Error in data conversion: {e}")
        return None

if __name__ == "__main__":
    print("🧪 yfinance Financial Data Structure Debug Tool")
    print("This script investigates the exact structure of yfinance financial data")
    print("to fix the 'period' and 'cost_of_revenue' error.\n")
    
    # Test AAPL
    ticker = debug_yfinance_financials("AAPL")
    
    # Test data conversion
    converted_data = test_data_conversion("AAPL")
    
    print(f"\n🎯 SOLUTION RECOMMENDATIONS")
    print("=" * 60)
    print("1. yfinance returns DataFrames with dates as columns, not rows")
    print("2. Financial items are index values, not column names")
    print("3. Need to transpose and convert data format")
    print("4. Use proper yfinance column names like 'Cost Of Revenue'")
    print("5. Handle missing data gracefully")
    
    print(f"\n🔧 NEXT STEPS")
    print("=" * 60)
    print("1. Update data_manager.py to NOT call load_financial_data for real-time stocks")
    print("2. Create separate yfinance financial data handler")
    print("3. Update Financial Statements page to use correct data source")
    print("4. Add proper error handling for missing financial data")