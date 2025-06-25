"""
Real-time stock data fetcher using yfinance
Handles downloading and processing live stock data
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple
import logging

class RealDataFetcher:
    """Fetches real stock data using yfinance"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    @st.cache_data(ttl=300, max_entries=20, show_spinner="Fetching real-time data...")  # 5 minute cache
    def fetch_stock_data(_self, symbol: str, period: str = "2y") -> pd.DataFrame:
        """
        Fetch real stock data from yfinance
        
        Args:
            symbol: Stock ticker symbol
            period: Data period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            
        Returns:
            DataFrame with OHLCV data and basic indicators
        """
        try:
            # Create yfinance ticker object
            ticker = yf.Ticker(symbol.upper())
            
            # Fetch historical data
            hist = ticker.history(period=period, auto_adjust=True, prepost=True)
            
            if hist.empty:
                raise ValueError(f"No data found for symbol {symbol}")
            
            # Rename columns to match our format
            hist.columns = [col.lower() for col in hist.columns]
            hist = hist.rename(columns={'adj close': 'adj_close'})
            
            # Reset index to make date a column, then set it back
            hist.reset_index(inplace=True)
            hist['date'] = hist['Date']
            hist.set_index('date', inplace=True)
            hist.drop('Date', axis=1, inplace=True)
            
            # Add basic technical indicators
            hist = _self._add_basic_indicators(hist)
            
            return hist
            
        except Exception as e:
            _self.logger.error(f"Error fetching data for {symbol}: {e}")
            raise e
    
    @st.cache_data(ttl=600, max_entries=10, show_spinner="Fetching company info...")  # 10 minute cache
    def fetch_company_info(_self, symbol: str) -> Dict:
        """
        Fetch company information from yfinance
        
        Args:
            symbol: Stock ticker symbol
            
        Returns:
            Dictionary with company information
        """
        try:
            ticker = yf.Ticker(symbol.upper())
            info = ticker.info
            
            # Extract relevant information
            company_info = {
                'name': info.get('longName', info.get('shortName', symbol.upper())),
                'sector': info.get('sector', 'Unknown'),
                'industry': info.get('industry', 'Unknown'),
                'market_cap': info.get('marketCap', 0),
                'description': info.get('longBusinessSummary', 'No description available'),
                'website': info.get('website', ''),
                'employees': info.get('fullTimeEmployees', 0),
                'country': info.get('country', 'Unknown'),
                'currency': info.get('currency', 'USD'),
                'exchange': info.get('exchange', 'Unknown'),
                'pe_ratio': info.get('trailingPE', None),
                'pb_ratio': info.get('priceToBook', None),
                'dividend_yield': info.get('dividendYield', None),
                'beta': info.get('beta', None),
                '52_week_high': info.get('fiftyTwoWeekHigh', None),
                '52_week_low': info.get('fiftyTwoWeekLow', None),
                'avg_volume': info.get('averageVolume', None),
                'shares_outstanding': info.get('sharesOutstanding', None),
                # Additional financial metrics
                'forward_pe': info.get('forwardPE', None),
                'peg_ratio': info.get('pegRatio', None),
                'price_to_sales': info.get('priceToSalesTrailing12Months', None),
                'enterprise_value': info.get('enterpriseValue', None),
                'ebitda': info.get('ebitda', None),
                'revenue': info.get('totalRevenue', None),
                'gross_profit': info.get('grossProfits', None),
                'free_cashflow': info.get('freeCashflow', None),
                'operating_cashflow': info.get('operatingCashflow', None),
                'total_debt': info.get('totalDebt', None),
                'total_cash': info.get('totalCash', None),
                'book_value': info.get('bookValue', None),
                'current_ratio': info.get('currentRatio', None),
                'debt_to_equity': info.get('debtToEquity', None),
                'return_on_equity': info.get('returnOnEquity', None),
                'return_on_assets': info.get('returnOnAssets', None),
                'profit_margins': info.get('profitMargins', None),
                'operating_margins': info.get('operatingMargins', None),
                'gross_margins': info.get('grossMargins', None),
            }
            
            return company_info
            
        except Exception as e:
            _self.logger.error(f"Error fetching company info for {symbol}: {e}")
            return {
                'name': symbol.upper(),
                'sector': 'Unknown',
                'industry': 'Unknown',
                'market_cap': 0,
                'description': f'Real-time data for {symbol.upper()}',
            }
    
    @st.cache_data(ttl=300, max_entries=10, show_spinner="Fetching latest quote...")  # 5 minute cache
    def fetch_current_quote(_self, symbol: str) -> Dict:
        """
        Fetch current quote data
        
        Args:
            symbol: Stock ticker symbol
            
        Returns:
            Dictionary with current quote information
        """
        try:
            ticker = yf.Ticker(symbol.upper())
            
            # Get current info
            info = ticker.info
            
            # Get recent history for calculations
            hist = ticker.history(period="5d")
            
            if hist.empty:
                raise ValueError("No recent data available")
            
            current = hist.iloc[-1]
            previous = hist.iloc[-2] if len(hist) > 1 else current
            
            quote = {
                'symbol': symbol.upper(),
                'current_price': current['Close'],
                'previous_close': previous['Close'],
                'price_change': current['Close'] - previous['Close'],
                'price_change_pct': ((current['Close'] - previous['Close']) / previous['Close']) * 100,
                'volume': current['Volume'],
                'high': current['High'],
                'low': current['Low'],
                'open': current['Open'],
                'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'market_cap': info.get('marketCap', 0),
                'avg_volume': info.get('averageVolume', current['Volume']),
            }
            
            return quote
            
        except Exception as e:
            _self.logger.error(f"Error fetching quote for {symbol}: {e}")
            raise e
    
    def _add_basic_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add basic technical indicators to the dataframe"""
        
        # Simple Moving Averages
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        df['sma_200'] = df['close'].rolling(window=200).mean()
        
        # Exponential Moving Averages
        df['ema_12'] = df['close'].ewm(span=12).mean()
        df['ema_26'] = df['close'].ewm(span=26).mean()
        
        # Volume Moving Average
        df['volume_ma'] = df['volume'].rolling(window=20).mean()
        
        # Price change
        df['price_change'] = df['close'].diff()
        df['price_change_pct'] = df['close'].pct_change() * 100
        
        # RSI calculation
        df['rsi'] = self._calculate_rsi(df['close'])
        
        # MACD calculation
        macd_data = self._calculate_macd(df['close'])
        df['macd'] = macd_data['macd']
        df['macd_signal'] = macd_data['signal']
        df['macd_histogram'] = macd_data['histogram']
        
        # Bollinger Bands
        bb_data = self._calculate_bollinger_bands(df['close'])
        df['bb_upper'] = bb_data['upper']
        df['bb_middle'] = bb_data['middle']
        df['bb_lower'] = bb_data['lower']
        
        return df
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict:
        """Calculate MACD"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal).mean()
        histogram = macd - signal_line
        
        return {
            'macd': macd,
            'signal': signal_line,
            'histogram': histogram
        }
    
    def _calculate_bollinger_bands(self, prices: pd.Series, window: int = 20, num_std: int = 2) -> Dict:
        """Calculate Bollinger Bands"""
        rolling_mean = prices.rolling(window=window).mean()
        rolling_std = prices.rolling(window=window).std()
        
        upper_band = rolling_mean + (rolling_std * num_std)
        lower_band = rolling_mean - (rolling_std * num_std)
        
        return {
            'upper': upper_band,
            'middle': rolling_mean,
            'lower': lower_band
        }

    @st.cache_data(ttl=3600, max_entries=5, show_spinner="Fetching financial statements...")  # 1 hour cache
    def fetch_financial_statements(_self, symbol: str) -> Dict:
        """
        Fetch financial statements from yfinance and convert to expected format

        Args:
            symbol: Stock ticker symbol

        Returns:
            Dictionary with financial statements in expected format
        """
        try:
            ticker = yf.Ticker(symbol.upper())

            # Fetch financial statements
            income_stmt = ticker.income_stmt
            balance_sheet = ticker.balance_sheet
            cashflow = ticker.cashflow
            quarterly_income = ticker.quarterly_income_stmt

            # Convert to expected format
            financial_data = {
                "income_statement": _self._convert_income_statement(income_stmt, quarterly_income),
                "balance_sheet": _self._convert_balance_sheet(balance_sheet),
                "cash_flow": _self._convert_cashflow(cashflow),
                "ratios": _self._calculate_financial_ratios(income_stmt, balance_sheet, cashflow),
            }

            return financial_data

        except Exception as e:
            _self.logger.error(f"Error fetching financial statements for {symbol}: {e}")
            return {}

    def _convert_income_statement(self, income_stmt, quarterly_income):
        """Convert yfinance income statement to expected format"""
        try:
            # Use annual data as primary, quarterly as backup
            if not income_stmt.empty:
                df = income_stmt.T  # Transpose: dates become rows, items become columns
            elif not quarterly_income.empty:
                df = quarterly_income.T
            else:
                return []

            df.reset_index(inplace=True)
            df.rename(columns={"index": "period"}, inplace=True)

            # Rename columns to match expected format
            column_mapping = {
                "Total Revenue": "revenue",
                "Cost Of Revenue": "cost_of_revenue",
                "Gross Profit": "gross_profit",
                "Operating Income": "operating_income",
                "Net Income": "net_income",
                "Operating Revenue": "revenue",  # Alternative name
                "Reconciled Cost Of Revenue": "cost_of_revenue",  # Alternative name
            }

            for old_name, new_name in column_mapping.items():
                if old_name in df.columns:
                    df.rename(columns={old_name: new_name}, inplace=True)

            # Convert to list of dictionaries
            result = []
            for _, row in df.iterrows():
                period_str = str(row["period"])[:10]  # Format as YYYY-MM-DD

                # Safe getter function for row values
                def safe_row_get(key, default=0):
                    value = row.get(key, default)
                    if pd.isna(value):
                        return default
                    try:
                        return float(value)
                    except (ValueError, TypeError):
                        return default

                record = {
                    "year": int(period_str[:4]) if period_str != "NaT" and len(period_str) >= 4 else 2024,
                    "quarter": 4,  # Default for annual data
                    "period": period_str,
                    "revenue": safe_row_get("revenue"),
                    "cost_of_revenue": safe_row_get("cost_of_revenue"),
                    "gross_profit": safe_row_get("gross_profit"),
                    "operating_income": safe_row_get("operating_income"),
                    "net_income": safe_row_get("net_income"),
                }
                result.append(record)

            return result

        except Exception as e:
            self.logger.error(f"Error converting income statement: {e}")
            return []

    def _convert_balance_sheet(self, balance_sheet):
        """Convert yfinance balance sheet to expected format"""
        try:
            if balance_sheet.empty:
                return []

            df = balance_sheet.T
            df.reset_index(inplace=True)
            df.rename(columns={"index": "period"}, inplace=True)

            # Map balance sheet items
            column_mapping = {
                "Cash And Cash Equivalents": "cash_and_equivalents",
                "Accounts Receivable": "accounts_receivable",
                "Inventory": "inventory",
                "Current Assets": "current_assets",
                "Total Assets": "total_assets",
                "Current Liabilities": "current_liabilities",
                "Long Term Debt": "long_term_debt",
                "Total Liabilities Net Minority Interest": "total_liabilities",
                "Stockholders Equity": "stockholders_equity",
                "Ordinary Shares Number": "shares_outstanding",
            }

            for old_name, new_name in column_mapping.items():
                if old_name in df.columns:
                    df.rename(columns={old_name: new_name}, inplace=True)

            result = []
            for _, row in df.iterrows():
                period_str = str(row["period"])[:10]

                # Safe getter function for row values
                def safe_row_get(key, default=0):
                    value = row.get(key, default)
                    if pd.isna(value):
                        return default
                    try:
                        return float(value)
                    except (ValueError, TypeError):
                        return default

                record = {
                    "period": period_str,
                    "cash_and_equivalents": safe_row_get("cash_and_equivalents"),
                    "accounts_receivable": safe_row_get("accounts_receivable"),
                    "inventory": safe_row_get("inventory"),
                    "current_assets": safe_row_get("current_assets"),
                    "total_assets": safe_row_get("total_assets"),
                    "current_liabilities": safe_row_get("current_liabilities"),
                    "long_term_debt": safe_row_get("long_term_debt"),
                    "total_liabilities": safe_row_get("total_liabilities"),
                    "stockholders_equity": safe_row_get("stockholders_equity"),
                    "shares_outstanding": safe_row_get("shares_outstanding"),
                }
                result.append(record)

            return result

        except Exception as e:
            self.logger.error(f"Error converting balance sheet: {e}")
            return []

    def _convert_cashflow(self, cashflow):
        """Convert yfinance cash flow to expected format"""
        try:
            if cashflow.empty:
                return []

            df = cashflow.T
            df.reset_index(inplace=True)
            df.rename(columns={"index": "period"}, inplace=True)

            column_mapping = {
                "Operating Cash Flow": "operating_cash_flow",
                "Investing Cash Flow": "investing_cash_flow",
                "Financing Cash Flow": "financing_cash_flow",
                "Free Cash Flow": "free_cash_flow",
                "End Cash Position": "net_change_cash",
            }

            for old_name, new_name in column_mapping.items():
                if old_name in df.columns:
                    df.rename(columns={old_name: new_name}, inplace=True)

            result = []
            for _, row in df.iterrows():
                period_str = str(row["period"])[:10]

                record = {
                    "period": period_str,
                    "operating_cash_flow": float(row.get("operating_cash_flow", 0))
                    if pd.notna(row.get("operating_cash_flow"))
                    else 0,
                    "investing_cash_flow": float(row.get("investing_cash_flow", 0))
                    if pd.notna(row.get("investing_cash_flow"))
                    else 0,
                    "financing_cash_flow": float(row.get("financing_cash_flow", 0))
                    if pd.notna(row.get("financing_cash_flow"))
                    else 0,
                    "free_cash_flow": float(row.get("free_cash_flow", 0)) if pd.notna(row.get("free_cash_flow")) else 0,
                    "net_change_cash": float(row.get("net_change_cash", 0))
                    if pd.notna(row.get("net_change_cash"))
                    else 0,
                }
                result.append(record)

            return result

        except Exception as e:
            self.logger.error(f"Error converting cash flow: {e}")
            return []

    def _calculate_financial_ratios(self, income_stmt, balance_sheet, cashflow):
        """Calculate financial ratios from yfinance data"""
        try:
            # Get latest period data
            if income_stmt.empty or balance_sheet.empty:
                return {}

            latest_income = income_stmt.iloc[:, 0]  # First column (latest period)
            latest_balance = balance_sheet.iloc[:, 0]

            # Extract key metrics with safe access
            def safe_get(series, key, default=0):
                return float(series.get(key, default)) if pd.notna(series.get(key)) else default

            revenue = safe_get(latest_income, "Total Revenue")
            cost_of_revenue = safe_get(latest_income, "Cost Of Revenue")
            gross_profit = safe_get(latest_income, "Gross Profit")
            operating_income = safe_get(latest_income, "Operating Income")
            net_income = safe_get(latest_income, "Net Income")

            total_assets = safe_get(latest_balance, "Total Assets")
            current_assets = safe_get(latest_balance, "Current Assets")
            current_liabilities = safe_get(latest_balance, "Current Liabilities")
            stockholders_equity = safe_get(latest_balance, "Stockholders Equity")
            long_term_debt = safe_get(latest_balance, "Long Term Debt")

            # Calculate ratios with safe division
            def safe_divide(a, b):
                return a / b if b != 0 else 0

            ratios = {
                "profitability": {
                    "gross_margin": safe_divide(gross_profit, revenue),
                    "operating_margin": safe_divide(operating_income, revenue),
                    "net_margin": safe_divide(net_income, revenue),
                    "roe": safe_divide(net_income, stockholders_equity),
                    "roa": safe_divide(net_income, total_assets),
                },
                "liquidity": {
                    "current_ratio": safe_divide(current_assets, current_liabilities),
                    "quick_ratio": safe_divide(current_assets, current_liabilities),  # Simplified
                },
                "leverage": {
                    "debt_to_equity": safe_divide(long_term_debt, stockholders_equity),
                    "debt_to_assets": safe_divide(long_term_debt, total_assets),
                },
                "additional": {
                    "ttm_revenue": revenue,
                    "ttm_net_income": net_income,
                },
            }

            return ratios

        except Exception as e:
            self.logger.error(f"Error calculating financial ratios: {e}")
            return {}

    def validate_symbol(self, symbol: str) -> Tuple[bool, str]:
        """
        Validate if a stock symbol exists and can be fetched
        
        Args:
            symbol: Stock ticker symbol
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            ticker = yf.Ticker(symbol.upper())
            
            # Try to fetch recent data (1 day)
            hist = ticker.history(period="1d")
            
            if hist.empty:
                return False, f"No data available for symbol '{symbol}'"
            
            # Try to get basic info
            info = ticker.info
            if not info or 'regularMarketPrice' not in info and len(hist) == 0:
                return False, f"Invalid symbol '{symbol}'"
            
            return True, "Valid symbol"
            
        except Exception as e:
            return False, f"Error validating symbol '{symbol}': {str(e)}"

# Global instance
real_data_fetcher = RealDataFetcher()