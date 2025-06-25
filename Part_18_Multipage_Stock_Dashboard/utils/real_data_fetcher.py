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