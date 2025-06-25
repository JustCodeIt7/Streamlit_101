"""
Data Manager for Stock Analysis Dashboard
Handles data loading, caching, and sample data generation
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

# Import configuration
import sys
sys.path.append(str(Path(__file__).parent.parent))
from config.stock_symbols import STOCK_SYMBOLS, DEFAULT_SYMBOL
from config.settings import (
    CACHE_CONFIG, DATA_GENERATION_CONFIG, 
    SAMPLE_STOCKS_DIR, NEWS_SAMPLES_DIR, FINANCIAL_STATEMENTS_DIR
)

class StockDataManager:
    """Centralized data management for stock analysis dashboard"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    @st.cache_data(ttl=CACHE_CONFIG['data_ttl'], max_entries=CACHE_CONFIG['max_entries_data'])
    def load_stock_data(_self, symbol: str) -> pd.DataFrame:
        """
        Load stock data for given symbol with caching
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            
        Returns:
            DataFrame with OHLCV data and technical indicators
        """
        try:
            file_path = SAMPLE_STOCKS_DIR / f"{symbol}.csv"
            
            if file_path.exists():
                df = pd.read_csv(file_path, parse_dates=['date'], index_col='date')
                return df
            else:
                # Generate sample data if file doesn't exist
                return _self._generate_stock_data(symbol)
                
        except Exception as e:
            st.error(f"Error loading data for {symbol}: {e}")
            return _self._get_empty_dataframe()
    
    def _generate_stock_data(self, symbol: str) -> pd.DataFrame:
        """
        Generate realistic stock data using geometric Brownian motion
        
        Args:
            symbol: Stock symbol
            
        Returns:
            DataFrame with generated OHLCV data
        """
        config = DATA_GENERATION_CONFIG
        
        # Set random seed for reproducible data
        np.random.seed(hash(symbol) % 2**32)
        
        # Date range
        start_date = pd.to_datetime(config['start_date'])
        end_date = pd.to_datetime(config['end_date'])
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Filter to business days only
        dates = dates[dates.weekday < 5]  # Monday=0, Friday=4
        
        # Stock parameters
        initial_price = config['initial_prices'].get(symbol, 100.0)
        volatility = config['volatility'].get(symbol, 0.2)
        drift = config['drift'].get(symbol, 0.08)
        
        # Generate price series using geometric Brownian motion
        n_days = len(dates)
        dt = 1/252  # Daily time step (252 trading days per year)
        
        # Random price movements
        random_shocks = np.random.normal(0, 1, n_days)
        price_changes = (drift - 0.5 * volatility**2) * dt + volatility * np.sqrt(dt) * random_shocks
        
        # Generate closing prices
        log_prices = np.log(initial_price) + np.cumsum(price_changes)
        close_prices = np.exp(log_prices)
        
        # Generate OHLV data based on close prices
        data = []
        for i, (date, close) in enumerate(zip(dates, close_prices)):
            # Add some intraday volatility
            daily_vol = volatility / np.sqrt(252) * close
            
            # Generate realistic OHLV
            if i == 0:
                open_price = close * (1 + np.random.normal(0, 0.005))
            else:
                open_price = data[i-1]['close'] * (1 + np.random.normal(0, 0.01))
            
            high = max(open_price, close) * (1 + abs(np.random.normal(0, 0.01)))
            low = min(open_price, close) * (1 - abs(np.random.normal(0, 0.01)))
            
            # Volume (higher volume on larger price moves)
            price_change_pct = abs((close - open_price) / open_price)
            base_volume = 1000000 if symbol != 'SPY' else 50000000
            volume = int(base_volume * (1 + price_change_pct * 5) * np.random.lognormal(0, 0.5))
            
            data.append({
                'date': date,
                'open': round(open_price, 2),
                'high': round(high, 2),
                'low': round(low, 2),
                'close': round(close, 2),
                'volume': volume
            })
        
        df = pd.DataFrame(data)
        df.set_index('date', inplace=True)
        
        # Add basic technical indicators
        df = self._add_basic_indicators(df)
        
        # Save generated data
        self._save_stock_data(df, symbol)
        
        return df
    
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
        
        return df
    
    def _save_stock_data(self, df: pd.DataFrame, symbol: str):
        """Save generated stock data to CSV file"""
        try:
            SAMPLE_STOCKS_DIR.mkdir(parents=True, exist_ok=True)
            file_path = SAMPLE_STOCKS_DIR / f"{symbol}.csv"
            df.to_csv(file_path)
            self.logger.info(f"Saved stock data for {symbol} to {file_path}")
        except Exception as e:
            self.logger.error(f"Error saving stock data for {symbol}: {e}")
    
    def _get_empty_dataframe(self) -> pd.DataFrame:
        """Return empty dataframe with correct structure"""
        columns = ['open', 'high', 'low', 'close', 'volume', 
                  'sma_20', 'sma_50', 'sma_200', 'ema_12', 'ema_26',
                  'volume_ma', 'price_change', 'price_change_pct']
        return pd.DataFrame(columns=columns)
    
    @st.cache_data(ttl=CACHE_CONFIG['data_ttl'], max_entries=5)
    def load_financial_data(_self, symbol: str) -> Dict:
        """
        Load financial statements data for given symbol
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Dictionary with financial statements data
        """
        try:
            file_path = FINANCIAL_STATEMENTS_DIR / f"{symbol}_financials.json"
            
            if file_path.exists():
                with open(file_path, 'r') as f:
                    return json.load(f)
            else:
                return _self._generate_financial_data(symbol)
                
        except Exception as e:
            st.error(f"Error loading financial data for {symbol}: {e}")
            return {}
    
    def _generate_financial_data(self, symbol: str) -> Dict:
        """Generate sample financial statements data"""
        
        # Base financial metrics (different by company)
        base_metrics = {
            'AAPL': {'revenue': 365000000000, 'net_income': 55000000000, 'total_assets': 325000000000},
            'MSFT': {'revenue': 180000000000, 'net_income': 45000000000, 'total_assets': 280000000000},
            'GOOGL': {'revenue': 240000000000, 'net_income': 40000000000, 'total_assets': 320000000000},
            'TSLA': {'revenue': 80000000000, 'net_income': 5000000000, 'total_assets': 90000000000},
            'SPY': {'revenue': 0, 'net_income': 0, 'total_assets': 400000000000},  # ETF
        }
        
        metrics = base_metrics.get(symbol, base_metrics['AAPL'])
        
        # Generate 5 years of quarterly data
        financial_data = {
            'income_statement': [],
            'balance_sheet': [],
            'cash_flow': [],
            'ratios': {}
        }
        
        # Generate quarterly data for last 5 years
        for year in range(2019, 2024):
            for quarter in range(1, 5):
                # Add some growth and seasonality
                growth_factor = (1 + np.random.normal(0.02, 0.05)) ** (year - 2019)
                seasonal_factor = 1 + 0.1 * np.sin(2 * np.pi * quarter / 4)
                
                revenue = int(metrics['revenue'] * growth_factor * seasonal_factor / 4)
                net_income = int(metrics['net_income'] * growth_factor * seasonal_factor / 4)
                
                financial_data['income_statement'].append({
                    'year': year,
                    'quarter': quarter,
                    'revenue': revenue,
                    'net_income': net_income,
                    'gross_profit': int(revenue * 0.4),
                    'operating_income': int(revenue * 0.25),
                })
        
        # Calculate key ratios
        financial_data['ratios'] = {
            'pe_ratio': 25.5,
            'pb_ratio': 6.2,
            'ps_ratio': 7.8,
            'roe': 0.28,
            'roa': 0.15,
            'debt_to_equity': 0.85,
            'current_ratio': 1.2,
            'gross_margin': 0.42,
            'net_margin': 0.15,
        }
        
        # Save generated data
        self._save_financial_data(financial_data, symbol)
        
        return financial_data
    
    def _save_financial_data(self, data: Dict, symbol: str):
        """Save financial data to JSON file"""
        try:
            FINANCIAL_STATEMENTS_DIR.mkdir(parents=True, exist_ok=True)
            file_path = FINANCIAL_STATEMENTS_DIR / f"{symbol}_financials.json"
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
            self.logger.info(f"Saved financial data for {symbol}")
        except Exception as e:
            self.logger.error(f"Error saving financial data for {symbol}: {e}")
    
    @st.cache_data(ttl=CACHE_CONFIG['data_ttl'], max_entries=5)
    def load_news_data(_self, symbol: str) -> List[Dict]:
        """
        Load news and sentiment data for given symbol
        
        Args:
            symbol: Stock symbol
            
        Returns:
            List of news articles with sentiment scores
        """
        try:
            file_path = NEWS_SAMPLES_DIR / f"{symbol}_news.json"
            
            if file_path.exists():
                with open(file_path, 'r') as f:
                    return json.load(f)
            else:
                return _self._generate_news_data(symbol)
                
        except Exception as e:
            st.error(f"Error loading news data for {symbol}: {e}")
            return []
    
    def _generate_news_data(self, symbol: str) -> List[Dict]:
        """Generate sample news data with sentiment scores"""
        
        company_name = STOCK_SYMBOLS[symbol]['name']
        
        # Sample headlines templates
        headline_templates = [
            f"{company_name} Reports Strong Quarterly Earnings",
            f"{company_name} Announces New Product Launch",
            f"Analysts Upgrade {company_name} Stock Rating",
            f"{company_name} Faces Regulatory Challenges",
            f"Market Volatility Affects {company_name} Shares",
            f"{company_name} CEO Discusses Future Strategy",
            f"Industry Trends Favor {company_name}",
            f"{company_name} Stock Reaches New High",
            f"Institutional Investors Increase {company_name} Holdings",
            f"{company_name} Partners with Major Technology Firm",
        ]
        
        sources = ['Reuters', 'Bloomberg', 'CNBC', 'MarketWatch', 'Yahoo Finance']
        
        news_data = []
        
        # Generate 50 articles over the last 30 days
        for i in range(50):
            days_ago = np.random.randint(0, 30)
            article_date = datetime.now() - timedelta(days=days_ago)
            
            headline = np.random.choice(headline_templates)
            source = np.random.choice(sources)
            
            # Generate sentiment score (-1 to 1)
            sentiment_score = np.random.normal(0.1, 0.3)  # Slightly positive bias
            sentiment_score = np.clip(sentiment_score, -1, 1)
            
            # Confidence score (0.5 to 1.0)
            confidence = np.random.uniform(0.5, 1.0)
            
            news_data.append({
                'date': article_date.strftime('%Y-%m-%d'),
                'headline': headline,
                'source': source,
                'sentiment_score': round(sentiment_score, 3),
                'confidence': round(confidence, 3),
                'url': f"https://example.com/news/{i}"
            })
        
        # Sort by date (newest first)
        news_data.sort(key=lambda x: x['date'], reverse=True)
        
        # Save generated data
        self._save_news_data(news_data, symbol)
        
        return news_data
    
    def _save_news_data(self, data: List[Dict], symbol: str):
        """Save news data to JSON file"""
        try:
            NEWS_SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
            file_path = NEWS_SAMPLES_DIR / f"{symbol}_news.json"
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
            self.logger.info(f"Saved news data for {symbol}")
        except Exception as e:
            self.logger.error(f"Error saving news data for {symbol}: {e}")
    
    def get_stock_metrics(self, symbol: str) -> Dict:
        """Get current stock metrics and key statistics"""
        df = self.load_stock_data(symbol)
        
        if df.empty:
            return {}
        
        current_data = df.iloc[-1]
        previous_data = df.iloc[-2] if len(df) > 1 else current_data
        
        # Calculate 52-week high/low
        year_data = df.tail(252)  # Last 252 trading days
        
        metrics = {
            'current_price': current_data['close'],
            'previous_close': previous_data['close'],
            'price_change': current_data['close'] - previous_data['close'],
            'price_change_pct': ((current_data['close'] - previous_data['close']) / previous_data['close']) * 100,
            'volume': current_data['volume'],
            'avg_volume': df['volume'].tail(20).mean(),
            'high_52w': year_data['high'].max(),
            'low_52w': year_data['low'].min(),
            'market_cap': STOCK_SYMBOLS[symbol].get('market_cap', 0),
        }
        
        return metrics
    
    def filter_data_by_period(self, df: pd.DataFrame, period: str) -> pd.DataFrame:
        """Filter dataframe by time period"""
        from config.stock_symbols import TIME_PERIODS
        
        if period not in TIME_PERIODS:
            return df
            
        days = TIME_PERIODS[period]['days']
        return df.tail(days)
    
    def clear_cache(self):
        """Clear all cached data"""
        st.cache_data.clear()
        st.success("Cache cleared successfully!")

# Global instance
data_manager = StockDataManager()