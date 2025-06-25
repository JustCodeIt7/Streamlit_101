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
    
    def _calculate_comprehensive_ratios(self, latest_income, latest_balance, latest_cashflow,
                                      ttm_income, ttm_revenue, ttm_operating_income, ttm_operating_cf,
                                      symbol, base_metrics):
        """Calculate comprehensive financial ratios"""
        
        # Safe division function
        def safe_divide(numerator, denominator, default=0):
            return numerator / denominator if denominator != 0 else default
        
        # Get current stock price (mock based on symbol)
        price_multiples = {'AAPL': 180, 'MSFT': 350, 'GOOGL': 130, 'TSLA': 200, 'SPY': 420}
        current_price = price_multiples.get(symbol, 100)
        
        # Market metrics
        shares = latest_balance['shares_outstanding']
        market_cap = current_price * shares
        enterprise_value = market_cap + latest_balance['long_term_debt'] - latest_balance['cash_and_equivalents']
        
        # Book value
        book_value = latest_balance['stockholders_equity']
        book_value_per_share = safe_divide(book_value, shares)
        
        # PROFITABILITY RATIOS
        profitability = {
            'gross_margin': safe_divide(ttm_revenue - sum([q['cost_of_revenue'] for q in [latest_income]*4]), ttm_revenue),
            'operating_margin': safe_divide(ttm_operating_income, ttm_revenue),
            'net_margin': safe_divide(ttm_income, ttm_revenue),
            'roe': safe_divide(ttm_income, book_value),  # Return on Equity
            'roa': safe_divide(ttm_income, latest_balance['total_assets']),  # Return on Assets
            'roic': safe_divide(ttm_operating_income * 0.79, book_value + latest_balance['long_term_debt']),  # Return on Invested Capital (tax-adjusted)
            'roc': safe_divide(ttm_income, latest_balance['stockholders_equity']),  # Return on Capital
        }
        
        # LIQUIDITY RATIOS
        liquidity = {
            'current_ratio': safe_divide(latest_balance['current_assets'], latest_balance['current_liabilities']),
            'quick_ratio': safe_divide(latest_balance['current_assets'] - latest_balance['inventory'], latest_balance['current_liabilities']),
            'cash_ratio': safe_divide(latest_balance['cash_and_equivalents'], latest_balance['current_liabilities']),
            'operating_cash_ratio': safe_divide(ttm_operating_cf, latest_balance['current_liabilities']),
        }
        
        # LEVERAGE RATIOS
        leverage = {
            'debt_to_equity': safe_divide(latest_balance['long_term_debt'], latest_balance['stockholders_equity']),
            'debt_to_assets': safe_divide(latest_balance['total_liabilities'], latest_balance['total_assets']),
            'equity_ratio': safe_divide(latest_balance['stockholders_equity'], latest_balance['total_assets']),
            'interest_coverage': safe_divide(ttm_operating_income, sum([q['interest_expense'] for q in [latest_income]*4]) or 1),
            'debt_service_coverage': safe_divide(ttm_operating_cf, sum([q['interest_expense'] for q in [latest_income]*4]) or 1),
        }
        
        # EFFICIENCY RATIOS
        efficiency = {
            'asset_turnover': safe_divide(ttm_revenue, latest_balance['total_assets']),
            'inventory_turnover': safe_divide(sum([q['cost_of_revenue'] for q in [latest_income]*4]), latest_balance['inventory']) if latest_balance['inventory'] > 0 else 0,
            'receivables_turnover': safe_divide(ttm_revenue, latest_balance['accounts_receivable']) if latest_balance['accounts_receivable'] > 0 else 0,
            'days_sales_outstanding': safe_divide(latest_balance['accounts_receivable'] * 365, ttm_revenue) if ttm_revenue > 0 else 0,
            'days_inventory_outstanding': safe_divide(latest_balance['inventory'] * 365, sum([q['cost_of_revenue'] for q in [latest_income]*4])) if latest_balance['inventory'] > 0 else 0,
        }
        
        # VALUATION RATIOS
        ttm_eps = safe_divide(ttm_income, shares)
        
        valuation = {
            'pe_ratio': safe_divide(current_price, ttm_eps),
            'pb_ratio': safe_divide(current_price, book_value_per_share),
            'ps_ratio': safe_divide(market_cap, ttm_revenue),
            'peg_ratio': safe_divide(safe_divide(current_price, ttm_eps), 15),  # Assuming 15% growth
            'ev_ebitda': safe_divide(enterprise_value, ttm_operating_income + sum([q.get('depreciation_amortization', 0) for q in [latest_cashflow]*4])),
            'ev_sales': safe_divide(enterprise_value, ttm_revenue),
            'price_to_fcf': safe_divide(market_cap, sum([q['free_cash_flow'] for q in [latest_cashflow]*4])),
        }
        
        # ADDITIONAL METRICS
        additional = {
            'market_cap': market_cap,
            'enterprise_value': enterprise_value,
            'book_value': book_value,
            'book_value_per_share': book_value_per_share,
            'current_price': current_price,
            'ttm_eps': ttm_eps,
            'ttm_revenue': ttm_revenue,
            'ttm_net_income': ttm_income,
            'working_capital': latest_balance['current_assets'] - latest_balance['current_liabilities'],
        }
        
        return {
            'profitability': profitability,
            'liquidity': liquidity,
            'leverage': leverage,
            'efficiency': efficiency,
            'valuation': valuation,
            'additional': additional,
            'calculation_date': datetime.now().strftime('%Y-%m-%d'),
            'data_period': f"TTM ending {latest_income['period']}"
        }
    
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
    def load_news_data(_self, symbol: str) -> Dict:
        """
        Load news and sentiment data for given symbol
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Dictionary with comprehensive sentiment analysis data
        """
        try:
            file_path = NEWS_SAMPLES_DIR / f"{symbol}_news.json"
            
            if file_path.exists():
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    # Handle legacy format (list) vs new format (dict)
                    if isinstance(data, list):
                        # Convert legacy format to new format for backward compatibility
                        return {
                            'news_articles': data,
                            'social_media': [],
                            'summary_stats': {'news_sentiment': {'overall_avg': 0}},
                            'market_impact': {}
                        }
                    return data
            else:
                return _self._generate_news_data(symbol)
                
        except Exception as e:
            st.error(f"Error loading news data for {symbol}: {e}")
            return {
                'news_articles': [],
                'social_media': [],
                'summary_stats': {'news_sentiment': {'overall_avg': 0}},
                'market_impact': {}
            }
    
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
    
    def _calculate_sentiment_stats(self, news_data: List[Dict], social_data: List[Dict]) -> Dict:
        """Calculate comprehensive sentiment statistics"""
        
        # News sentiment stats
        news_sentiments = [article['sentiment_score'] for article in news_data]
        
        # Social media sentiment stats
        social_sentiments = [post['sentiment_score'] for post in social_data]
        
        # Time-based analysis (last 7, 14, 30 days)
        now = datetime.now()
        
        def get_recent_sentiment(data, days, sentiment_key='sentiment_score'):
            cutoff = now - timedelta(days=days)
            recent = [item for item in data if datetime.strptime(item['date'], '%Y-%m-%d') >= cutoff]
            return [item[sentiment_key] for item in recent]
        
        stats = {
            'news_sentiment': {
                'overall_avg': np.mean(news_sentiments) if news_sentiments else 0,
                'overall_std': np.std(news_sentiments) if news_sentiments else 0,
                'positive_ratio': len([s for s in news_sentiments if s > 0.2]) / len(news_sentiments) if news_sentiments else 0,
                'negative_ratio': len([s for s in news_sentiments if s < -0.2]) / len(news_sentiments) if news_sentiments else 0,
                'neutral_ratio': len([s for s in news_sentiments if -0.2 <= s <= 0.2]) / len(news_sentiments) if news_sentiments else 0,
                'last_7_days': np.mean(get_recent_sentiment(news_data, 7)) if get_recent_sentiment(news_data, 7) else 0,
                'last_14_days': np.mean(get_recent_sentiment(news_data, 14)) if get_recent_sentiment(news_data, 14) else 0,
                'last_30_days': np.mean(get_recent_sentiment(news_data, 30)) if get_recent_sentiment(news_data, 30) else 0,
            },
            'social_sentiment': {
                'overall_avg': np.mean(social_sentiments) if social_sentiments else 0,
                'overall_std': np.std(social_sentiments) if social_sentiments else 0,
                'total_volume': sum([post['engagement'] for post in social_data]),
                'last_7_days': np.mean(get_recent_sentiment(social_data, 7)) if get_recent_sentiment(social_data, 7) else 0,
                'last_14_days': np.mean(get_recent_sentiment(social_data, 14)) if get_recent_sentiment(social_data, 14) else 0,
                'last_30_days': np.mean(get_recent_sentiment(social_data, 30)) if get_recent_sentiment(social_data, 30) else 0,
            },
            'trends': {
                'sentiment_momentum': self._calculate_sentiment_momentum(news_data),
                'volatility': np.std(news_sentiments + social_sentiments) if (news_sentiments + social_sentiments) else 0,
                'news_frequency': len([item for item in news_data if datetime.strptime(item['date'], '%Y-%m-%d') >= now - timedelta(days=7)]),
            }
        }
        
        return stats
    
    def _calculate_sentiment_momentum(self, news_data: List[Dict]) -> float:
        """Calculate sentiment momentum (recent vs historical)"""
        now = datetime.now()
        
        recent_cutoff = now - timedelta(days=7)
        historical_cutoff = now - timedelta(days=30)
        
        recent_sentiment = [
            item['sentiment_score'] for item in news_data
            if datetime.strptime(item['date'], '%Y-%m-%d') >= recent_cutoff
        ]
        
        historical_sentiment = [
            item['sentiment_score'] for item in news_data
            if recent_cutoff > datetime.strptime(item['date'], '%Y-%m-%d') >= historical_cutoff
        ]
        
        if not recent_sentiment or not historical_sentiment:
            return 0
        
        recent_avg = np.mean(recent_sentiment)
        historical_avg = np.mean(historical_sentiment)
        
        return recent_avg - historical_avg
    
    def _generate_market_impact_data(self, symbol: str, news_data: List[Dict]) -> Dict:
        """Generate market impact correlation data"""
        
        # Load stock price data for correlation
        try:
            stock_df = self.load_stock_data(symbol)
            if stock_df.empty:
                return {}
            
            # Calculate daily returns
            stock_df['daily_return'] = stock_df['close'].pct_change()
            
            # Aggregate news sentiment by date
            sentiment_by_date = {}
            for article in news_data:
                date = article['date']
                if date not in sentiment_by_date:
                    sentiment_by_date[date] = []
                sentiment_by_date[date].append(article['sentiment_score'])
            
            # Calculate daily average sentiment
            daily_sentiment = {
                date: np.mean(scores) for date, scores in sentiment_by_date.items()
            }
            
            # Find correlation between sentiment and price movements
            correlations = []
            for date, sentiment in daily_sentiment.items():
                try:
                    date_obj = pd.to_datetime(date)
                    if date_obj in stock_df.index:
                        price_return = stock_df.loc[date_obj, 'daily_return']
                        if not pd.isna(price_return):
                            correlations.append((sentiment, price_return))
                except:
                    continue
            
            if len(correlations) > 10:
                sentiments, returns = zip(*correlations)
                correlation = np.corrcoef(sentiments, returns)[0, 1] if len(set(sentiments)) > 1 else 0
            else:
                correlation = 0
            
            # Identify major sentiment events
            major_events = []
            for article in news_data[:20]:  # Top 20 recent articles
                if abs(article['sentiment_score']) > 0.7:
                    major_events.append({
                        'date': article['date'],
                        'headline': article['headline'],
                        'sentiment': article['sentiment_score'],
                        'impact_type': 'positive' if article['sentiment_score'] > 0 else 'negative'
                    })
            
            return {
                'sentiment_price_correlation': round(correlation, 3),
                'major_sentiment_events': major_events,
                'prediction_accuracy': min(abs(correlation) * 100, 85),  # Simulated accuracy
                'market_sensitivity': abs(correlation) * np.random.uniform(0.8, 1.2)
            }
            
        except Exception as e:
            return {'error': f"Could not calculate market impact: {str(e)}"}
    
    def _save_news_data(self, data: Dict, symbol: str):
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