# 📊 Advanced Multi-Page Streamlit Stock Analysis Dashboard

A comprehensive financial analysis application built with Streamlit, featuring professional-grade visualizations, technical analysis, fundamental analysis, sentiment tracking, and machine learning predictions.

## 🎯 Project Overview

This dashboard demonstrates advanced Streamlit capabilities for building professional financial applications. It's designed for YouTube demonstration with consistent sample data and optimal performance.

### Key Features

- 📈 **Stock Overview**: Real-time metrics, interactive candlestick charts, and performance analysis
- 🎯 **Custom Ticker Input**: Enter any valid stock ticker to analyze any publicly traded stock
- 📡 **Real-Time Data**: Fetch live data from Yahoo Finance for up-to-date analysis
- � **Technical Analysis**: 15+ technical indicators with customizable parameters
- 💰 **Financial Statements**: Comprehensive financial health analysis and ratios
- 📰 **Sentiment Analysis**: News sentiment tracking and market impact correlation
- 🤖 **Price Prediction**: Machine learning models for price forecasting

### Stock Analysis Options

#### Pre-defined Stocks (Sample Data)
- **AAPL** - Apple Inc.
- **MSFT** - Microsoft Corporation
- **GOOGL** - Alphabet Inc.
- **TSLA** - Tesla, Inc.
- **SPY** - SPDR S&P 500 ETF

#### Custom Stock Input
- Enter any valid ticker symbol (e.g., NVDA, AMD, META, AMZN)
- Real-time data fetching from Yahoo Finance
- Automatic validation and error handling
- Support for stocks, ETFs, and other securities

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone or download the project**
   ```bash
   cd Part_18_Multipage_Stock_Dashboard
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   streamlit run 18_app.py
   ```

4. **Open your browser** to `http://localhost:8501`

## 📁 Project Structure

```
Part_18_Multipage_Stock_Dashboard/
├── 18_app.py                          # Main application entry point
├── ARCHITECTURE_SPECIFICATION.md      # Detailed technical specification
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── pages/                             # Streamlit pages
│   ├── 01_📈_overview.py              # Stock overview & charts
│   ├── 02_🔧_technical_analysis.py    # Technical indicators (coming soon)
│   ├── 03_💰_financial_statements.py  # Financial ratios (coming soon)
│   ├── 04_📰_sentiment_analysis.py    # News sentiment (coming soon)
│   └── 05_🤖_price_prediction.py      # ML predictions (coming soon)
├── data/                              # Sample data storage
│   ├── sample_stocks/                 # Generated stock data
│   ├── news_samples/                  # Sample news data
│   └── financial_statements/          # Sample financial data
├── utils/                             # Utility modules
│   ├── __init__.py
│   ├── data_manager.py               # Data loading & caching
│   └── chart_components.py           # Reusable chart components
└── config/                           # Configuration files
    ├── settings.py                   # App configuration
    └── stock_symbols.py              # Stock metadata
```

## 🔧 Current Implementation Status

### ✅ Completed (Phase 1)

- [x] Project structure and configuration
- [x] Data management system with intelligent caching
- [x] Sample data generation with realistic patterns
- [x] Main application with navigation and **custom ticker input**
- [x] **Real-time data integration** with yfinance
- [x] Stock Overview page with:
  - Key metrics display
  - Interactive candlestick charts
  - Volume analysis
  - Performance summaries
  - Recent news integration
  - **Real-time and custom stock support**
- [x] Financial Statements page with:
  - **Basic real-time financial metrics**
  - **Comprehensive sample data analysis**
  - **Intelligent data source handling**
- [x] Reusable chart components
- [x] Multi-level caching strategy
- [x] **Robust error handling and validation**

### 🚧 In Progress (Phase 2)

- [ ] Technical Analysis page (real-time integration)
- [ ] Advanced technical indicators
- [ ] Enhanced financial statements for real-time data

### 📋 Planned (Phase 3-4)

- [ ] Sentiment Analysis page
- [ ] Price Prediction page with ML models
- [ ] **Real-time news integration**
- [ ] Performance optimization
- [ ] UI/UX polish

## 🎨 Features Showcase

### Interactive Charts
- **Candlestick Charts**: Professional OHLCV visualization with Plotly
- **Volume Analysis**: Integrated volume bars with price correlation
- **Technical Indicators**: Moving averages, RSI, MACD, Bollinger Bands
- **Multi-timeframe**: 1D to 2Y analysis periods

### Real-time Metrics
- Current price with daily change
- Volume vs average volume
- 52-week high/low ranges
- Volatility and risk metrics

### Performance Analytics
- Return calculations across multiple timeframes
- Risk-adjusted metrics (Sharpe ratio, max drawdown)
- Comparative performance analysis

## 🔧 Technical Implementation

### Architecture Highlights

- **Multi-level Caching**: Optimized performance with Streamlit's caching system
- **Modular Design**: Reusable components for scalability
- **Sample Data**: High-quality synthetic data for consistent demos
- **Error Handling**: Graceful degradation with fallback mechanisms

### Technology Stack

- **Frontend**: Streamlit with custom styling
- **Visualization**: Plotly for interactive financial charts
- **Data Processing**: Pandas and NumPy for efficient calculations
- **Caching**: Streamlit's built-in caching for performance
- **Configuration**: Centralized settings management

### Performance Features

- Smart data filtering for optimal chart rendering
- Cached calculations for technical indicators
- Memory-efficient data structures
- Progressive loading for complex analysis

## 📊 Data Generation

The dashboard uses sophisticated sample data generation to create realistic financial patterns:

### Stock Price Generation
- **Geometric Brownian Motion**: Realistic price movements
- **Volatility Modeling**: Stock-specific volatility patterns
- **Correlation Structure**: Realistic relationships between indicators

### News & Sentiment
- Sample news articles with sentiment scoring
- Realistic publication patterns and sources
- Correlation with price movements

### Financial Data
- Quarterly financial statements
- Industry-appropriate financial ratios
- Historical trend modeling

## 🎬 YouTube Demonstration

This dashboard is optimized for YouTube presentation with:

- **Consistent Performance**: Pre-generated sample data eliminates API delays
- **Professional Appearance**: Dark theme with financial industry styling
- **Interactive Features**: Engaging demonstrations of Streamlit capabilities
- **Educational Value**: Real-world financial analysis patterns

## 🔍 Usage Examples

### Basic Navigation
1. **Choose your stock selection method:**
   - **Predefined Stocks**: Select from AAPL, MSFT, GOOGL, TSLA, SPY
   - **Custom Ticker**: Enter any valid stock ticker symbol

2. **For Custom Tickers:**
   - Switch to "Custom Ticker" mode in the sidebar
   - Enter a ticker symbol (e.g., NVDA, AMD, META)
   - Click "Load Stock" to fetch real-time data
   - The app will validate the ticker and load the data

3. **Configure data source:**
   - Toggle "Use Real-Time Data" for live market data
   - Sample data provides consistent demo experience
   - Real-time data shows current market conditions

4. Choose your analysis timeframe
5. Navigate between pages using sidebar links
6. Interact with charts (zoom, pan, hover)

### Advanced Features
- **Custom Stock Analysis**: Enter any ticker for real-time analysis
- **Data Source Switching**: Toggle between sample and real-time data
- **Automatic Validation**: Invalid tickers are caught and reported
- Customize chart indicators in the Overview page
- Compare performance across different time periods
- Analyze volume patterns and price correlations
- Explore risk metrics and volatility analysis

### Ticker Input Examples
```
Valid Examples:
✅ NVDA (NVIDIA Corporation)
✅ AMD (Advanced Micro Devices)
✅ META (Meta Platforms)
✅ AMZN (Amazon.com Inc.)
✅ QQQ (Invesco QQQ ETF)

Invalid Examples:
❌ INVALID (Non-existent ticker)
❌ 123 (Numbers only)
❌ TOOLONG (More than 5 characters)
```

## 🛠️ Development

### Adding New Features

1. **New Indicators**: Add to `utils/technical_indicators.py`
2. **New Charts**: Extend `utils/chart_components.py`
3. **New Pages**: Create in `pages/` directory following naming convention
4. **Configuration**: Update `config/settings.py` as needed

### Debugging

- Enable debug mode in `config/settings.py`
- Use cache clearing for data refresh
- Check browser console for JavaScript errors
- Monitor Streamlit server logs

## 📚 Learning Resources

- [Streamlit Documentation](https://docs.streamlit.io)
- [Plotly Financial Charts](https://plotly.com/python/candlestick-charts/)
- [Technical Analysis Library](https://technical-analysis-library-in-python.readthedocs.io/)
- [Financial Data Science](https://www.investopedia.com/financial-analysis-4689643)

## 🤝 Contributing

This project is designed for educational purposes. Feel free to:

- Extend functionality with new analysis features
- Improve chart visualizations
- Add new technical indicators
- Enhance the user interface
- Optimize performance

## 📄 License

This project is created for educational purposes as part of a Streamlit tutorial series. Feel free to use and modify for learning and demonstration.

---

**Built with ❤️ using Streamlit • Created for educational demonstration**

*Last updated: December 2024*