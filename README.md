# Streamlit 101 - Tutorial Code Repository

This repository contains the code examples and applications for the "Streamlit 101" YouTube tutorial series. It's designed to help you learn Streamlit by exploring various features and building a comprehensive multi-page stock analysis dashboard.

## Project Overview

The primary goal of this repository is to provide hands-on code for viewers of the Streamlit 101 tutorial series. It includes:
- Numerous small examples focusing on specific Streamlit functionalities (widgets, layout, data display, etc.).
- A flagship multi-page stock analysis dashboard (`Part_18_Multipage_Stock_Dashboard`) showcasing a more complex, real-world application.

## Directory Structure

The repository is organized into parts, where each `Part_X_...` directory corresponds to a segment of the tutorial series and contains the relevant Streamlit code examples.

- **`Part_1_write/` to `Part_17_Advanced_MultiPage_Nav/`**: These directories contain focused examples demonstrating specific Streamlit features such as text elements, data display, input widgets, media embedding, charting, layout options, chat elements, status indicators, navigation techniques, state management, connection secrets, caching, and custom components.
- **`Part_18_Multipage_Stock_Dashboard/`**: This directory houses a comprehensive multi-page stock analysis dashboard.
- **`Part_18_Multipage_Stock_Dashboard-dev/`**: A development or alternative version of the stock dashboard.
- **`context_portal/`**: Contains files related to a context portal, possibly for managing or accessing data.
- Root directory: Contains general project files like `.gitignore`, `LICENSE`, and a global `requirments.txt` for some of the simpler examples.

## Stock Dashboard (`Part_18_Multipage_Stock_Dashboard`)

This is the main application built throughout the tutorial series, demonstrating how to create a sophisticated, interactive multi-page Streamlit app.

### Key Features:
- **Stock Overview**: Real-time metrics, interactive candlestick charts, volume analysis, and performance summaries.
- **Custom Ticker Input**: Analyze any publicly traded stock by entering its ticker symbol.
- **Real-Time Data**: Fetches live market data from Yahoo Finance.
- **Technical Analysis**: Includes 15+ technical indicators (RSI, MACD, Bollinger Bands, etc.) with customizable parameters, trading signals, and trend analysis.
- **Financial Statements**: Analysis of income statements, balance sheets, and cash flow statements, along with key financial ratios and a financial health assessment. (Note: Full detailed statements are primarily for sample data; real-time data provides key metrics and ratios).
- **Sentiment Analysis**: News and social media sentiment tracking, market impact correlation, and sentiment trend visualization.
- **Price Prediction**: Machine learning models (ARIMA, Prophet, Linear Regression, Ensemble) for stock price forecasting, including model performance comparison and feature importance.

### Setup and Running the Stock Dashboard:
1. Navigate to the dashboard directory:
   ```bash
   cd Part_18_Multipage_Stock_Dashboard
   ```
2. Install the specific dependencies for the dashboard:
   ```bash
   pip install -r requirements.txt
   ```
   Key dependencies include: `streamlit, pandas, numpy, plotly, ta-lib, pandas-ta, scikit-learn, prophet, yfinance`.
3. Run the Streamlit application:
   ```bash
   streamlit run 18_app.py
   ```
4. Open your web browser and go to `http://localhost:8501`.

Refer to `Part_18_Multipage_Stock_Dashboard/README.md` for more detailed information on this specific application.

## Other Tutorial Parts

The directories named `Part_1_write`, `Part_2_Text`, `Part_3_Data_Elements`, etc., contain smaller, self-contained Streamlit applications. These are designed to illustrate individual Streamlit functionalities covered in the tutorials. Users are encouraged to explore these parts to understand specific features in isolation.

## General Usage

To run any of the individual Streamlit apps within the subdirectories (e.g., `Part_1_write`):
1. Navigate to the specific part's directory:
   ```bash
   cd Part_X_Directory_Name
   ```
2. Run the Streamlit application, which is typically named `app.py` or follows a `pX_... .py` pattern:
   ```bash
   streamlit run app.py
   ```
   (Replace `app.py` with the actual main Python file name if different for a particular part).

## Dependencies

- Basic dependencies for some of the simpler tutorial parts are listed in the root `requirments.txt` file (e.g., `streamlit, pandas, numpy, matplotlib`). You can install these using:
  ```bash
  pip install -r requirments.txt
  ```
- More complex applications, particularly the **`Part_18_Multipage_Stock_Dashboard`**, have their own `requirements.txt` file within their respective directories. These should be used to install dependencies for that specific part to ensure all necessary libraries are available.

## Contributing

We welcome contributions and suggestions!
- If you find any issues or have ideas for improvements, please open an issue on the GitHub repository.
- If you'd like to contribute code, please fork the repository and submit a pull request with your changes.

## License

This project is licensed under the terms of the `LICENSE` file. Please see the `LICENSE` file for more details.
---

Happy Streamlit-ing!
