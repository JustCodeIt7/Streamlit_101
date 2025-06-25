"""
Financial Statements Page
========================

Comprehensive financial statement analysis with income statement, balance sheet, 
cash flow statement, financial ratios, and health assessment.
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from datetime import datetime, timedelta

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

# Import configuration and utilities
from config.settings import STREAMLIT_CONFIG, FINANCIAL_RATIOS
from config.stock_symbols import get_stock_info, is_predefined_stock
from utils.data_manager import data_manager
from utils.chart_components import chart_components

# Configure page
st.set_page_config(**STREAMLIT_CONFIG)

def get_current_selection():
    """Get current stock selection from session state"""
    return st.session_state.get('selected_stock', 'AAPL')

def display_financial_header(symbol: str):
    """Display financial statements header"""
    stock_info = get_stock_info(symbol)
    
    if stock_info:
        st.title(f"📋 Financial Statements - {stock_info['name']} ({symbol})")
        st.markdown(f"**Sector:** {stock_info.get('sector', 'N/A')} | **Industry:** {stock_info.get('industry', 'N/A')}")
    else:
        st.title(f"📋 Financial Statements - {symbol}")

def display_financial_overview(financial_data: dict, symbol: str):
    """Display key financial metrics overview"""
    
    if not financial_data or 'ratios' not in financial_data:
        st.warning("Financial data not available")
        return
    
    ratios = financial_data['ratios']
    additional = ratios.get('additional', {})
    
    st.subheader("📊 Financial Overview")
    
    # Key metrics row
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        market_cap = additional.get('market_cap', 0)
        if market_cap >= 1e12:
            cap_str = f"${market_cap/1e12:.1f}T"
        elif market_cap >= 1e9:
            cap_str = f"${market_cap/1e9:.1f}B"
        else:
            cap_str = f"${market_cap/1e6:.1f}M"
        st.metric("Market Cap", cap_str)
    
    with col2:
        ttm_revenue = additional.get('ttm_revenue', 0)
        if ttm_revenue >= 1e9:
            rev_str = f"${ttm_revenue/1e9:.1f}B"
        else:
            rev_str = f"${ttm_revenue/1e6:.1f}M"
        st.metric("TTM Revenue", rev_str)
    
    with col3:
        ttm_income = additional.get('ttm_net_income', 0)
        if ttm_income >= 1e9:
            income_str = f"${ttm_income/1e9:.1f}B"
        else:
            income_str = f"${ttm_income/1e6:.1f}M"
        st.metric("TTM Net Income", income_str)
    
    with col4:
        eps = additional.get('ttm_eps', 0)
        st.metric("TTM EPS", f"${eps:.2f}")
    
    with col5:
        pe_ratio = ratios.get('valuation', {}).get('pe_ratio', 0)
        st.metric("P/E Ratio", f"{pe_ratio:.1f}")

def display_income_statement(financial_data: dict):
    """Display income statement"""
    
    st.subheader("💰 Income Statement")
    
    if 'income_statement' not in financial_data:
        st.warning("Income statement data not available")
        return
    
    income_data = financial_data['income_statement']
    
    # Period selection
    col1, col2 = st.columns([1, 3])
    
    with col1:
        view_type = st.selectbox("View Type", ["Quarterly", "Annual"], key="income_view")
        periods_to_show = st.slider("Periods to Show", 4, 16, 8, key="income_periods")
    
    # Process data based on selection
    if view_type == "Annual":
        # Aggregate quarterly data to annual
        annual_data = {}
        for item in income_data[-periods_to_show:]:
            year = item['year']
            if year not in annual_data:
                annual_data[year] = {k: 0 for k in item.keys() if isinstance(item[k], (int, float))}
                annual_data[year]['year'] = year
                annual_data[year]['period'] = str(year)
            
            for key, value in item.items():
                if isinstance(value, (int, float)) and key != 'year':
                    annual_data[year][key] += value
        
        display_data = list(annual_data.values())
    else:
        display_data = income_data[-periods_to_show:]
    
    if not display_data:
        st.warning("No data available for selected period")
        return
    
    # Create DataFrame for display
    df = pd.DataFrame(display_data)
    
    # Format the data for better display
    financial_columns = ['revenue', 'cost_of_revenue', 'gross_profit', 'operating_income', 'net_income']
    display_df = df[['period'] + financial_columns].copy()
    
    # Format numbers to millions/billions
    for col in financial_columns:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(lambda x: f"${x/1e9:.2f}B" if abs(x) >= 1e9 else f"${x/1e6:.1f}M")
    
    # Rename columns for better display
    display_df.columns = ['Period', 'Revenue', 'Cost of Revenue', 'Gross Profit', 'Operating Income', 'Net Income']
    
    st.dataframe(display_df, use_container_width=True)
    
    # Income statement chart
    fig = create_income_statement_chart(df, view_type)
    st.plotly_chart(fig, use_container_width=True)

def display_balance_sheet(financial_data: dict):
    """Display balance sheet"""
    
    st.subheader("⚖️ Balance Sheet")
    
    if 'balance_sheet' not in financial_data:
        st.warning("Balance sheet data not available")
        return
    
    balance_data = financial_data['balance_sheet']
    
    # Period selection
    col1, col2 = st.columns([1, 3])
    
    with col1:
        periods_to_show = st.slider("Periods to Show", 4, 16, 8, key="balance_periods")
    
    # Get recent data
    recent_data = balance_data[-periods_to_show:]
    
    if not recent_data:
        st.warning("No balance sheet data available")
        return
    
    df = pd.DataFrame(recent_data)
    
    # Assets, Liabilities, Equity breakdown
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Assets")
        assets_df = df[['period', 'cash_and_equivalents', 'accounts_receivable', 'inventory', 
                       'current_assets', 'total_assets']].copy()
        
        # Format numbers
        for col in assets_df.columns[1:]:
            assets_df[col] = assets_df[col].apply(lambda x: f"${x/1e9:.2f}B" if abs(x) >= 1e9 else f"${x/1e6:.1f}M")
        
        assets_df.columns = ['Period', 'Cash & Equiv.', 'Receivables', 'Inventory', 'Current Assets', 'Total Assets']
        st.dataframe(assets_df, use_container_width=True)
    
    with col2:
        st.markdown("#### Liabilities & Equity")
        liab_equity_df = df[['period', 'current_liabilities', 'long_term_debt', 
                            'total_liabilities', 'stockholders_equity']].copy()
        
        # Format numbers
        for col in liab_equity_df.columns[1:]:
            liab_equity_df[col] = liab_equity_df[col].apply(lambda x: f"${x/1e9:.2f}B" if abs(x) >= 1e9 else f"${x/1e6:.1f}M")
        
        liab_equity_df.columns = ['Period', 'Current Liab.', 'Long-term Debt', 'Total Liab.', 'Equity']
        st.dataframe(liab_equity_df, use_container_width=True)
    
    # Balance sheet visualization
    fig = create_balance_sheet_chart(df)
    st.plotly_chart(fig, use_container_width=True)

def display_cash_flow_statement(financial_data: dict):
    """Display cash flow statement"""
    
    st.subheader("💵 Cash Flow Statement")
    
    if 'cash_flow' not in financial_data:
        st.warning("Cash flow data not available")
        return
    
    cash_flow_data = financial_data['cash_flow']
    
    # Period selection
    periods_to_show = st.slider("Periods to Show", 4, 16, 8, key="cashflow_periods")
    
    # Get recent data
    recent_data = cash_flow_data[-periods_to_show:]
    
    if not recent_data:
        st.warning("No cash flow data available")
        return
    
    df = pd.DataFrame(recent_data)
    
    # Cash flow components
    cashflow_df = df[['period', 'operating_cash_flow', 'investing_cash_flow', 
                     'financing_cash_flow', 'net_change_cash', 'free_cash_flow']].copy()
    
    # Format numbers
    for col in cashflow_df.columns[1:]:
        cashflow_df[col] = cashflow_df[col].apply(lambda x: f"${x/1e9:.2f}B" if abs(x) >= 1e9 else f"${x/1e6:.1f}M")
    
    cashflow_df.columns = ['Period', 'Operating CF', 'Investing CF', 'Financing CF', 'Net Change', 'Free CF']
    st.dataframe(cashflow_df, use_container_width=True)
    
    # Cash flow waterfall chart
    fig = create_cash_flow_chart(df)
    st.plotly_chart(fig, use_container_width=True)

def display_financial_ratios(financial_data: dict):
    """Display comprehensive financial ratios analysis"""
    
    st.subheader("📈 Financial Ratio Analysis")
    
    if 'ratios' not in financial_data:
        st.warning("Ratio data not available")
        return
    
    ratios = financial_data['ratios']
    
    # Create tabs for different ratio categories
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Profitability", "Liquidity", "Leverage", "Efficiency", "Valuation"])
    
    with tab1:
        display_profitability_ratios(ratios.get('profitability', {}))
    
    with tab2:
        display_liquidity_ratios(ratios.get('liquidity', {}))
    
    with tab3:
        display_leverage_ratios(ratios.get('leverage', {}))
    
    with tab4:
        display_efficiency_ratios(ratios.get('efficiency', {}))
    
    with tab5:
        display_valuation_ratios(ratios.get('valuation', {}))

def display_profitability_ratios(profitability: dict):
    """Display profitability ratios"""
    
    if not profitability:
        st.warning("Profitability ratios not available")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Gross Margin", f"{profitability.get('gross_margin', 0):.1%}")
        st.metric("Operating Margin", f"{profitability.get('operating_margin', 0):.1%}")
        st.metric("Net Margin", f"{profitability.get('net_margin', 0):.1%}")
    
    with col2:
        st.metric("Return on Equity (ROE)", f"{profitability.get('roe', 0):.1%}")
        st.metric("Return on Assets (ROA)", f"{profitability.get('roa', 0):.1%}")
        st.metric("Return on Invested Capital (ROIC)", f"{profitability.get('roic', 0):.1%}")
    
    # Profitability chart
    fig = create_profitability_chart(profitability)
    st.plotly_chart(fig, use_container_width=True)

def display_liquidity_ratios(liquidity: dict):
    """Display liquidity ratios"""
    
    if not liquidity:
        st.warning("Liquidity ratios not available")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        current_ratio = liquidity.get('current_ratio', 0)
        st.metric("Current Ratio", f"{current_ratio:.2f}", 
                 help="Current Assets / Current Liabilities. >1.0 is generally good.")
        
        quick_ratio = liquidity.get('quick_ratio', 0)
        st.metric("Quick Ratio", f"{quick_ratio:.2f}",
                 help="(Current Assets - Inventory) / Current Liabilities")
    
    with col2:
        cash_ratio = liquidity.get('cash_ratio', 0)
        st.metric("Cash Ratio", f"{cash_ratio:.2f}",
                 help="Cash & Equivalents / Current Liabilities")
        
        op_cash_ratio = liquidity.get('operating_cash_ratio', 0)
        st.metric("Operating Cash Ratio", f"{op_cash_ratio:.2f}",
                 help="Operating Cash Flow / Current Liabilities")
    
    # Liquidity assessment
    liquidity_score = calculate_liquidity_score(liquidity)
    st.info(f"**Liquidity Assessment:** {liquidity_score}")

def display_leverage_ratios(leverage: dict):
    """Display leverage ratios"""
    
    if not leverage:
        st.warning("Leverage ratios not available")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        debt_equity = leverage.get('debt_to_equity', 0)
        st.metric("Debt-to-Equity", f"{debt_equity:.2f}",
                 help="Long-term Debt / Stockholders' Equity")
        
        debt_assets = leverage.get('debt_to_assets', 0)
        st.metric("Debt-to-Assets", f"{debt_assets:.2f}",
                 help="Total Liabilities / Total Assets")
        
        equity_ratio = leverage.get('equity_ratio', 0)
        st.metric("Equity Ratio", f"{equity_ratio:.2f}",
                 help="Stockholders' Equity / Total Assets")
    
    with col2:
        interest_coverage = leverage.get('interest_coverage', 0)
        st.metric("Interest Coverage", f"{interest_coverage:.1f}x",
                 help="Operating Income / Interest Expense")
        
        debt_service = leverage.get('debt_service_coverage', 0)
        st.metric("Debt Service Coverage", f"{debt_service:.1f}x",
                 help="Operating Cash Flow / Interest Expense")
    
    # Leverage assessment
    leverage_score = calculate_leverage_score(leverage)
    st.info(f"**Leverage Assessment:** {leverage_score}")

def display_efficiency_ratios(efficiency: dict):
    """Display efficiency ratios"""
    
    if not efficiency:
        st.warning("Efficiency ratios not available")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        asset_turnover = efficiency.get('asset_turnover', 0)
        st.metric("Asset Turnover", f"{asset_turnover:.2f}x",
                 help="Revenue / Total Assets")
        
        inventory_turnover = efficiency.get('inventory_turnover', 0)
        if inventory_turnover > 0:
            st.metric("Inventory Turnover", f"{inventory_turnover:.1f}x",
                     help="Cost of Goods Sold / Average Inventory")
        
        receivables_turnover = efficiency.get('receivables_turnover', 0)
        if receivables_turnover > 0:
            st.metric("Receivables Turnover", f"{receivables_turnover:.1f}x",
                     help="Revenue / Average Accounts Receivable")
    
    with col2:
        dso = efficiency.get('days_sales_outstanding', 0)
        if dso > 0:
            st.metric("Days Sales Outstanding", f"{dso:.0f} days",
                     help="Average days to collect receivables")
        
        dio = efficiency.get('days_inventory_outstanding', 0)
        if dio > 0:
            st.metric("Days Inventory Outstanding", f"{dio:.0f} days",
                     help="Average days inventory is held")

def display_valuation_ratios(valuation: dict):
    """Display valuation ratios"""
    
    if not valuation:
        st.warning("Valuation ratios not available")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        pe_ratio = valuation.get('pe_ratio', 0)
        st.metric("P/E Ratio", f"{pe_ratio:.1f}",
                 help="Price per Share / Earnings per Share")
        
        pb_ratio = valuation.get('pb_ratio', 0)
        st.metric("P/B Ratio", f"{pb_ratio:.1f}",
                 help="Price per Share / Book Value per Share")
        
        ps_ratio = valuation.get('ps_ratio', 0)
        st.metric("P/S Ratio", f"{ps_ratio:.1f}",
                 help="Market Cap / Revenue")
    
    with col2:
        peg_ratio = valuation.get('peg_ratio', 0)
        st.metric("PEG Ratio", f"{peg_ratio:.2f}",
                 help="P/E Ratio / Earnings Growth Rate")
        
        ev_ebitda = valuation.get('ev_ebitda', 0)
        st.metric("EV/EBITDA", f"{ev_ebitda:.1f}",
                 help="Enterprise Value / EBITDA")
        
        price_fcf = valuation.get('price_to_fcf', 0)
        st.metric("Price/FCF", f"{price_fcf:.1f}",
                 help="Market Cap / Free Cash Flow")

def display_financial_health_assessment(financial_data: dict, symbol: str):
    """Display overall financial health assessment"""
    
    st.subheader("🏥 Financial Health Assessment")
    
    if 'ratios' not in financial_data:
        st.warning("Cannot assess financial health without ratio data")
        return
    
    ratios = financial_data['ratios']
    
    # Calculate health scores
    profitability_score = calculate_profitability_score(ratios.get('profitability', {}))
    liquidity_score_val = calculate_liquidity_score_value(ratios.get('liquidity', {}))
    leverage_score_val = calculate_leverage_score_value(ratios.get('leverage', {}))
    efficiency_score_val = calculate_efficiency_score_value(ratios.get('efficiency', {}))
    
    # Overall health score (weighted average)
    overall_score = (profitability_score * 0.3 + liquidity_score_val * 0.2 + 
                    leverage_score_val * 0.3 + efficiency_score_val * 0.2)
    
    # Health rating
    if overall_score >= 80:
        health_rating = "Excellent"
        health_color = "green"
    elif overall_score >= 65:
        health_rating = "Good"
        health_color = "blue"
    elif overall_score >= 50:
        health_rating = "Fair"
        health_color = "orange"
    else:
        health_rating = "Poor"
        health_color = "red"
    
    # Display overall assessment
    col1, col2, col3 = st.columns([2, 1, 2])
    
    with col1:
        st.markdown(f"### Overall Financial Health: <span style='color:{health_color}'>{health_rating}</span>", 
                   unsafe_allow_html=True)
        st.metric("Health Score", f"{overall_score:.0f}/100")
    
    with col2:
        # Health gauge chart
        fig = create_health_gauge(overall_score)
        st.plotly_chart(fig, use_container_width=True)
    
    with col3:
        st.markdown("#### Component Scores")
        st.metric("Profitability", f"{profitability_score:.0f}/100")
        st.metric("Liquidity", f"{liquidity_score_val:.0f}/100")
        st.metric("Leverage", f"{leverage_score_val:.0f}/100")
        st.metric("Efficiency", f"{efficiency_score_val:.0f}/100")
    
    # Detailed analysis
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🟢 Strengths")
        strengths = identify_strengths(ratios)
        for strength in strengths:
            st.write(f"• {strength}")
    
    with col2:
        st.markdown("#### 🔴 Areas for Improvement")
        weaknesses = identify_weaknesses(ratios)
        for weakness in weaknesses:
            st.write(f"• {weakness}")

# Chart creation functions
def create_income_statement_chart(df: pd.DataFrame, view_type: str) -> go.Figure:
    """Create income statement trend chart"""
    
    fig = go.Figure()
    
    # Revenue trend
    fig.add_trace(go.Scatter(
        x=df['period'],
        y=df['revenue']/1e9,
        mode='lines+markers',
        name='Revenue',
        line=dict(color='blue', width=3)
    ))
    
    # Gross profit
    fig.add_trace(go.Scatter(
        x=df['period'],
        y=df['gross_profit']/1e9,
        mode='lines+markers',
        name='Gross Profit',
        line=dict(color='green', width=2)
    ))
    
    # Operating income
    fig.add_trace(go.Scatter(
        x=df['period'],
        y=df['operating_income']/1e9,
        mode='lines+markers',
        name='Operating Income',
        line=dict(color='orange', width=2)
    ))
    
    # Net income
    fig.add_trace(go.Scatter(
        x=df['period'],
        y=df['net_income']/1e9,
        mode='lines+markers',
        name='Net Income',
        line=dict(color='red', width=2)
    ))
    
    fig.update_layout(
        title=f"Income Statement Trends ({view_type})",
        xaxis_title="Period",
        yaxis_title="Amount (Billions USD)",
        template="plotly_dark",
        height=400,
        hovermode='x unified'
    )
    
    return fig

def create_balance_sheet_chart(df: pd.DataFrame) -> go.Figure:
    """Create balance sheet composition chart"""
    
    # Create stacked bar chart
    fig = go.Figure()
    
    periods = df['period'].tolist()
    
    # Assets
    fig.add_trace(go.Bar(
        name='Current Assets',
        x=periods,
        y=df['current_assets']/1e9,
        marker_color='lightblue'
    ))
    
    fig.add_trace(go.Bar(
        name='Non-Current Assets',
        x=periods,
        y=(df['total_assets'] - df['current_assets'])/1e9,
        marker_color='darkblue'
    ))
    
    fig.update_layout(
        title="Balance Sheet - Assets Composition",
        xaxis_title="Period",
        yaxis_title="Amount (Billions USD)",
        barmode='stack',
        template="plotly_dark",
        height=300
    )
    
    return fig

def create_cash_flow_chart(df: pd.DataFrame) -> go.Figure:
    """Create cash flow waterfall chart"""
    
    # Use latest period data
    latest = df.iloc[-1]
    
    # Create waterfall components
    categories = ['Operating CF', 'Investing CF', 'Financing CF']
    values = [
        latest['operating_cash_flow']/1e9,
        latest['investing_cash_flow']/1e9,
        latest['financing_cash_flow']/1e9
    ]
    
    colors = ['green' if v > 0 else 'red' for v in values]
    
    fig = go.Figure(go.Bar(
        x=categories,
        y=values,
        marker_color=colors,
        text=[f"${v:.1f}B" for v in values],
        textposition='auto'
    ))
    
    fig.update_layout(
        title=f"Cash Flow Components - {latest['period']}",
        xaxis_title="Cash Flow Type",
        yaxis_title="Amount (Billions USD)",
        template="plotly_dark",
        height=300
    )
    
    return fig

def create_profitability_chart(profitability: dict) -> go.Figure:
    """Create profitability ratios chart"""
    
    ratios = ['Gross Margin', 'Operating Margin', 'Net Margin', 'ROE', 'ROA', 'ROIC']
    values = [
        profitability.get('gross_margin', 0) * 100,
        profitability.get('operating_margin', 0) * 100,
        profitability.get('net_margin', 0) * 100,
        profitability.get('roe', 0) * 100,
        profitability.get('roa', 0) * 100,
        profitability.get('roic', 0) * 100
    ]
    
    fig = go.Figure(go.Bar(
        x=ratios,
        y=values,
        marker_color='green',
        text=[f"{v:.1f}%" for v in values],
        textposition='auto'
    ))
    
    fig.update_layout(
        title="Profitability Ratios",
        xaxis_title="Ratio",
        yaxis_title="Percentage (%)",
        template="plotly_dark",
        height=300
    )
    
    return fig

def create_health_gauge(score: float) -> go.Figure:
    """Create financial health gauge chart"""
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Health Score"},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 50], 'color': "lightgray"},
                {'range': [50, 65], 'color': "yellow"},
                {'range': [65, 80], 'color': "lightgreen"},
                {'range': [80, 100], 'color': "green"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(
        template="plotly_dark",
        height=250
    )
    
    return fig

# Helper functions for scoring
def calculate_profitability_score(profitability: dict) -> float:
    """Calculate profitability score out of 100"""
    if not profitability:
        return 0
    
    gross_margin = profitability.get('gross_margin', 0) * 100
    net_margin = profitability.get('net_margin', 0) * 100
    roe = profitability.get('roe', 0) * 100
    roa = profitability.get('roa', 0) * 100
    
    # Score based on typical good values
    score = 0
    score += min(gross_margin * 2, 30)  # Up to 30 points for 15%+ gross margin
    score += min(net_margin * 4, 25)   # Up to 25 points for 6.25%+ net margin
    score += min(roe * 2, 25)          # Up to 25 points for 12.5%+ ROE
    score += min(roa * 4, 20)          # Up to 20 points for 5%+ ROA
    
    return min(score, 100)

def calculate_liquidity_score_value(liquidity: dict) -> float:
    """Calculate liquidity score out of 100"""
    if not liquidity:
        return 0
    
    current_ratio = liquidity.get('current_ratio', 0)
    quick_ratio = liquidity.get('quick_ratio', 0)
    cash_ratio = liquidity.get('cash_ratio', 0)
    
    score = 0
    # Current ratio: optimal around 1.5-2.5
    if current_ratio >= 1.5:
        score += min((current_ratio - 1) * 20, 40)
    
    # Quick ratio: optimal around 1.0+
    if quick_ratio >= 1.0:
        score += 30
    elif quick_ratio >= 0.5:
        score += quick_ratio * 30
    
    # Cash ratio: 0.2+ is good
    score += min(cash_ratio * 150, 30)
    
    return min(score, 100)

def calculate_leverage_score_value(leverage: dict) -> float:
    """Calculate leverage score out of 100"""
    if not leverage:
        return 0
    
    debt_equity = leverage.get('debt_to_equity', 0)
    debt_assets = leverage.get('debt_to_assets', 0)
    interest_coverage = leverage.get('interest_coverage', 0)
    
    score = 100  # Start at 100 and deduct for poor metrics
    
    # Penalize high debt ratios
    if debt_equity > 1.0:
        score -= (debt_equity - 1) * 20
    
    if debt_assets > 0.5:
        score -= (debt_assets - 0.5) * 100
    
    # Reward good interest coverage
    if interest_coverage < 2:
        score -= (2 - interest_coverage) * 25
    
    return max(score, 0)

def calculate_efficiency_score_value(efficiency: dict) -> float:
    """Calculate efficiency score out of 100"""
    if not efficiency:
        return 0
    
    asset_turnover = efficiency.get('asset_turnover', 0)
    inventory_turnover = efficiency.get('inventory_turnover', 0)
    receivables_turnover = efficiency.get('receivables_turnover', 0)
    
    score = 0
    score += min(asset_turnover * 50, 40)  # Up to 40 points for 0.8+ asset turnover
    
    if inventory_turnover > 0:
        score += min(inventory_turnover * 2, 30)  # Up to 30 points for 15+ inventory turnover
    else:
        score += 30  # No inventory is fine for service companies
    
    if receivables_turnover > 0:
        score += min(receivables_turnover * 3, 30)  # Up to 30 points for 10+ receivables turnover
    else:
        score += 30  # No receivables is fine for some businesses
    
    return min(score, 100)

def calculate_liquidity_score(liquidity: dict) -> str:
    """Calculate liquidity assessment string"""
    score = calculate_liquidity_score_value(liquidity)
    
    if score >= 80:
        return "Excellent liquidity position"
    elif score >= 65:
        return "Good liquidity position"
    elif score >= 50:
        return "Adequate liquidity position"
    else:
        return "Potential liquidity concerns"

def calculate_leverage_score(leverage: dict) -> str:
    """Calculate leverage assessment string"""
    score = calculate_leverage_score_value(leverage)
    
    if score >= 80:
        return "Conservative debt levels"
    elif score >= 65:
        return "Moderate debt levels"
    elif score >= 50:
        return "Elevated debt levels"
    else:
        return "High debt levels - monitor closely"

def identify_strengths(ratios: dict) -> list:
    """Identify financial strengths"""
    strengths = []
    
    profitability = ratios.get('profitability', {})
    liquidity = ratios.get('liquidity', {})
    leverage = ratios.get('leverage', {})
    efficiency = ratios.get('efficiency', {})
    
    # Check profitability strengths
    if profitability.get('gross_margin', 0) > 0.4:
        strengths.append("Strong gross margins indicating pricing power")
    
    if profitability.get('roe', 0) > 0.15:
        strengths.append("Excellent return on equity")
    
    if profitability.get('net_margin', 0) > 0.10:
        strengths.append("Strong net profit margins")
    
    # Check liquidity strengths
    if liquidity.get('current_ratio', 0) > 2.0:
        strengths.append("Strong liquidity position")
    
    # Check leverage strengths
    if leverage.get('debt_to_equity', 0) < 0.5:
        strengths.append("Conservative debt levels")
    
    if leverage.get('interest_coverage', 0) > 5:
        strengths.append("Strong ability to service debt")
    
    # Check efficiency strengths
    if efficiency.get('asset_turnover', 0) > 1.0:
        strengths.append("Efficient asset utilization")
    
    return strengths[:5]  # Limit to top 5

def identify_weaknesses(ratios: dict) -> list:
    """Identify financial weaknesses"""
    weaknesses = []
    
    profitability = ratios.get('profitability', {})
    liquidity = ratios.get('liquidity', {})
    leverage = ratios.get('leverage', {})
    efficiency = ratios.get('efficiency', {})
    
    # Check profitability weaknesses
    if profitability.get('net_margin', 0) < 0.05:
        weaknesses.append("Low net profit margins")
    
    if profitability.get('roe', 0) < 0.10:
        weaknesses.append("Below-average return on equity")
    
    # Check liquidity weaknesses
    if liquidity.get('current_ratio', 0) < 1.0:
        weaknesses.append("Potential liquidity concerns")
    
    # Check leverage weaknesses
    if leverage.get('debt_to_equity', 0) > 2.0:
        weaknesses.append("High debt levels")
    
    if leverage.get('interest_coverage', 0) < 2.0:
        weaknesses.append("Low interest coverage ratio")
    
    # Check efficiency weaknesses
    if efficiency.get('asset_turnover', 0) < 0.5:
        weaknesses.append("Low asset utilization efficiency")
    
    return weaknesses[:5]  # Limit to top 5

def main():
    """Main function for the financial statements page"""
    
    # Get current selection
    current_stock = get_current_selection()
    
    # Display header
    display_financial_header(current_stock)
    
    # Load financial data
    try:
        financial_data = data_manager.load_financial_data(current_stock)
        
        if not financial_data:
            st.error("Unable to load financial data. Please try another stock or refresh the page.")
            return
        
        # Display financial overview
        display_financial_overview(financial_data, current_stock)
        
        st.markdown("---")
        
        # Create tabs for different financial statements
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Income Statement", "⚖️ Balance Sheet", "💵 Cash Flow", "📈 Financial Ratios", "🏥 Health Assessment"])
        
        with tab1:
            display_income_statement(financial_data)
        
        with tab2:
            display_balance_sheet(financial_data)
        
        with tab3:
            display_cash_flow_statement(financial_data)
        
        with tab4:
            display_financial_ratios(financial_data)
        
        with tab5:
            display_financial_health_assessment(financial_data, current_stock)
        
        # Page footer
        st.markdown("---")
        st.info("""
        💡 **Financial Analysis Tips:**  
        • Use different time periods to spot trends and seasonality  
        • Compare ratios with industry averages for context  
        • Monitor financial health scores regularly  
        • Pay attention to cash flow trends as much as profitability
        """)
        
        # Track page visit
        if 'page_visits' in st.session_state:
            st.session_state.page_visits['financial_statements'] = st.session_state.page_visits.get('financial_statements', 0) + 1
        
    except Exception as e:
        st.error(f"Error loading financial statements page: {e}")
        st.info("Please try refreshing the page or selecting a different stock.")

if __name__ == "__main__":
    main()