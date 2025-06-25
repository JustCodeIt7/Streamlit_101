"""
Dashboard Page for Navigation Demos
===================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime, timedelta

st.set_page_config(page_title="Dashboard", page_icon="📊", layout="wide")

st.title("📊 Dashboard")
st.markdown("**Welcome to the interactive dashboard!**")

# Generate sample data
@st.cache_data
def load_dashboard_data():
    """Load sample dashboard data"""
    np.random.seed(42)  # For consistent demo data
    
    # Generate date range
    dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
    
    # Generate sample metrics
    data = {
        'date': dates,
        'users': np.random.poisson(1000, len(dates)) + np.random.randint(50, 200, len(dates)),
        'revenue': np.random.normal(5000, 1000, len(dates)).clip(min=1000),
        'orders': np.random.poisson(150, len(dates)) + np.random.randint(10, 50, len(dates)),
        'conversion_rate': np.random.normal(0.035, 0.01, len(dates)).clip(min=0.01, max=0.08)
    }
    
    df = pd.DataFrame(data)
    df['revenue'] = df['revenue'].round(2)
    df['conversion_rate'] = (df['conversion_rate'] * 100).round(2)
    
    return df

# Load data
df = load_dashboard_data()

# Key Metrics Row
st.subheader("📈 Key Performance Indicators")

col1, col2, col3, col4 = st.columns(4)

# Calculate metrics
total_users = df['users'].sum()
total_revenue = df['revenue'].sum()
total_orders = df['orders'].sum()
avg_conversion = df['conversion_rate'].mean()

# Previous period comparison (last 30 days vs previous 30 days)
recent_data = df.tail(30)
previous_data = df.iloc[-60:-30]

users_change = recent_data['users'].sum() - previous_data['users'].sum()
revenue_change = recent_data['revenue'].sum() - previous_data['revenue'].sum()
orders_change = recent_data['orders'].sum() - previous_data['orders'].sum()
conversion_change = recent_data['conversion_rate'].mean() - previous_data['conversion_rate'].mean()

with col1:
    st.metric(
        label="Total Users",
        value=f"{total_users:,}",
        delta=f"{users_change:,}"
    )

with col2:
    st.metric(
        label="Revenue",
        value=f"${total_revenue:,.0f}",
        delta=f"${revenue_change:,.0f}"
    )

with col3:
    st.metric(
        label="Orders",
        value=f"{total_orders:,}",
        delta=f"{orders_change:,}"
    )

with col4:
    st.metric(
        label="Conversion Rate",
        value=f"{avg_conversion:.2f}%",
        delta=f"{conversion_change:.2f}%"
    )

st.markdown("---")

# Charts Section
st.subheader("📊 Analytics Charts")

# Chart selection
chart_col1, chart_col2 = st.columns([3, 1])

with chart_col2:
    chart_metric = st.selectbox(
        "Select Metric:",
        ["users", "revenue", "orders", "conversion_rate"],
        format_func=lambda x: x.replace('_', ' ').title()
    )
    
    time_period = st.selectbox(
        "Time Period:",
        ["Last 30 Days", "Last 90 Days", "Last 6 Months", "Full Year"]
    )
    
    chart_type = st.radio(
        "Chart Type:",
        ["Line Chart", "Area Chart", "Bar Chart"]
    )

with chart_col1:
    # Filter data based on time period
    if time_period == "Last 30 Days":
        chart_data = df.tail(30)
    elif time_period == "Last 90 Days":
        chart_data = df.tail(90)
    elif time_period == "Last 6 Months":
        chart_data = df.tail(180)
    else:
        chart_data = df
    
    # Create chart based on selection
    if chart_type == "Line Chart":
        fig = px.line(chart_data, x='date', y=chart_metric, 
                     title=f'{chart_metric.replace("_", " ").title()} - {time_period}')
    elif chart_type == "Area Chart":
        fig = px.area(chart_data, x='date', y=chart_metric,
                     title=f'{chart_metric.replace("_", " ").title()} - {time_period}')
    else:  # Bar Chart
        if len(chart_data) > 31:
            # Group by month for bar chart if too many data points
            monthly_data = chart_data.groupby(chart_data['date'].dt.to_period('M'))[chart_metric].sum().reset_index()
            monthly_data['date'] = monthly_data['date'].astype(str)
            fig = px.bar(monthly_data, x='date', y=chart_metric,
                        title=f'Monthly {chart_metric.replace("_", " ").title()}')
        else:
            fig = px.bar(chart_data, x='date', y=chart_metric,
                        title=f'{chart_metric.replace("_", " ").title()} - {time_period}')
    
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# Data Table Section
st.subheader("📋 Recent Data")

# Show recent data with filtering options
col1, col2, col3 = st.columns(3)

with col1:
    show_rows = st.selectbox("Show Rows:", [10, 25, 50, 100], index=1)

with col2:
    sort_by = st.selectbox("Sort By:", 
                          ["date", "users", "revenue", "orders", "conversion_rate"])

with col3:
    sort_order = st.selectbox("Order:", ["Descending", "Ascending"])

# Apply filters and sorting
display_data = df.tail(show_rows).copy()
ascending = sort_order == "Ascending"
display_data = display_data.sort_values(sort_by, ascending=ascending)

# Format the dataframe for display
display_data['revenue'] = display_data['revenue'].apply(lambda x: f"${x:,.2f}")
display_data['conversion_rate'] = display_data['conversion_rate'].apply(lambda x: f"{x:.2f}%")
display_data['date'] = display_data['date'].dt.strftime('%Y-%m-%d')

st.dataframe(display_data, use_container_width=True)

# Navigation Section
st.markdown("---")
st.subheader("🧭 Navigation")

nav_col1, nav_col2, nav_col3 = st.columns(3)

with nav_col1:
    st.page_link("../app1_page_links.py", label="🏠 Home (Page Links)", icon="🏠")
    st.page_link("about.py", label="ℹ️ About", icon="ℹ️")

with nav_col2:
    st.page_link("../app2_switch_page.py", label="🔄 Switch Page Demo", icon="🔄")
    st.page_link("contact.py", label="📞 Contact", icon="📞")

with nav_col3:
    st.page_link("../app3_custom_selector.py", label="🎛️ Custom Selector", icon="🎛️")

# Footer info
st.markdown("---")
st.info("💡 **Dashboard Tip**: This dashboard demonstrates how different navigation methods can lead to the same content pages, providing flexibility in user experience design.")