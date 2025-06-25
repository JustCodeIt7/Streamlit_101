"""
Sentiment Analysis Page
======================

Comprehensive news and social media sentiment analysis with market impact correlation.
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
from collections import Counter
import re

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

# Import configuration and utilities
from config.settings import STREAMLIT_CONFIG
from config.stock_symbols import get_stock_info
from utils.data_manager import data_manager
from utils.chart_components import chart_components

# Configure page
st.set_page_config(**STREAMLIT_CONFIG)

def get_current_selection():
    """Get current stock selection from session state"""
    return st.session_state.get('selected_stock', 'AAPL')

def display_sentiment_header(symbol: str):
    """Display sentiment analysis header"""
    stock_info = get_stock_info(symbol)
    
    if stock_info:
        st.title(f"📰 Sentiment Analysis - {stock_info['name']} ({symbol})")
        st.markdown(f"**Sector:** {stock_info.get('sector', 'N/A')} | **Industry:** {stock_info.get('industry', 'N/A')}")
    else:
        st.title(f"📰 Sentiment Analysis - {symbol}")

def display_sentiment_overview(sentiment_data: dict, symbol: str):
    """Display sentiment overview metrics"""
    
    if not sentiment_data or 'summary_stats' not in sentiment_data:
        st.warning("Sentiment data not available")
        return
    
    stats = sentiment_data['summary_stats']
    news_stats = stats.get('news_sentiment', {})
    social_stats = stats.get('social_sentiment', {})
    trends = stats.get('trends', {})
    
    st.subheader("📊 Sentiment Overview")
    
    # Key metrics row
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        overall_sentiment = news_stats.get('overall_avg', 0)
        sentiment_emoji = "😊" if overall_sentiment > 0.2 else "😐" if overall_sentiment > -0.2 else "😞"
        st.metric(
            "Overall News Sentiment", 
            f"{sentiment_emoji} {overall_sentiment:.2f}",
            f"{trends.get('sentiment_momentum', 0):+.2f} momentum"
        )
    
    with col2:
        social_sentiment = social_stats.get('overall_avg', 0)
        social_emoji = "🚀" if social_sentiment > 0.3 else "📈" if social_sentiment > 0 else "📉"
        st.metric(
            "Social Media Sentiment",
            f"{social_emoji} {social_sentiment:.2f}",
            f"{social_stats.get('total_volume', 0):,} mentions"
        )
    
    with col3:
        positive_ratio = news_stats.get('positive_ratio', 0)
        st.metric(
            "Positive News Ratio",
            f"{positive_ratio:.1%}",
            "Last 30 days"
        )
    
    with col4:
        news_frequency = trends.get('news_frequency', 0)
        st.metric(
            "News Frequency",
            f"{news_frequency} articles",
            "Last 7 days"
        )
    
    with col5:
        if 'market_impact' in sentiment_data:
            correlation = sentiment_data['market_impact'].get('sentiment_price_correlation', 0)
            st.metric(
                "Price Correlation",
                f"{correlation:.2f}",
                f"{abs(correlation)*100:.0f}% strength"
            )

def display_news_sentiment_analysis(sentiment_data: dict):
    """Display detailed news sentiment analysis"""
    
    st.subheader("📰 News Sentiment Analysis")
    
    if 'news_articles' not in sentiment_data:
        st.warning("News sentiment data not available")
        return
    
    news_articles = sentiment_data['news_articles']
    
    # Time period selection
    col1, col2 = st.columns([1, 3])
    
    with col1:
        time_period = st.selectbox(
            "Analysis Period",
            ["7 days", "14 days", "30 days", "90 days"],
            index=2,
            key="news_period"
        )
        
        sentiment_filter = st.selectbox(
            "Sentiment Filter",
            ["All", "Positive", "Neutral", "Negative"],
            key="sentiment_filter"
        )
    
    # Filter data based on selections
    days = int(time_period.split()[0])
    cutoff_date = datetime.now() - timedelta(days=days)
    
    filtered_articles = [
        article for article in news_articles
        if datetime.strptime(article['date'], '%Y-%m-%d') >= cutoff_date
    ]
    
    # Apply sentiment filter
    if sentiment_filter != "All":
        if sentiment_filter == "Positive":
            filtered_articles = [a for a in filtered_articles if a['sentiment_score'] > 0.2]
        elif sentiment_filter == "Negative":
            filtered_articles = [a for a in filtered_articles if a['sentiment_score'] < -0.2]
        else:  # Neutral
            filtered_articles = [a for a in filtered_articles if -0.2 <= a['sentiment_score'] <= 0.2]
    
    if not filtered_articles:
        st.warning("No articles found for the selected criteria")
        return
    
    # News sentiment trend chart
    with col2:
        fig = create_sentiment_trend_chart(filtered_articles)
        st.plotly_chart(fig, use_container_width=True)
    
    # Sentiment distribution
    col1, col2 = st.columns(2)
    
    with col1:
        fig = create_sentiment_distribution_chart(filtered_articles)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = create_source_credibility_chart(filtered_articles)
        st.plotly_chart(fig, use_container_width=True)
    
    # Recent articles table
    st.markdown("### 📄 Recent Articles")
    
    display_count = st.slider("Number of articles to display", 5, 20, 10, key="article_count")
    
    articles_df = pd.DataFrame(filtered_articles[:display_count])
    
    if not articles_df.empty:
        # Format for display
        display_df = articles_df[['date', 'headline', 'source', 'sentiment_score', 'confidence']].copy()
        display_df['sentiment_score'] = display_df['sentiment_score'].apply(lambda x: f"{x:.2f}")
        display_df['confidence'] = display_df['confidence'].apply(lambda x: f"{x:.1%}")
        
        # Add sentiment emoji
        display_df['sentiment'] = display_df['sentiment_score'].apply(
            lambda x: "😊 Positive" if float(x) > 0.2 else "😐 Neutral" if float(x) > -0.2 else "😞 Negative"
        )
        
        display_df.columns = ['Date', 'Headline', 'Source', 'Score', 'Confidence', 'Sentiment']
        
        st.dataframe(display_df, use_container_width=True)

def display_social_media_sentiment(sentiment_data: dict):
    """Display social media sentiment analysis"""
    
    st.subheader("📱 Social Media Sentiment")
    
    if 'social_media' not in sentiment_data:
        st.warning("Social media sentiment data not available")
        return
    
    social_data = sentiment_data['social_media']
    
    # Social media platform analysis
    platform_stats = {}
    for post in social_data:
        platform = post['platform']
        if platform not in platform_stats:
            platform_stats[platform] = {
                'count': 0,
                'total_sentiment': 0,
                'total_engagement': 0
            }
        platform_stats[platform]['count'] += 1
        platform_stats[platform]['total_sentiment'] += post['sentiment_score']
        platform_stats[platform]['total_engagement'] += post['engagement']
    
    # Calculate averages
    for platform in platform_stats:
        stats = platform_stats[platform]
        stats['avg_sentiment'] = stats['total_sentiment'] / stats['count']
        stats['avg_engagement'] = stats['total_engagement'] / stats['count']
    
    # Display platform metrics
    col1, col2, col3 = st.columns(3)
    
    platforms = ['Twitter', 'Reddit', 'StockTwits']
    for i, platform in enumerate(platforms):
        with [col1, col2, col3][i]:
            if platform in platform_stats:
                stats = platform_stats[platform]
                sentiment_emoji = "🚀" if stats['avg_sentiment'] > 0.3 else "📈" if stats['avg_sentiment'] > 0 else "📉"
                st.metric(
                    f"{platform} Sentiment",
                    f"{sentiment_emoji} {stats['avg_sentiment']:.2f}",
                    f"{stats['count']} posts, {stats['avg_engagement']:.0f} avg engagement"
                )
    
    # Social media trends
    col1, col2 = st.columns(2)
    
    with col1:
        fig = create_social_sentiment_timeline(social_data)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = create_engagement_sentiment_scatter(social_data)
        st.plotly_chart(fig, use_container_width=True)

def display_market_impact_analysis(sentiment_data: dict, symbol: str):
    """Display market impact and correlation analysis"""
    
    st.subheader("💹 Market Impact Analysis")
    
    if 'market_impact' not in sentiment_data:
        st.warning("Market impact data not available")
        return
    
    market_impact = sentiment_data['market_impact']
    
    # Market impact metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        correlation = market_impact.get('sentiment_price_correlation', 0)
        correlation_strength = "Strong" if abs(correlation) > 0.5 else "Moderate" if abs(correlation) > 0.3 else "Weak"
        st.metric(
            "Sentiment-Price Correlation",
            f"{correlation:.3f}",
            f"{correlation_strength} relationship"
        )
    
    with col2:
        accuracy = market_impact.get('prediction_accuracy', 0)
        st.metric(
            "Prediction Accuracy",
            f"{accuracy:.1f}%",
            "Historical accuracy"
        )
    
    with col3:
        sensitivity = market_impact.get('market_sensitivity', 0)
        st.metric(
            "Market Sensitivity",
            f"{sensitivity:.2f}",
            "Price response to sentiment"
        )
    
    with col4:
        major_events = market_impact.get('major_sentiment_events', [])
        st.metric(
            "Major Events",
            f"{len(major_events)}",
            "Last 90 days"
        )
    
    # Major sentiment events
    if major_events:
        st.markdown("### 🎯 Major Sentiment Events")
        
        events_df = pd.DataFrame(major_events)
        events_df['sentiment'] = events_df['sentiment'].apply(lambda x: f"{x:.2f}")
        events_df['impact_type'] = events_df['impact_type'].apply(
            lambda x: f"🟢 {x.title()}" if x == 'positive' else f"🔴 {x.title()}"
        )
        
        events_df.columns = ['Date', 'Headline', 'Sentiment Score', 'Impact Type']
        st.dataframe(events_df, use_container_width=True)
    
    # Sentiment vs Price correlation chart
    try:
        stock_data = data_manager.load_stock_data(symbol)
        if not stock_data.empty:
            fig = create_sentiment_price_correlation_chart(sentiment_data, stock_data, symbol)
            st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.warning("Unable to load price correlation chart")

def display_sentiment_insights(sentiment_data: dict, symbol: str):
    """Display actionable sentiment insights"""
    
    st.subheader("💡 Sentiment Insights & Recommendations")
    
    if 'summary_stats' not in sentiment_data:
        st.warning("Sentiment insights not available")
        return
    
    stats = sentiment_data['summary_stats']
    news_stats = stats.get('news_sentiment', {})
    trends = stats.get('trends', {})
    
    # Generate insights
    insights = []
    recommendations = []
    
    # News sentiment insights
    overall_sentiment = news_stats.get('overall_avg', 0)
    momentum = trends.get('sentiment_momentum', 0)
    
    if overall_sentiment > 0.3:
        insights.append("📈 **Strong positive sentiment** in recent news coverage")
        recommendations.append("Consider monitoring for potential upward price momentum")
    elif overall_sentiment < -0.3:
        insights.append("📉 **Significant negative sentiment** in recent news coverage")
        recommendations.append("Exercise caution; consider defensive strategies")
    
    if momentum > 0.2:
        insights.append("🚀 **Improving sentiment momentum** over recent periods")
        recommendations.append("Positive sentiment trend may support price appreciation")
    elif momentum < -0.2:
        insights.append("⚠️ **Declining sentiment momentum** detected")
        recommendations.append("Watch for potential sentiment-driven selling pressure")
    
    # Market impact insights
    if 'market_impact' in sentiment_data:
        correlation = sentiment_data['market_impact'].get('sentiment_price_correlation', 0)
        if abs(correlation) > 0.4:
            insights.append(f"🔗 **Strong sentiment-price correlation** ({correlation:.2f})")
            recommendations.append("Sentiment changes likely to impact stock price movements")
    
    # Volatility insights
    volatility = trends.get('volatility', 0)
    if volatility > 0.5:
        insights.append("⚡ **High sentiment volatility** indicates market uncertainty")
        recommendations.append("Prepare for potential price swings based on news flow")
    
    # Display insights and recommendations
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🔍 Key Insights")
        if insights:
            for insight in insights:
                st.markdown(f"• {insight}")
        else:
            st.info("No significant sentiment patterns detected")
    
    with col2:
        st.markdown("#### 📋 Recommendations")
        if recommendations:
            for rec in recommendations:
                st.markdown(f"• {rec}")
        else:
            st.info("Monitor sentiment trends for trading opportunities")
    
    # Sentiment alert thresholds
    st.markdown("### 🚨 Sentiment Alert Levels")
    
    alert_col1, alert_col2, alert_col3 = st.columns(3)
    
    with alert_col1:
        if overall_sentiment > 0.5:
            st.success("🟢 **BULLISH SENTIMENT**\nStrong positive news flow")
        elif overall_sentiment > 0.2:
            st.info("🔵 **POSITIVE SENTIMENT**\nModerately positive coverage")
        elif overall_sentiment > -0.2:
            st.warning("🟡 **NEUTRAL SENTIMENT**\nMixed or balanced coverage")
        elif overall_sentiment > -0.5:
            st.warning("🟠 **NEGATIVE SENTIMENT**\nConcerning news trends")
        else:
            st.error("🔴 **BEARISH SENTIMENT**\nSignificant negative coverage")
    
    with alert_col2:
        news_frequency = trends.get('news_frequency', 0)
        if news_frequency > 15:
            st.info("📰 **HIGH NEWS ACTIVITY**\nAbove average coverage")
        elif news_frequency < 3:
            st.warning("📰 **LOW NEWS ACTIVITY**\nLimited recent coverage")
        else:
            st.success("📰 **NORMAL NEWS ACTIVITY**\nTypical coverage levels")
    
    with alert_col3:
        if momentum > 0.3:
            st.success("📈 **IMPROVING TREND**\nSentiment getting better")
        elif momentum < -0.3:
            st.error("📉 **DECLINING TREND**\nSentiment getting worse")
        else:
            st.info("📊 **STABLE TREND**\nConsistent sentiment levels")

# Chart creation functions
def create_sentiment_trend_chart(articles: list) -> go.Figure:
    """Create sentiment trend over time chart"""
    
    # Group articles by date and calculate daily average sentiment
    daily_sentiment = {}
    for article in articles:
        date = article['date']
        if date not in daily_sentiment:
            daily_sentiment[date] = []
        daily_sentiment[date].append(article['sentiment_score'])
    
    # Calculate daily averages
    dates = []
    sentiments = []
    for date in sorted(daily_sentiment.keys()):
        dates.append(date)
        sentiments.append(np.mean(daily_sentiment[date]))
    
    fig = go.Figure()
    
    # Sentiment line
    fig.add_trace(go.Scatter(
        x=dates,
        y=sentiments,
        mode='lines+markers',
        name='Daily Sentiment',
        line=dict(color='blue', width=2),
        marker=dict(size=6)
    ))
    
    # Add reference lines
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    fig.add_hline(y=0.2, line_dash="dot", line_color="green", opacity=0.5, annotation_text="Positive")
    fig.add_hline(y=-0.2, line_dash="dot", line_color="red", opacity=0.5, annotation_text="Negative")
    
    fig.update_layout(
        title="News Sentiment Trend",
        xaxis_title="Date",
        yaxis_title="Sentiment Score",
        template="plotly_dark",
        height=300,
        yaxis=dict(range=[-1, 1])
    )
    
    return fig

def create_sentiment_distribution_chart(articles: list) -> go.Figure:
    """Create sentiment distribution histogram"""
    
    sentiments = [article['sentiment_score'] for article in articles]
    
    fig = go.Figure(data=[go.Histogram(
        x=sentiments,
        nbinsx=20,
        marker_color='skyblue',
        opacity=0.7,
        name='Sentiment Distribution'
    )])
    
    fig.add_vline(x=0, line_dash="dash", line_color="gray", annotation_text="Neutral")
    fig.add_vline(x=0.2, line_dash="dot", line_color="green", annotation_text="Positive")
    fig.add_vline(x=-0.2, line_dash="dot", line_color="red", annotation_text="Negative")
    
    fig.update_layout(
        title="Sentiment Distribution",
        xaxis_title="Sentiment Score",
        yaxis_title="Number of Articles",
        template="plotly_dark",
        height=300
    )
    
    return fig

def create_source_credibility_chart(articles: list) -> go.Figure:
    """Create source credibility analysis chart"""
    
    source_data = {}
    for article in articles:
        source = article['source']
        if source not in source_data:
            source_data[source] = {
                'credibility': article.get('credibility', 0.75),
                'sentiment': [],
                'count': 0
            }
        source_data[source]['sentiment'].append(article['sentiment_score'])
        source_data[source]['count'] += 1
    
    sources = []
    credibilities = []
    avg_sentiments = []
    article_counts = []
    
    for source, data in source_data.items():
        sources.append(source)
        credibilities.append(data['credibility'])
        avg_sentiments.append(np.mean(data['sentiment']))
        article_counts.append(data['count'])
    
    fig = go.Figure(data=go.Scatter(
        x=credibilities,
        y=avg_sentiments,
        mode='markers+text',
        text=sources,
        textposition="top center",
        marker=dict(
            size=[count * 3 for count in article_counts],
            color=avg_sentiments,
            colorscale='RdYlGn',
            showscale=True,
            colorbar=dict(title="Avg Sentiment")
        ),
        name='Sources'
    ))
    
    fig.update_layout(
        title="Source Credibility vs Sentiment",
        xaxis_title="Source Credibility",
        yaxis_title="Average Sentiment",
        template="plotly_dark",
        height=300
    )
    
    return fig

def create_social_sentiment_timeline(social_data: list) -> go.Figure:
    """Create social media sentiment timeline"""
    
    # Group by date and platform
    daily_data = {}
    for post in social_data:
        date = post['date']
        platform = post['platform']
        if date not in daily_data:
            daily_data[date] = {}
        if platform not in daily_data[date]:
            daily_data[date][platform] = []
        daily_data[date][platform].append(post['sentiment_score'])
    
    fig = go.Figure()
    
    platforms = ['Twitter', 'Reddit', 'StockTwits']
    colors = ['#1DA1F2', '#FF4500', '#1ABC9C']
    
    for i, platform in enumerate(platforms):
        dates = []
        sentiments = []
        
        for date in sorted(daily_data.keys()):
            if platform in daily_data[date]:
                dates.append(date)
                sentiments.append(np.mean(daily_data[date][platform]))
        
        if dates and sentiments:
            fig.add_trace(go.Scatter(
                x=dates,
                y=sentiments,
                mode='lines+markers',
                name=platform,
                line=dict(color=colors[i], width=2)
            ))
    
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    
    fig.update_layout(
        title="Social Media Sentiment Timeline",
        xaxis_title="Date",
        yaxis_title="Average Sentiment",
        template="plotly_dark",
        height=300
    )
    
    return fig

def create_engagement_sentiment_scatter(social_data: list) -> go.Figure:
    """Create engagement vs sentiment scatter plot"""
    
    sentiments = [post['sentiment_score'] for post in social_data]
    engagements = [post['engagement'] for post in social_data]
    platforms = [post['platform'] for post in social_data]
    
    fig = go.Figure()
    
    platform_colors = {'Twitter': '#1DA1F2', 'Reddit': '#FF4500', 'StockTwits': '#1ABC9C'}
    
    for platform in ['Twitter', 'Reddit', 'StockTwits']:
        platform_sentiments = [sentiments[i] for i, p in enumerate(platforms) if p == platform]
        platform_engagements = [engagements[i] for i, p in enumerate(platforms) if p == platform]
        
        if platform_sentiments and platform_engagements:
            fig.add_trace(go.Scatter(
                x=platform_sentiments,
                y=platform_engagements,
                mode='markers',
                name=platform,
                marker=dict(
                    color=platform_colors[platform],
                    size=8,
                    opacity=0.6
                )
            ))
    
    fig.update_layout(
        title="Engagement vs Sentiment",
        xaxis_title="Sentiment Score",
        yaxis_title="Engagement",
        template="plotly_dark",
        height=300
    )
    
    return fig

def create_sentiment_price_correlation_chart(sentiment_data: dict, stock_data: pd.DataFrame, symbol: str) -> go.Figure:
    """Create sentiment vs price correlation chart"""
    
    # Get recent stock data (last 30 days)
    recent_stock = stock_data.tail(30).copy()
    recent_stock['daily_return'] = recent_stock['close'].pct_change()
    
    # Get news articles
    news_articles = sentiment_data.get('news_articles', [])
    
    # Group sentiment by date
    daily_sentiment = {}
    for article in news_articles:
        date = article['date']
        if date not in daily_sentiment:
            daily_sentiment[date] = []
        daily_sentiment[date].append(article['sentiment_score'])
    
    # Calculate daily averages
    for date in daily_sentiment:
        daily_sentiment[date] = np.mean(daily_sentiment[date])
    
    # Create subplot
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=(f'{symbol} Price', 'Daily Sentiment'),
        row_width=[0.3, 0.7]
    )
    
    # Price chart
    fig.add_trace(
        go.Scatter(
            x=recent_stock.index,
            y=recent_stock['close'],
            mode='lines',
            name='Price',
            line=dict(color='blue', width=2)
        ),
        row=1, col=1
    )
    
    # Sentiment chart
    sentiment_dates = list(daily_sentiment.keys())
    sentiment_values = list(daily_sentiment.values())
    
    fig.add_trace(
        go.Scatter(
            x=sentiment_dates,
            y=sentiment_values,
            mode='lines+markers',
            name='Sentiment',
            line=dict(color='green', width=2),
            marker=dict(size=6)
        ),
        row=2, col=1
    )
    
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5, row=2, col=1)
    
    fig.update_layout(
        title="Sentiment vs Price Correlation",
        template="plotly_dark",
        height=500,
        showlegend=True
    )
    
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="Sentiment", row=2, col=1)
    
    return fig

def main():
    """Main function for the sentiment analysis page"""
    
    # Get current selection
    current_stock = get_current_selection()
    
    # Display header
    display_sentiment_header(current_stock)
    
    # Load sentiment data
    try:
        sentiment_data = data_manager.load_news_data(current_stock)
        
        if not sentiment_data:
            st.error("Unable to load sentiment data. Please try another stock or refresh the page.")
            return
        
        # Display sentiment overview
        display_sentiment_overview(sentiment_data, current_stock)
        
        st.markdown("---")
        
        # Create tabs for different analyses
        tab1, tab2, tab3, tab4 = st.tabs(["📰 News Analysis", "📱 Social Media", "💹 Market Impact", "💡 Insights"])
        
        with tab1:
            display_news_sentiment_analysis(sentiment_data)
        
        with tab2:
            display_social_media_sentiment(sentiment_data)
        
        with tab3:
            display_market_impact_analysis(sentiment_data, current_stock)
        
        with tab4:
            display_sentiment_insights(sentiment_data, current_stock)
        
        # Page footer
        st.markdown("---")
        st.info("""
        💡 **Sentiment Analysis Tips:**  
        • Monitor sentiment momentum for early trend detection  
        • Consider source credibility when evaluating news impact  
        • Use correlation analysis to gauge market sensitivity  
        • Combine sentiment with technical analysis for better decisions
        """)
        
        # Track page visit
        if 'page_visits' in st.session_state:
            st.session_state.page_visits['sentiment'] = st.session_state.page_visits.get('sentiment', 0) + 1
        
    except Exception as e:
        st.error(f"Error loading sentiment analysis page: {e}")
        st.info("Please try refreshing the page or selecting a different stock.")

if __name__ == "__main__":
    main()