"""
Chart Components for Stock Analysis Dashboard
Reusable Plotly chart components with consistent styling
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import streamlit as st

# Import configuration
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from config.settings import CHART_CONFIG


class ChartComponents:
    """Reusable chart components for the dashboard"""
    
    def __init__(self):
        self.config = CHART_CONFIG
        self.theme = self.config['theme']
        self.height = self.config['height']
        self.colors = self.config['colors']
    
    def create_candlestick_chart(
        self, 
        df: pd.DataFrame, 
        title: str = "Stock Price",
        show_volume: bool = True,
        indicators: Optional[List[str]] = None
    ) -> go.Figure:
        """
        Create candlestick chart with optional volume and indicators
        
        Args:
            df: DataFrame with OHLCV data
            title: Chart title
            show_volume: Whether to show volume subplot
            indicators: List of indicators to overlay
            
        Returns:
            Plotly figure object
        """
        
        # Determine subplot configuration
        if show_volume:
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.03,
                subplot_titles=(title, 'Volume'),
                row_width=[0.2, 0.7]
            )
        else:
            fig = go.Figure()
        
        # Main candlestick chart
        candlestick = go.Candlestick(
            x=df.index,
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name="Price",
            increasing_line_color=self.colors['increasing'],
            decreasing_line_color=self.colors['decreasing'],
            increasing_fillcolor=self.colors['increasing'],
            decreasing_fillcolor=self.colors['decreasing']
        )
        
        if show_volume:
            fig.add_trace(candlestick, row=1, col=1)
        else:
            fig.add_trace(candlestick)
        
        # Add indicators if specified
        if indicators:
            self._add_indicators_to_chart(fig, df, indicators, show_volume)
        
        # Add volume subplot if requested
        if show_volume:
            # Color volume bars based on price direction
            colors = []
            for i in range(len(df)):
                if df['close'].iloc[i] > df['open'].iloc[i]:
                    colors.append(self.colors['increasing'])
                else:
                    colors.append(self.colors['decreasing'])
            
            volume_trace = go.Bar(
                x=df.index,
                y=df['volume'],
                name="Volume",
                marker_color=colors,
                opacity=0.7
            )
            fig.add_trace(volume_trace, row=2, col=1)
        
        # Update layout
        self._update_chart_layout(fig, title, show_volume)
        
        return fig
    
    def _add_indicators_to_chart(
        self, 
        fig: go.Figure, 
        df: pd.DataFrame, 
        indicators: List[str],
        show_volume: bool
    ):
        """Add technical indicators to the chart"""
        
        row = 1  # Main chart row
        
        for indicator in indicators:
            if indicator == 'sma_20' and 'sma_20' in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df['sma_20'],
                        name='SMA 20',
                        line=dict(color=self.colors['sma'], width=2)
                    ),
                    row=row, col=1
                )
            
            elif indicator == 'sma_50' and 'sma_50' in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df['sma_50'],
                        name='SMA 50',
                        line=dict(color=self.colors['ema'], width=2)
                    ),
                    row=row, col=1
                )
            
            elif indicator == 'ema_12' and 'ema_12' in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df['ema_12'],
                        name='EMA 12',
                        line=dict(color='orange', width=1, dash='dash')
                    ),
                    row=row, col=1
                )
    
    def create_line_chart(
        self,
        df: pd.DataFrame,
        y_column: str,
        title: str,
        color: Optional[str] = None
    ) -> go.Figure:
        """Create a simple line chart"""
        
        fig = go.Figure()
        
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df[y_column],
                mode='lines',
                name=y_column.replace('_', ' ').title(),
                line=dict(
                    color=color or self.colors['increasing'],
                    width=2
                )
            )
        )
        
        self._update_chart_layout(fig, title, False)
        
        return fig
    
    def create_indicator_chart(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, pd.Series],
        title: str,
        y_range: Optional[Tuple[float, float]] = None
    ) -> go.Figure:
        """Create chart for oscillator-type indicators"""
        
        fig = go.Figure()
        
        for name, series in indicators.items():
            fig.add_trace(
                go.Scatter(
                    x=series.index,
                    y=series.values,
                    name=name,
                    line=dict(width=2)
                )
            )
        
        # Add reference lines for common oscillators
        if y_range:
            fig.add_hline(y=y_range[0], line_dash="dash", line_color="gray", opacity=0.5)
            fig.add_hline(y=y_range[1], line_dash="dash", line_color="gray", opacity=0.5)
            
            # Add middle line for RSI, Stochastic, etc.
            if y_range == (0, 100):
                fig.add_hline(y=50, line_dash="dash", line_color="gray", opacity=0.3)
                fig.add_hline(y=30, line_dash="dash", line_color="red", opacity=0.5)
                fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5)
        
        self._update_chart_layout(fig, title, False)
        
        if y_range:
            fig.update_yaxes(range=y_range)
        
        return fig
    
    def create_volume_profile_chart(self, df: pd.DataFrame) -> go.Figure:
        """Create volume profile chart"""
        
        # Calculate volume profile
        price_range = np.linspace(df['low'].min(), df['high'].max(), 50)
        volume_profile = []
        
        for i in range(len(price_range) - 1):
            low_price = price_range[i]
            high_price = price_range[i + 1]
            
            # Find volume traded in this price range
            mask = (df['low'] <= high_price) & (df['high'] >= low_price)
            volume = df.loc[mask, 'volume'].sum()
            
            volume_profile.append({
                'price': (low_price + high_price) / 2,
                'volume': volume
            })
        
        profile_df = pd.DataFrame(volume_profile)
        
        fig = go.Figure()
        
        fig.add_trace(
            go.Bar(
                x=profile_df['volume'],
                y=profile_df['price'],
                orientation='h',
                name='Volume Profile',
                marker_color=self.colors['volume'],
                opacity=0.7
            )
        )
        
        fig.update_layout(
            title="Volume Profile",
            xaxis_title="Volume",
            yaxis_title="Price",
            template=self.theme,
            height=self.height
        )
        
        return fig
    
    def create_comparison_chart(
        self,
        data: Dict[str, pd.Series],
        title: str,
        normalize: bool = True
    ) -> go.Figure:
        """Create comparison chart for multiple stocks/indices"""
        
        fig = go.Figure()
        
        for name, series in data.items():
            if normalize:
                # Normalize to percentage change from first value
                normalized_series = (series / series.iloc[0] - 1) * 100
                y_data = normalized_series
                y_title = "Percentage Change (%)"
            else:
                y_data = series
                y_title = "Price"
            
            fig.add_trace(
                go.Scatter(
                    x=series.index,
                    y=y_data,
                    name=name,
                    mode='lines',
                    line=dict(width=2)
                )
            )
        
        fig.update_layout(
            title=title,
            xaxis_title="Date",
            yaxis_title=y_title,
            template=self.theme,
            height=self.height,
            hovermode='x unified'
        )
        
        return fig
    
    def create_correlation_heatmap(self, correlation_matrix: pd.DataFrame) -> go.Figure:
        """Create correlation heatmap"""
        
        fig = go.Figure(data=go.Heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns,
            y=correlation_matrix.index,
            colorscale='RdBu',
            zmid=0,
            text=correlation_matrix.round(2).values,
            texttemplate="%{text}",
            textfont={"size": 12},
            hovertemplate='%{x} vs %{y}<br>Correlation: %{z}<extra></extra>'
        ))
        
        fig.update_layout(
            title="Stock Correlation Matrix",
            template=self.theme,
            height=500
        )
        
        return fig
    
    def create_metrics_chart(
        self,
        metrics: Dict[str, float],
        title: str,
        chart_type: str = 'bar'
    ) -> go.Figure:
        """Create metrics visualization (bar or gauge charts)"""
        
        if chart_type == 'bar':
            fig = go.Figure(data=go.Bar(
                x=list(metrics.keys()),
                y=list(metrics.values()),
                marker_color=self.colors['increasing'],
                text=[f"{v:.2f}" for v in metrics.values()],
                textposition='auto'
            ))
            
            fig.update_layout(
                title=title,
                template=self.theme,
                height=400
            )
        
        else:  # gauge chart for single metric
            if len(metrics) == 1:
                metric_name, metric_value = list(metrics.items())[0]
                
                fig = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=metric_value,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': metric_name},
                    gauge={
                        'axis': {'range': [None, 100]},
                        'bar': {'color': self.colors['increasing']},
                        'steps': [
                            {'range': [0, 30], 'color': "lightgray"},
                            {'range': [30, 70], 'color': "gray"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 90
                        }
                    }
                ))
                
                fig.update_layout(
                    template=self.theme,
                    height=400
                )
        
        return fig
    
    def _update_chart_layout(self, fig: go.Figure, title: str, has_volume: bool):
        """Update chart layout with consistent styling"""
        
        fig.update_layout(
            title=title,
            template=self.theme,
            height=self.height,
            showlegend=True,
            xaxis_rangeslider_visible=False,
            hovermode='x unified'
        )
        
        # Update x-axis
        fig.update_xaxes(
            title_text="Date",
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(128,128,128,0.2)'
        )
        
        # Update y-axes
        if has_volume:
            fig.update_yaxes(
                title_text="Price ($)",
                row=1, col=1,
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(128,128,128,0.2)'
            )
            fig.update_yaxes(
                title_text="Volume",
                row=2, col=1,
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(128,128,128,0.2)'
            )
        else:
            fig.update_yaxes(
                title_text="Price ($)",
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(128,128,128,0.2)'
            )
    
    @st.cache_data(ttl=600)  # Cache charts for 10 minutes
    def create_cached_chart(_self, chart_type: str, **kwargs) -> go.Figure:
        """Create cached chart to improve performance"""
        
        chart_methods = {
            'candlestick': _self.create_candlestick_chart,
            'line': _self.create_line_chart,
            'indicator': _self.create_indicator_chart,
            'comparison': _self.create_comparison_chart,
            'correlation': _self.create_correlation_heatmap,
            'metrics': _self.create_metrics_chart
        }
        
        if chart_type in chart_methods:
            return chart_methods[chart_type](**kwargs)
        else:
            raise ValueError(f"Unknown chart type: {chart_type}")


# Global instance
chart_components = ChartComponents()