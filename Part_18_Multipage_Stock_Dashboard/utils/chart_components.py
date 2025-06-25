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
    
    def create_rsi_chart(self, df: pd.DataFrame, rsi_series: pd.Series, title: str = "RSI") -> go.Figure:
        """Create RSI chart with overbought/oversold levels"""
        
        fig = go.Figure()
        
        # RSI line
        fig.add_trace(
            go.Scatter(
                x=rsi_series.index,
                y=rsi_series.values,
                name='RSI',
                line=dict(color=self.colors['rsi'], width=2)
            )
        )
        
        # Reference lines
        fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.7, annotation_text="Overbought")
        fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.7, annotation_text="Oversold")
        fig.add_hline(y=50, line_dash="dash", line_color="gray", opacity=0.5)
        
        fig.update_layout(
            title=title,
            template=self.theme,
            height=300,
            yaxis=dict(range=[0, 100], title="RSI"),
            xaxis_title="Date",
            showlegend=False
        )
        
        return fig
    
    def create_macd_chart(self, df: pd.DataFrame, macd_data: Dict[str, pd.Series], title: str = "MACD") -> go.Figure:
        """Create MACD chart with signal line and histogram"""
        
        fig = go.Figure()
        
        # MACD line
        fig.add_trace(
            go.Scatter(
                x=macd_data['macd'].index,
                y=macd_data['macd'].values,
                name='MACD',
                line=dict(color=self.colors['macd'], width=2)
            )
        )
        
        # Signal line
        fig.add_trace(
            go.Scatter(
                x=macd_data['signal'].index,
                y=macd_data['signal'].values,
                name='Signal',
                line=dict(color=self.colors['signal'], width=2)
            )
        )
        
        # Histogram
        colors = ['green' if val >= 0 else 'red' for val in macd_data['histogram']]
        fig.add_trace(
            go.Bar(
                x=macd_data['histogram'].index,
                y=macd_data['histogram'].values,
                name='Histogram',
                marker_color=colors,
                opacity=0.6
            )
        )
        
        # Zero line
        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
        
        fig.update_layout(
            title=title,
            template=self.theme,
            height=300,
            yaxis_title="MACD",
            xaxis_title="Date",
            showlegend=True
        )
        
        return fig
    
    def create_bollinger_bands_chart(self, df: pd.DataFrame, bb_data: Dict[str, pd.Series], title: str = "Bollinger Bands") -> go.Figure:
        """Create Bollinger Bands chart overlaid on price"""
        
        fig = go.Figure()
        
        # Price line
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['close'],
                name='Close Price',
                line=dict(color='blue', width=2)
            )
        )
        
        # Upper band
        fig.add_trace(
            go.Scatter(
                x=bb_data['upper'].index,
                y=bb_data['upper'].values,
                name='Upper Band',
                line=dict(color='rgba(173, 204, 255, 0.8)', width=1),
                fill=None
            )
        )
        
        # Lower band (with fill)
        fig.add_trace(
            go.Scatter(
                x=bb_data['lower'].index,
                y=bb_data['lower'].values,
                name='Lower Band',
                line=dict(color='rgba(173, 204, 255, 0.8)', width=1),
                fill='tonexty',
                fillcolor='rgba(173, 204, 255, 0.1)'
            )
        )
        
        # Middle line (SMA)
        fig.add_trace(
            go.Scatter(
                x=bb_data['middle'].index,
                y=bb_data['middle'].values,
                name='Middle (SMA)',
                line=dict(color='rgba(173, 204, 255, 0.8)', width=1, dash='dash')
            )
        )
        
        fig.update_layout(
            title=title,
            template=self.theme,
            height=400,
            yaxis_title="Price ($)",
            xaxis_title="Date",
            showlegend=True
        )
        
        return fig
    
    def create_stochastic_chart(self, stoch_data: Dict[str, pd.Series], title: str = "Stochastic Oscillator") -> go.Figure:
        """Create Stochastic Oscillator chart"""
        
        fig = go.Figure()
        
        # %K line
        fig.add_trace(
            go.Scatter(
                x=stoch_data['k'].index,
                y=stoch_data['k'].values,
                name='%K',
                line=dict(color='blue', width=2)
            )
        )
        
        # %D line
        fig.add_trace(
            go.Scatter(
                x=stoch_data['d'].index,
                y=stoch_data['d'].values,
                name='%D',
                line=dict(color='red', width=2)
            )
        )
        
        # Reference lines
        fig.add_hline(y=80, line_dash="dash", line_color="red", opacity=0.7, annotation_text="Overbought")
        fig.add_hline(y=20, line_dash="dash", line_color="green", opacity=0.7, annotation_text="Oversold")
        fig.add_hline(y=50, line_dash="dash", line_color="gray", opacity=0.5)
        
        fig.update_layout(
            title=title,
            template=self.theme,
            height=300,
            yaxis=dict(range=[0, 100], title="Stochastic"),
            xaxis_title="Date",
            showlegend=True
        )
        
        return fig
    
    def create_williams_r_chart(self, williams_series: pd.Series, title: str = "Williams %R") -> go.Figure:
        """Create Williams %R chart"""
        
        fig = go.Figure()
        
        # Williams %R line
        fig.add_trace(
            go.Scatter(
                x=williams_series.index,
                y=williams_series.values,
                name='Williams %R',
                line=dict(color='purple', width=2)
            )
        )
        
        # Reference lines
        fig.add_hline(y=-20, line_dash="dash", line_color="red", opacity=0.7, annotation_text="Overbought")
        fig.add_hline(y=-80, line_dash="dash", line_color="green", opacity=0.7, annotation_text="Oversold")
        fig.add_hline(y=-50, line_dash="dash", line_color="gray", opacity=0.5)
        
        fig.update_layout(
            title=title,
            template=self.theme,
            height=300,
            yaxis=dict(range=[-100, 0], title="Williams %R"),
            xaxis_title="Date",
            showlegend=False
        )
        
        return fig
    
    def create_signal_chart(self, df: pd.DataFrame, signals: List[Dict], title: str = "Trading Signals") -> go.Figure:
        """Create chart showing trading signals on price"""
        
        fig = go.Figure()
        
        # Price line
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['close'],
                name='Close Price',
                line=dict(color='blue', width=2)
            )
        )
        
        # Add buy signals
        buy_signals = [s for s in signals if s['type'] == 'BUY']
        if buy_signals:
            buy_dates = [s['date'] for s in buy_signals]
            buy_prices = [s['price'] for s in buy_signals]
            buy_text = [f"{s['indicator']}: {s['reason']}" for s in buy_signals]
            
            fig.add_trace(
                go.Scatter(
                    x=buy_dates,
                    y=buy_prices,
                    mode='markers',
                    name='Buy Signal',
                    marker=dict(
                        symbol='triangle-up',
                        color='green',
                        size=12,
                        line=dict(color='darkgreen', width=2)
                    ),
                    text=buy_text,
                    hovertemplate='<b>BUY SIGNAL</b><br>%{text}<br>Price: $%{y}<br>Date: %{x}<extra></extra>'
                )
            )
        
        # Add sell signals
        sell_signals = [s for s in signals if s['type'] == 'SELL']
        if sell_signals:
            sell_dates = [s['date'] for s in sell_signals]
            sell_prices = [s['price'] for s in sell_signals]
            sell_text = [f"{s['indicator']}: {s['reason']}" for s in sell_signals]
            
            fig.add_trace(
                go.Scatter(
                    x=sell_dates,
                    y=sell_prices,
                    mode='markers',
                    name='Sell Signal',
                    marker=dict(
                        symbol='triangle-down',
                        color='red',
                        size=12,
                        line=dict(color='darkred', width=2)
                    ),
                    text=sell_text,
                    hovertemplate='<b>SELL SIGNAL</b><br>%{text}<br>Price: $%{y}<br>Date: %{x}<extra></extra>'
                )
            )
        
        fig.update_layout(
            title=title,
            template=self.theme,
            height=400,
            yaxis_title="Price ($)",
            xaxis_title="Date",
            showlegend=True
        )
        
        return fig

    @st.cache_data(ttl=600)  # Cache charts for 10 minutes
    def create_cached_chart(_self, chart_type: str, **kwargs) -> go.Figure:
        """Create cached chart to improve performance"""
        
        chart_methods = {
            'candlestick': _self.create_candlestick_chart,
            'line': _self.create_line_chart,
            'indicator': _self.create_indicator_chart,
            'comparison': _self.create_comparison_chart,
            'correlation': _self.create_correlation_heatmap,
            'metrics': _self.create_metrics_chart,
            'rsi': _self.create_rsi_chart,
            'macd': _self.create_macd_chart,
            'bollinger': _self.create_bollinger_bands_chart,
            'stochastic': _self.create_stochastic_chart,
            'williams': _self.create_williams_r_chart,
            'signals': _self.create_signal_chart
        }
        
        if chart_type in chart_methods:
            return chart_methods[chart_type](**kwargs)
        else:
            raise ValueError(f"Unknown chart type: {chart_type}")


# Global instance
chart_components = ChartComponents()