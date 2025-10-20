import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time

def render_tradingview_chart_realtime(chart_ws):
    """
    Render real-time TradingView-style chart with WebSocket updates
    
    Args:
        chart_ws: ChartWebSocket instance
    """
    
    # Create placeholder for chart
    chart_placeholder = st.empty()
    
    # Fragment that auto-refreshes every 300ms
    @st.fragment(run_every="0.3s")
    def update_chart():
        df = chart_ws.get_dataframe()
        
        if df is None or len(df) == 0:
            chart_placeholder.warning("⏳ Waiting for real-time data...")
            return
        
        # Create chart
        fig = create_chart(df, chart_ws.symbol.upper(), chart_ws.interval)
        
        # Render chart
        chart_placeholder.plotly_chart(
            fig, 
            use_container_width=True, 
            config={
                'scrollZoom': True,
                'displayModeBar': True,
                'displaylogo': False,
                'modeBarButtonsToRemove': ['select2d', 'lasso2d'],
                'responsive': True
            }
        )
    
    update_chart()
    
    # Real-time metrics below chart
    @st.fragment(run_every="0.3s")
    def update_metrics():
        df = chart_ws.get_dataframe()
        
        if df is not None and len(df) > 0:
            current = df.iloc[-1]
            prev = df.iloc[-2] if len(df) > 1 else current
            
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.metric("Open", f"${current['open']:.2f}")
            
            with col2:
                high_change = ((current['high'] / prev['high']) - 1) * 100 if prev['high'] > 0 else 0
                st.metric("High", f"${current['high']:.2f}", 
                         delta=f"{high_change:+.2f}%")
            
            with col3:
                low_change = ((current['low'] / prev['low']) - 1) * 100 if prev['low'] > 0 else 0
                st.metric("Low", f"${current['low']:.2f}", 
                         delta=f"{low_change:+.2f}%")
            
            with col4:
                close_change = ((current['close'] / prev['close']) - 1) * 100 if prev['close'] > 0 else 0
                st.metric("Close", f"${current['close']:.2f}", 
                         delta=f"{close_change:+.2f}%")
            
            with col5:
                vol_change = ((current['volume'] / prev['volume']) - 1) * 100 if prev['volume'] > 0 else 0
                st.metric("Volume", f"{current['volume']:,.0f}", 
                         delta=f"{vol_change:+.1f}%")
    
    update_metrics()


def create_chart(df, symbol, interval):
    """
    Create Plotly candlestick chart with volume
    
    Args:
        df: DataFrame with OHLCV data
        symbol: Trading symbol
        interval: Timeframe
    
    Returns:
        Plotly figure
    """
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.7, 0.3],
        subplot_titles=(f'{symbol} - {interval.upper()} (Real-Time)', 'Volume')
    )
    
    # Prepare hover text for candlestick
    hover_texts = []
    for _, row in df.iterrows():
        text = (
            f"{row['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}<br>"
            f"<b>Open:</b> ${row['open']:.2f}<br>"
            f"<b>High:</b> ${row['high']:.2f}<br>"
            f"<b>Low:</b> ${row['low']:.2f}<br>"
            f"<b>Close:</b> ${row['close']:.2f}<br>"
            f"<b>Volume:</b> {row['volume']:,.0f}"
        )
        hover_texts.append(text)
    
    # Add candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=df['timestamp'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='Price',
            increasing_line_color='#26a69a',
            decreasing_line_color='#ef5350',
            increasing_fillcolor='#26a69a',
            decreasing_fillcolor='#ef5350',
            hovertext=hover_texts,
            hoverinfo='text'
        ),
        row=1, col=1
    )
    
    # Add volume bars
    colors = ['#26a69a' if row['close'] >= row['open'] else '#ef5350' 
             for _, row in df.iterrows()]
    
    volume_texts = [
        f"{row['timestamp'].strftime('%Y-%m-%d %H:%M')}<br>"
        f"<b>Volume:</b> {row['volume']:,.0f}"
        for _, row in df.iterrows()
    ]
    
    fig.add_trace(
        go.Bar(
            x=df['timestamp'],
            y=df['volume'],
            name='Volume',
            marker_color=colors,
            opacity=0.6,
            hovertext=volume_texts,
            hoverinfo='text'
        ),
        row=2, col=1
    )
    
    # Current price line with label
    current_price = df['close'].iloc[-1]
    prev_price = df['close'].iloc[-2] if len(df) > 1 else current_price
    is_price_up = current_price >= prev_price
    price_color = "#26a69a" if is_price_up else "#ef5350"
    price_change = ((current_price / prev_price) - 1) * 100 if prev_price > 0 else 0
    
    fig.add_hline(
        y=current_price,
        line_dash="solid",
        line_color=price_color,
        line_width=2,
        annotation_text=f"${current_price:,.2f} ({price_change:+.2f}%)",
        annotation_position="right",
        annotation=dict(
            font=dict(size=12, color="#ffffff", family="monospace"),
            bgcolor=price_color,
            bordercolor=price_color,
            borderwidth=2,
            borderpad=5,
            opacity=0.95
        ),
        row=1, col=1
    )
    
    # Layout configuration
    fig.update_layout(
        height=700,
        template='plotly_dark',
        xaxis_rangeslider_visible=False,
        showlegend=False,
        hovermode='x unified',
        dragmode='pan',
        transition={'duration': 100},
        hoverlabel=dict(
            bgcolor="#2d2d2d",
            font_size=13,
            font_family="monospace",
            font_color="#ffffff",
            bordercolor="#667eea"
        ),
        paper_bgcolor='#1e1e1e',
        plot_bgcolor='#1e1e1e'
    )
    
    # X-axis configuration with spikes
    fig.update_xaxes(
        fixedrange=False,
        showspikes=True,
        spikemode='across',
        spikesnap='cursor',
        spikecolor='rgba(255,255,255,0.5)',
        spikethickness=1,
        spikedash='dot',
        gridcolor='#2d2d2d'
    )
    
    # Y-axis configuration - Price chart
    fig.update_yaxes(
        row=1, col=1,
        fixedrange=False,
        showspikes=True,
        spikemode='across+toaxis',
        spikesnap='cursor',
        spikecolor='rgba(255,255,255,0.5)',
        spikethickness=1,
        spikedash='dot',
        gridcolor='#2d2d2d'
    )
    
    # Y-axis configuration - Volume chart
    fig.update_yaxes(
        row=2, col=1,
        fixedrange=False,
        showspikes=True,
        spikemode='across',
        spikesnap='cursor',
        spikecolor='rgba(255,255,255,0.5)',
        spikethickness=1,
        spikedash='dot',
        gridcolor='#2d2d2d'
    )
    
    # Axis labels
    fig.update_xaxes(title_text="Time", row=2, col=1)
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    
    return fig
