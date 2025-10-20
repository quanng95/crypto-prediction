import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit.components.v1 as components

def render_tradingview_chart(df, symbol, interval):
    """
    Render TradingView-style chart with dynamic price label
    
    Args:
        df: DataFrame with OHLCV data
        symbol: Trading symbol (e.g., 'ETHUSDT')
        interval: Timeframe (e.g., '15m', '1h', '4h', '1d')
    """
    
    if df is None or len(df) == 0:
        st.error("No data available for chart")
        return
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.7, 0.3],
        subplot_titles=(f'{symbol} - {interval.upper()}', 'Volume')
    )
    
    # Prepare hover text for candlestick (KHÔNG DÙNG hovertemplate)
    hover_texts = []
    for _, row in df.iterrows():
        text = (
            f"{row['timestamp'].strftime('%Y-%m-%d %H:%M')}<br>"
            f"O: ${row['open']:.2f}<br>"
            f"H: ${row['high']:.2f}<br>"
            f"L: ${row['low']:.2f}<br>"
            f"C: ${row['close']:.2f}"
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
            hovertext=hover_texts,  # ← DÙNG hovertext thay vì hovertemplate
            hoverinfo='text'        # ← Hiển thị text
        ),
        row=1, col=1
    )
    
    # Add volume bars
    colors = ['#26a69a' if row['close'] >= row['open'] else '#ef5350' 
             for _, row in df.iterrows()]
    
    # Prepare hover text for volume
    volume_texts = [f"Vol: {vol:,.0f}" for vol in df['volume']]
    
    fig.add_trace(
        go.Bar(
            x=df['timestamp'],
            y=df['volume'],
            name='Volume',
            marker_color=colors,
            opacity=0.6,
            hovertext=volume_texts,  # ← DÙNG hovertext
            hoverinfo='text'         # ← Hiển thị text
        ),
        row=2, col=1
    )
    
    # Current price line with static label
    current_price = df['close'].iloc[-1]
    is_price_up = df['close'].iloc[-1] >= df['close'].iloc[-2] if len(df) > 1 else True
    price_color = "#26a69a" if is_price_up else "#ef5350"
    
    fig.add_hline(
        y=current_price,
        line_dash="solid",
        line_color=price_color,
        line_width=2,
        annotation_text=f"${current_price:,.2f}",
        annotation_position="right",
        annotation=dict(
            font=dict(size=12, color="#ffffff", family="monospace"),
            bgcolor=price_color,
            bordercolor=price_color,
            borderwidth=2,
            borderpad=5,
            opacity=0.9
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
        modebar_add=['zoom2d', 'zoomIn2d', 'zoomOut2d', 'autoScale2d', 'resetScale2d'],
        hoverlabel=dict(
            bgcolor="#2d2d2d",
            font_size=13,
            font_family="monospace",
            font_color="#ffffff",
            bordercolor="#667eea"
        )
    )
    
    # X-axis spikes (vertical crosshair)
    fig.update_xaxes(
        fixedrange=False,
        showspikes=True,
        spikemode='across',
        spikesnap='cursor',
        spikecolor='rgba(255,255,255,0.5)',
        spikethickness=1,
        spikedash='dot'
    )
    
    # Y-axis spikes (horizontal crosshair) - Price chart
    fig.update_yaxes(
        row=1, col=1,
        fixedrange=False,
        showspikes=True,
        spikemode='across+toaxis',
        spikesnap='cursor',
        spikecolor='rgba(255,255,255,0.5)',
        spikethickness=1,
        spikedash='dot'
    )
    
    # Y-axis spikes (horizontal crosshair) - Volume chart
    fig.update_yaxes(
        row=2, col=1,
        fixedrange=False,
        showspikes=True,
        spikemode='across',
        spikesnap='cursor',
        spikecolor='rgba(255,255,255,0.5)',
        spikethickness=1,
        spikedash='dot'
    )
    
    # Axis labels
    fig.update_xaxes(title_text="Time", row=2, col=1)
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    
    # Convert to HTML
    chart_html = fig.to_html(
        include_plotlyjs='cdn',
        config={
            'scrollZoom': True,
            'displayModeBar': True,
            'displaylogo': False,
            'modeBarButtonsToRemove': ['select2d', 'lasso2d']
        }
    )
    
    # ============================================
    # CUSTOM JAVASCRIPT FOR DYNAMIC PRICE LABEL
    # ============================================
    custom_js = f"""
    <style>
        .dynamic-price-label {{
            position: absolute;
            background-color: #667eea;
            color: white;
            padding: 5px 10px;
            border-radius: 4px;
            font-family: monospace;
            font-size: 13px;
            font-weight: bold;
            pointer-events: none;
            z-index: 10000;
            display: none;
            box-shadow: 0 2px 8px rgba(0,0,0,0.3);
            border: 1px solid rgba(255,255,255,0.2);
            white-space: nowrap;
        }}
        
        .dynamic-price-label.up {{
            background-color: #26a69a;
            border-color: #26a69a;
        }}
        
        .dynamic-price-label.down {{
            background-color: #ef5350;
            border-color: #ef5350;
        }}
    </style>
    
    <div id="dynamic-price-label" class="dynamic-price-label"></div>
    
    <script>
    (function() {{
        var currentPrice = {current_price};
        var priceLabel = null;
        var plotDiv = null;
        var isInitialized = false;
        
        function initCrosshair() {{
            plotDiv = document.querySelector('.plotly-graph-div');
            priceLabel = document.getElementById('dynamic-price-label');
            
            if (!plotDiv || !priceLabel) {{
                setTimeout(initCrosshair, 100);
                return;
            }}
            
            if (isInitialized) return;
            isInitialized = true;
            
            console.log('🎯 TradingView Crosshair Initialized');
            
            // Hover event - Show dynamic price label
            plotDiv.on('plotly_hover', function(data) {{
                try {{
                    if (!data.points || data.points.length === 0) return;
                    
                    var point = data.points[0];
                    var yval = point.y;
                    
                    // For candlestick, use close price
                    if (typeof yval === 'undefined' && point.close) {{
                        yval = point.close;
                    }}
                    
                    if (typeof yval === 'undefined') return;
                    
                    // Get chart dimensions
                    var plotArea = plotDiv.querySelector('.xy');
                    if (!plotArea) return;
                    
                    var rect = plotArea.getBoundingClientRect();
                    var containerRect = plotDiv.getBoundingClientRect();
                    
                    // Get Y-axis range from layout
                    var layout = plotDiv.layout;
                    if (!layout || !layout.yaxis || !layout.yaxis.range) return;
                    
                    var yrange = layout.yaxis.range;
                    var ymin = yrange[0];
                    var ymax = yrange[1];
                    
                    // Calculate Y position in pixels
                    var yPercent = (yval - ymin) / (ymax - ymin);
                    var yPos = rect.bottom - (yPercent * rect.height);
                    
                    // Position label on the right edge
                    var leftPos = rect.right - containerRect.left + 10;
                    var topPos = yPos - containerRect.top - 12;
                    
                    // Update label
                    priceLabel.style.display = 'block';
                    priceLabel.style.left = leftPos + 'px';
                    priceLabel.style.top = topPos + 'px';
                    priceLabel.textContent = '$' + yval.toFixed(2);
                    
                    // Change color based on current price
                    priceLabel.className = 'dynamic-price-label';
                    if (yval >= currentPrice) {{
                        priceLabel.classList.add('up');
                    }} else {{
                        priceLabel.classList.add('down');
                    }}
                }} catch(e) {{
                    console.error('Crosshair error:', e);
                }}
            }});
            
            // Unhover event - Hide label
            plotDiv.on('plotly_unhover', function() {{
                if (priceLabel) {{
                    priceLabel.style.display = 'none';
                }}
            }});
            
            // Relayout event - Update on zoom/pan
            plotDiv.on('plotly_relayout', function() {{
                if (priceLabel && priceLabel.style.display === 'block') {{
                    priceLabel.style.display = 'none';
                }}
            }});
        }}
        
        // Initialize when DOM is ready
        if (document.readyState === 'loading') {{
            document.addEventListener('DOMContentLoaded', initCrosshair);
        }} else {{
            setTimeout(initCrosshair, 100);
        }}
    }})();
    </script>
    """
    
    # Render chart with custom JavaScript
    components.html(chart_html + custom_js, height=750, scrolling=False)
    
    # Metrics below chart
    current = df.iloc[-1]
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Open", f"${current['open']:.2f}")
    with col2:
        st.metric("High", f"${current['high']:.2f}")
    with col3:
        st.metric("Low", f"${current['low']:.2f}")
    with col4:
        st.metric("Close", f"${current['close']:.2f}")
