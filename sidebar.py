import streamlit as st
import time

def format_price_sidebar(price):
    """Format price for sidebar display"""
    if price >= 1000:
        return f"${price:,.2f}"
    elif price >= 100:
        return f"${price:,.2f}"
    elif price >= 1:
        return f"${price:,.4f}"
    elif price >= 0.01:
        return f"${price:,.6f}"
    else:
        return f"${price:.8f}".rstrip('0').rstrip('.')

def render_sidebar(symbols):
    """
    Render sidebar with real-time prices from WebSocket
    NO auto-refresh, prices update from background WebSocket
    """
    
    # Custom CSS for sidebar styling ONLY
    st.markdown("""
    <style>
    /* Sidebar background */
    [data-testid="stSidebar"] {
        background-color: #1e1e1e;
    }
    
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
        color: #ffffff;
    }
    
    /* Header styling */
    .sidebar-header {
        padding: 10px 0;
        margin-bottom: 15px;
        border-bottom: 1px solid #3d3d3d;
    }
    
    .sidebar-title {
        color: #ffffff;
        font-weight: bold;
        font-size: 20px;
        margin-bottom: 5px;
    }
    
    .sidebar-subtitle {
        color: #7f8c8d;
        font-size: 14px;
    }
    
    /* Symbol item styling */
    .sidebar-item {
        padding: 12px;
        margin-bottom: 8px;
        background-color: #2d2d2d;
        border-radius: 8px;
        border: 1px solid #3d3d3d;
        transition: all 0.2s;
    }
    
    .sidebar-item:hover {
        background-color: #353535;
        border-color: #4d4d4d;
    }
    
    .sidebar-symbol {
        color: #ffffff;
        font-weight: 600;
        font-size: 16px;
        margin-bottom: 4px;
    }
    
    .sidebar-price {
        font-size: 15px;
        font-weight: 500;
        margin-bottom: 2px;
    }
    
    .sidebar-change {
        font-size: 13px;
    }
    
    .price-up {
        color: #27ae60;
    }
    
    .price-down {
        color: #e74c3c;
    }
    
    .sidebar-loading {
        color: #7f8c8d;
        font-size: 13px;
        font-style: italic;
    }
    
    /* Refresh button styling */
    .stButton > button {
        width: 100%;
        background-color: #2d2d2d;
        color: #ffffff;
        border: 1px solid #3d3d3d;
        transition: all 0.2s;
    }
    
    .stButton > button:hover {
        background-color: #353535;
        border-color: #4d4d4d;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Use Streamlit's NATIVE sidebar (collapsible by default)
    with st.sidebar:
        # Header
        st.markdown("""
        <div class="sidebar-header">
            <div class="sidebar-title">📊 Symbols</div>
            <div class="sidebar-subtitle">Live via WebSocket</div>
        </div>
        """, unsafe_allow_html=True)
        
        if not symbols:
            st.info("No symbols added yet")
        else:
            # Get WebSocket handler
            ws_handler = st.session_state.get('ws_handler')
            
            if not ws_handler:
                st.warning("⚠️ WebSocket not connected")
            else:
                # Render each symbol with WebSocket data
                for symbol in symbols:
                    # Get price from WebSocket (already real-time in background)
                    ticker_data = ws_handler.get_price(symbol)
                    
                    if ticker_data and (time.time() - ticker_data['timestamp']) < 5:
                        price = ticker_data['price']
                        change_pct = ticker_data['change_percent']
                        is_up = change_pct >= 0
                        
                        formatted_price = format_price_sidebar(price)
                        
                        # Display symbol card
                        st.markdown(f"""
                        <div class="sidebar-item">
                            <div class="sidebar-symbol">{symbol}</div>
                            <div class="sidebar-price {'price-up' if is_up else 'price-down'}">
                                {formatted_price}
                            </div>
                            <div class="sidebar-change {'price-up' if is_up else 'price-down'}">
                                {'▲' if is_up else '▼'} {abs(change_pct):.2f}%
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Optional: Chart button for each symbol
                        if st.button("📊", key=f"sidebar_chart_{symbol}", help=f"View {symbol} chart"):
                            st.session_state.show_chart = True
                            st.session_state.chart_symbol = symbol
                            st.rerun()
                    else:
                        st.markdown(f"""
                        <div class="sidebar-item">
                            <div class="sidebar-symbol">{symbol}</div>
                            <div class="sidebar-loading">Connecting...</div>
                        </div>
                        """, unsafe_allow_html=True)
        
        # Footer with manual refresh button
        st.markdown("---")
        
        # Manual refresh button (optional)
        if st.button("🔄 Refresh Prices", use_container_width=True):
            st.rerun()
        
        st.markdown(f"""
        <div style="text-align: center; color: #7f8c8d; font-size: 12px; margin-top: 10px;">
            Total: {len(symbols)} symbols<br>
            Live via WebSocket
        </div>
        """, unsafe_allow_html=True)
