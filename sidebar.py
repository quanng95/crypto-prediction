# sidebar.py
import streamlit as st
import time
import requests

def get_ticker_realtime_sidebar(symbol):
    """Get real-time ticker from WebSocket handler"""
    # Try to get from WebSocket first (same as ticker carousel)
    if 'ws_handler' in st.session_state:
        data = st.session_state.ws_handler.get_price(symbol)
        if data and (time.time() - data['timestamp']) < 5:
            return data
    
    # Fallback to REST API
    try:
        url = "https://api.binance.com/api/v3/ticker/24hr"
        response = requests.get(url, params={'symbol': symbol}, timeout=5)
        data = response.json()
        
        return {
            'price': float(data['lastPrice']),
            'change_percent': float(data['priceChangePercent']),
            'timestamp': time.time()
        }
    except:
        return None

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
    Render Streamlit native sidebar with symbol list
    Real-time prices without boxes, clean text format
    """
    
    # Custom CSS for sidebar styling
    st.markdown("""
    <style>
    /* Sidebar background */
    [data-testid="stSidebar"] {
        background-color: #1e1e1e;
    }
    
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
        color: #ffffff;
    }
    
    /* Header styling - GIẢM SIZE */
    [data-testid="stSidebar"] h2 {
        font-size: 18px !important;  /* Giảm từ 24px xuống 18px */
        margin-bottom: 10px !important;
    }
                
    /* Symbol item - no box, just text */
    .sidebar-symbol-item {
        padding: 8px 0;
        border-bottom: 1px solid #2d2d2d;
    }
    
    .sidebar-symbol-name {
        color: #ffffff;
        font-weight: 600;
        font-size: 15px;
        margin-bottom: 4px;
    }
    
    .sidebar-price {
        font-size: 14px;
        font-weight: 500;
        margin-bottom: 2px;
    }
    
    .sidebar-change {
        font-size: 12px;
    }
    
    .price-up {
        color: #27ae60;
    }
    
    .price-down {
        color: #e74c3c;
    }
    
    /* Divider styling */
    [data-testid="stSidebar"] hr {
        margin: 15px 0;
        border-color: #3d3d3d;
    }
    
    /* Remove extra spacing */
    [data-testid="stSidebar"] .element-container {
        margin-bottom: 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Use native Streamlit sidebar
    with st.sidebar:
        # Header
        st.markdown("### 📊 Symbols")
        
        st.markdown("---")
        
        # Check if symbols exist
        if not symbols:
            st.info("No symbols added yet")
            return
        
        # # Symbol count
        # st.markdown(f"**Tracking {len(symbols)} symbols**")
        
        # st.markdown("---")
        
        # Real-time price display with fragment (auto-refresh like ticker carousel)
        @st.fragment(run_every="0.5s")
        def render_symbol_list():
            for symbol in symbols:
                ticker_data = get_ticker_realtime_sidebar(symbol)
                
                if ticker_data:
                    price = ticker_data['price']
                    change_pct = ticker_data['change_percent']
                    is_up = change_pct >= 0
                    
                    formatted_price = format_price_sidebar(price)
                    
                    # Render symbol item - simple text format, no box
                    st.markdown(f"""
                    <div class="sidebar-symbol-item">
                        <div class="sidebar-symbol-name">{symbol}</div>
                        <div class="sidebar-price {'price-up' if is_up else 'price-down'}">
                            {formatted_price}
                        </div>
                        <div class="sidebar-change {'price-up' if is_up else 'price-down'}">
                            {'▲' if is_up else '▼'} {abs(change_pct):.2f}%
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    # Loading state
                    st.markdown(f"""
                    <div class="sidebar-symbol-item">
                        <div class="sidebar-symbol-name">{symbol}</div>
                        <div style="color: #7f8c8d; font-size: 12px;">Loading...</div>
                    </div>
                    """, unsafe_allow_html=True)
        
        # Render the list
        render_symbol_list()
        

