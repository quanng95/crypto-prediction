import streamlit as st
import time
import requests

def get_ticker_realtime_sidebar(symbol):
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
    Render sidebar with symbol list - Real-time updates without refresh
    Uses Streamlit's native sidebar
    """
    
    # Custom CSS for sidebar styling
    st.markdown("""
    <style>
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #1e1e1e;
    }
    
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
        color: #ffffff;
    }
    
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
    </style>
    """, unsafe_allow_html=True)
    
    # Use Streamlit's native sidebar
    with st.sidebar:
        # Header
        st.markdown("""
        <div class="sidebar-header">
            <div class="sidebar-title">📊 Symbols</div>
            <div class="sidebar-subtitle">Real-time prices</div>
        </div>
        """, unsafe_allow_html=True)
        
        if not symbols:
            st.info("No symbols added yet")
        else:
            # Create placeholder for each symbol
            symbol_placeholders = {}
            for symbol in symbols:
                symbol_placeholders[symbol] = st.empty()
            
            # Real-time update function (runs in fragment)
            @st.fragment(run_every="1s")
            def update_prices():
                for symbol in symbols:
                    ticker_data = get_ticker_realtime_sidebar(symbol)
                    
                    if ticker_data:
                        price = ticker_data['price']
                        change_pct = ticker_data['change_percent']
                        is_up = change_pct >= 0
                        
                        formatted_price = format_price_sidebar(price)
                        
                        # Update placeholder with new data
                        symbol_placeholders[symbol].markdown(f"""
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
                    else:
                        symbol_placeholders[symbol].markdown(f"""
                        <div class="sidebar-item">
                            <div class="sidebar-symbol">{symbol}</div>
                            <div class="sidebar-loading">Loading...</div>
                        </div>
                        """, unsafe_allow_html=True)
            
            # Run real-time updates
            update_prices()
        
        # Footer info
        st.markdown("---")
        st.markdown(f"""
        <div style="text-align: center; color: #7f8c8d; font-size: 12px;">
            Total: {len(symbols)} symbols<br>
            Updates every 1s
        </div>
        """, unsafe_allow_html=True)
