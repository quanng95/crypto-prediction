# sidebar.py
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
    Render Streamlit native sidebar with symbol list
    Shows real-time prices with clean styling
    """
    
    # Custom CSS for sidebar styling (works with native sidebar)
    st.markdown("""
    <style>
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #1e1e1e;
    }
    
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
        color: #ffffff;
    }
    
    /* Symbol card styling */
    .sidebar-symbol-card {
        background-color: #2d2d2d;
        padding: 12px;
        border-radius: 8px;
        margin-bottom: 10px;
        border: 1px solid #3d3d3d;
        transition: all 0.2s ease;
    }
    
    .sidebar-symbol-card:hover {
        background-color: #353535;
        border-color: #4d4d4d;
    }
    
    .sidebar-symbol-name {
        color: #ffffff;
        font-weight: 600;
        font-size: 16px;
        margin-bottom: 6px;
    }
    
    .sidebar-price {
        font-size: 15px;
        font-weight: 500;
        margin-bottom: 4px;
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
    
    /* Divider styling */
    [data-testid="stSidebar"] hr {
        margin: 15px 0;
        border-color: #3d3d3d;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Use native Streamlit sidebar
    with st.sidebar:
        # Header
        st.markdown("## 📊 Symbols")
        
        # Info about collapse
        st.caption("💡 Use the arrow (◀) at top-left to collapse sidebar")
        
        st.markdown("---")
        
        # Check if symbols exist
        if not symbols:
            st.info("No symbols added yet")
            return
        
        # Symbol count
        st.markdown(f"**Tracking {len(symbols)} symbols**")
        
        st.markdown("---")
        
        # Auto-refresh price display using fragment
        @st.fragment(run_every="1s")
        def render_symbol_list():
            for idx, symbol in enumerate(symbols):
                ticker_data = get_ticker_realtime_sidebar(symbol)
                
                if ticker_data:
                    price = ticker_data['price']
                    change_pct = ticker_data['change_percent']
                    is_up = change_pct >= 0
                    
                    formatted_price = format_price_sidebar(price)
                    
                    # Render symbol card
                    st.markdown(f"""
                    <div class="sidebar-symbol-card">
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
                    <div class="sidebar-symbol-card">
                        <div class="sidebar-symbol-name">{symbol}</div>
                        <div style="color: #7f8c8d; font-size: 13px;">Loading...</div>
                    </div>
                    """, unsafe_allow_html=True)
        
        # Render the list
        render_symbol_list()
        
        # Footer
        st.markdown("---")
        st.caption("🔄 Auto-refresh: 1 second")
