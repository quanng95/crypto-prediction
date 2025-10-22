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
    Render sidebar with symbol list
    Shows real-time prices without box styling
    """
    
    # Custom CSS for sidebar
    st.markdown("""
    <style>
    .sidebar-container {
        position: fixed;
        left: 0;
        top: 80px;
        height: calc(100vh - 80px);
        background-color: #1e1e1e;
        border-right: 1px solid #3d3d3d;
        transition: all 0.3s ease;
        z-index: 999;
        overflow-y: auto;
    }
    
    .sidebar-header {
        padding: 15px;
        background-color: #2d2d2d;
        border-bottom: 1px solid #3d3d3d;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    
    .sidebar-title {
        color: #ffffff;
        font-weight: bold;
        font-size: 18px;
    }
    
    .sidebar-item {
        padding: 12px 15px;
        border-bottom: 1px solid #2d2d2d;
        cursor: pointer;
        transition: background-color 0.2s;
    }
    
    .sidebar-item:hover {
        background-color: #2d2d2d;
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
    }
    
    .sidebar-change {
        font-size: 13px;
        margin-top: 2px;
    }
    
    .price-up {
        color: #27ae60;
    }
    
    .price-down {
        color: #e74c3c;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Sidebar container
    with st.sidebar:
        # Header - chỉ hiển thị title, không có nút toggle
        st.markdown('<div class="sidebar-title">📊 Symbols</div>', unsafe_allow_html=True)
        st.markdown("---")
        
        # Show symbols
        if not symbols:
            st.info("No symbols added yet")
        else:
            # Auto-refresh price display
            @st.fragment(run_every="1s")
            def render_symbol_list():
                for symbol in symbols:
                    ticker_data = get_ticker_realtime_sidebar(symbol)
                    
                    if ticker_data:
                        price = ticker_data['price']
                        change_pct = ticker_data['change_percent']
                        is_up = change_pct >= 0
                        
                        formatted_price = format_price_sidebar(price)
                        
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
                    else:
                        st.markdown(f"""
                        <div class="sidebar-item">
                            <div class="sidebar-symbol">{symbol}</div>
                            <div style="color: #7f8c8d; font-size: 13px;">Loading...</div>
                        </div>
                        """, unsafe_allow_html=True)
            
            render_symbol_list()
