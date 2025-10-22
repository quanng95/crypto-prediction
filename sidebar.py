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
    Shows real-time prices without flickering
    """
    
    # Custom CSS for sidebar - ẩn button và giảm flickering
    st.markdown("""
    <style>
    /* Ẩn button toggle của Streamlit */
    button[kind="secondary"] {
        display: none !important;
    }
    
    /* Ẩn toàn bộ header controls nếu cần */
    .stSidebar button {
        display: none !important;
    }
    
    /* Smooth transitions để giảm flickering */
    .stSidebar {
        transition: none !important;
    }
    
    .sidebar-item {
        padding: 12px 15px;
        border-bottom: 1px solid #2d2d2d;
        transition: background-color 0.2s;
        /* Giảm flickering bằng cách giữ layout ổn định */
        min-height: 80px;
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
        /* Giữ chiều cao cố định */
        line-height: 1.4;
    }
    
    .sidebar-change {
        font-size: 13px;
        margin-top: 2px;
        line-height: 1.4;
    }
    
    .price-up {
        color: #27ae60;
    }
    
    .price-down {
        color: #e74c3c;
    }
    
    /* Giảm animation của Streamlit */
    .stMarkdown {
        animation: none !important;
    }
    
    /* Fix flickering cho fragment */
    [data-testid="stVerticalBlock"] {
        transition: none !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    with st.sidebar:
        st.markdown('<div style="padding: 15px; background-color: #2d2d2d; border-bottom: 1px solid #3d3d3d;">'
                   '<span style="color: #ffffff; font-weight: bold; font-size: 18px;">📊 Symbols</span>'
                   '</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        if not symbols:
            st.info("No symbols added yet")
        else:
            # Sử dụng fragment với interval dài hơn để giảm flickering
            @st.fragment(run_every="2s")  # Tăng từ 1s lên 2s
            def render_symbol_list():
                # Cache data để giảm flickering
                symbol_data = {}
                for symbol in symbols:
                    ticker_data = get_ticker_realtime_sidebar(symbol)
                    if ticker_data:
                        symbol_data[symbol] = ticker_data
                
                # Render tất cả cùng lúc
                for symbol in symbols:
                    if symbol in symbol_data:
                        ticker_data = symbol_data[symbol]
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
