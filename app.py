import streamlit as st
import pandas as pd
import time
import requests
from datetime import datetime
import atexit

# Import các module
from websocket_handler import BinanceWebSocket
from eth import AdvancedETHPredictor
from chart_component import render_tradingview_chart
from custom_css import get_custom_css
from tabs_content import render_all_tabs
from symbol_manager import render_simple_add_symbol
from auth_pages import render_login_page, render_signup_page, render_user_menu
from database_postgres import Database
from admin_panel import render_admin_login, render_admin_panel

# Page config
st.set_page_config(
    page_title="🔮 Crypto Prediction",
    page_icon="🔮",
    layout="wide"
)

# Apply custom CSS
st.markdown(get_custom_css(), unsafe_allow_html=True)

# Initialize page state
if 'page' not in st.session_state:
    st.session_state.page = "home"

if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False

if 'user' not in st.session_state:
    st.session_state.user = None

if 'admin_authenticated' not in st.session_state:
    st.session_state.admin_authenticated = False

# Admin panel routing
if st.session_state.page == "admin":
    if not st.session_state.admin_authenticated:
        render_admin_login()
    else:
        render_admin_panel()
    st.stop()

# Handle page routing
if st.session_state.page == "login":
    render_login_page()
    st.stop()

if st.session_state.page == "signup":
    render_signup_page()
    st.stop()

# HOME PAGE - Continue with normal app
# Initialize default symbols in session state
if 'SYMBOLS' not in st.session_state:
    if st.session_state.authenticated:
        # Load user's symbols
        db = Database()
        symbols = db.get_user_symbols(st.session_state.user['id'])
        st.session_state.SYMBOLS = symbols if symbols else [
            "ETHUSDT", "BTCUSDT", "PAXGUSDT", "BNBUSDT", "SOLUSDT",
            "LINKUSDT", "PEPEUSDT", "XRPUSDT", "DOGEUSDT", "KAITOUSDT", "ADAUSDT"
        ]
    else:
        st.session_state.SYMBOLS = [
            "ETHUSDT", "BTCUSDT", "PAXGUSDT", "BNBUSDT", "SOLUSDT",
            "LINKUSDT", "PEPEUSDT", "XRPUSDT", "DOGEUSDT", "KAITOUSDT", "ADAUSDT"
        ]

SYMBOLS = st.session_state.SYMBOLS

# Initialize session state
if 'show_chart' not in st.session_state:
    st.session_state.show_chart = False
if 'chart_symbol' not in st.session_state:
    st.session_state.chart_symbol = "ETHUSDT"
if 'chart_interval' not in st.session_state:
    st.session_state.chart_interval = "1h"
if 'predictor' not in st.session_state:
    st.session_state.predictor = None
if 'predictions' not in st.session_state:
    st.session_state.predictions = None
if 'ticker_start_index' not in st.session_state:
    st.session_state.ticker_start_index = 0
if 'selected_symbol' not in st.session_state:
    st.session_state.selected_symbol = "ETHUSDT"
if 'selected_timezone' not in st.session_state:
    st.session_state.selected_timezone = "Asia/Ho_Chi_Minh"
if 'trigger_analysis' not in st.session_state:
    st.session_state.trigger_analysis = False
if 'last_refresh' not in st.session_state:
    st.session_state.last_refresh = time.time()
if 'chart_data_cache' not in st.session_state:
    st.session_state.chart_data_cache = {}
if 'hover_symbol' not in st.session_state:
    st.session_state.hover_symbol = None

# Initialize WebSocket
if 'ws_handler' not in st.session_state:
    st.session_state.ws_handler = BinanceWebSocket()
    st.session_state.ws_handler.start(SYMBOLS)
    st.session_state.ws_symbols = SYMBOLS.copy()
    print("🚀 WebSocket initialized")
elif st.session_state.ws_symbols != SYMBOLS:
    st.session_state.ws_handler.stop()
    st.session_state.ws_handler = BinanceWebSocket()
    st.session_state.ws_handler.start(SYMBOLS)
    st.session_state.ws_symbols = SYMBOLS.copy()
    print("🔄 WebSocket restarted")

@st.cache_data(ttl=5)
def get_ticker(symbol):
    """Get ticker from Binance REST API (fallback)"""
    try:
        url = f"https://api.binance.com/api/v3/ticker/24hr"
        response = requests.get(url, params={'symbol': symbol}, timeout=5)
        data = response.json()
        return {
            'price': float(data['lastPrice']),
            'change_percent': float(data['priceChangePercent']),
            'high': float(data['highPrice']),
            'low': float(data['lowPrice']),
            'volume': float(data['volume'])
        }
    except:
        return None

def get_ticker_realtime(symbol):
    """Get ticker from WebSocket (real-time) with REST API fallback"""
    data = st.session_state.ws_handler.get_price(symbol)
    
    if data and (time.time() - data['timestamp']) < 5:
        return data
    
    return get_ticker(symbol)

@st.cache_data(ttl=60)
def get_klines(symbol, interval='1h', limit=200):
    """Get candlestick data"""
    try:
        url = f"https://api.binance.com/api/v3/klines"
        response = requests.get(url, params={
            'symbol': symbol,
            'interval': interval,
            'limit': limit
        }, timeout=10)
        data = response.json()
        
        df = pd.DataFrame(data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])
        
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)
        
        return df
    except Exception as e:
        st.error(f"Error: {e}")
        return None

def format_price(price):
    """Format price based on its value"""
    if price >= 1000:
        return f"${price:,.2f}"
    elif price >= 100:
        return f"${price:,.2f}"
    elif price >= 1:
        return f"${price:,.4f}"
    elif price >= 0.01:
        return f"${price:,.6f}"
    elif price >= 0.0001:
        return f"${price:,.8f}"
    else:
        return f"${price:.11f}".rstrip('0').rstrip('.')

# ============================================
# USER MENU (TOP RIGHT)
# ============================================
render_user_menu()

# ============================================
# BEAUTIFUL HEADER
# ============================================
st.markdown("""
<div class="header-container">
    <h1 class="header-title">🔮 Crypto Prediction</h1>
</div>
""", unsafe_allow_html=True)

# ============================================
# SIMPLE ADD SYMBOL
# ============================================
st.markdown("---")

updated_symbols = render_simple_add_symbol(st.session_state.SYMBOLS)

if updated_symbols != st.session_state.SYMBOLS:
    st.session_state.SYMBOLS = updated_symbols
    SYMBOLS = updated_symbols
    
    # Save to database if authenticated
    if st.session_state.authenticated:
        db = Database()
        db.save_user_symbols(st.session_state.user['id'], SYMBOLS)
    
    # Restart WebSocket
    st.session_state.ws_handler.stop()
    st.session_state.ws_handler = BinanceWebSocket()
    st.session_state.ws_handler.start(SYMBOLS)
    st.session_state.ws_symbols = SYMBOLS.copy()
    
    st.rerun()

# ============================================
# TICKER CAROUSEL WITH REMOVE BUTTON (8-2 ratio)
# ============================================
@st.fragment(run_every="0.5s")
def ticker_carousel():
    """Ticker carousel with auto-refresh and remove functionality"""
    st.markdown("---")
    
    visible_count = 4
    total_symbols = len(SYMBOLS)
    start_idx = st.session_state.ticker_start_index
    
    nav_col1, *ticker_cols, nav_col2 = st.columns([1, 2, 2, 2, 2, 1])
    
    with nav_col1:
        if st.button("◀", key="prev_btn", help="Previous symbols"):
            if st.session_state.ticker_start_index == 0:
                st.session_state.ticker_start_index = total_symbols - visible_count
            else:
                st.session_state.ticker_start_index -= visible_count
            st.rerun()
    
    for idx, col in enumerate(ticker_cols):
        symbol_idx = start_idx + idx
        if symbol_idx < len(SYMBOLS):
            symbol = SYMBOLS[symbol_idx]
            
            with col:
                ticker_data = get_ticker_realtime(symbol)
                
                if ticker_data:
                    change_pct = ticker_data['change_percent']
                    is_up = change_pct >= 0
                    price = ticker_data['price']
                    formatted_price = format_price(price)
                    
                    # Ticker card
                    st.markdown(f"""
                    <div style="background-color: #2d2d2d; padding: 15px; border-radius: 8px; border: 1px solid #3d3d3d; text-align: center;">
                        <div style="color: #ffffff; font-weight: bold; font-size: 16px; margin-bottom: 8px;">{symbol}</div>
                        <div style="color: {'#27ae60' if is_up else '#e74c3c'}; font-weight: bold; font-size: 24px; margin-bottom: 5px;">{formatted_price}</div>
                        <div style="color: {'#27ae60' if is_up else '#e74c3c'}; font-size: 14px;">{'▲' if is_up else '▼'} {abs(change_pct):.2f}%</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Buttons with 8-2 ratio (Chart button wider, Remove button smaller)
                    col_chart, col_remove = st.columns([8, 2])
                    
                    with col_chart:
                        if st.button("📊 Chart", key=f"chart_{symbol}", use_container_width=True, type="secondary"):
                            st.session_state.show_chart = True
                            st.session_state.chart_symbol = symbol
                            st.rerun()
                    
                    with col_remove:
                        if st.button("❌", key=f"remove_{symbol}", use_container_width=True, type="secondary", help=f"Remove {symbol}"):
                            if len(SYMBOLS) > 1:
                                st.session_state.SYMBOLS.remove(symbol)
                                
                                # Auto-save to database if authenticated
                                if st.session_state.get('authenticated', False):
                                    db = Database()
                                    db.save_user_symbols(
                                        st.session_state.user['id'], 
                                        st.session_state.SYMBOLS
                                    )
                                    print(f"✅ Symbols auto-saved after removing {symbol}")
                                
                                st.success(f"✅ Removed {symbol}!")
                                st.rerun()
                            else:
                                st.warning("⚠️ Cannot remove last symbol!")
    
    with nav_col2:
        if st.button("▶", key="next_btn", help="Next symbols"):
            if st.session_state.ticker_start_index + visible_count >= total_symbols:
                st.session_state.ticker_start_index = 0
            else:
                st.session_state.ticker_start_index += visible_count
            st.rerun()

ticker_carousel()

# ============================================
# CHART CONTAINER
# ============================================
if st.session_state.show_chart:
    st.markdown("---")
    st.markdown("## 📈 Candlestick Chart")
    
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        st.markdown(f"### {st.session_state.chart_symbol}")
    
    with col2:
        interval = st.selectbox(
            "Timeframe",
            ['15m', '1h', '4h', '1d'],
            index=['15m', '1h', '4h', '1d'].index(st.session_state.chart_interval),
            key="chart_interval_select"
        )
        if interval != st.session_state.chart_interval:
            st.session_state.chart_interval = interval
            st.rerun()
    
    with col3:
        if st.button("❌ Close", type="primary", key="close_chart_btn"):
            st.session_state.show_chart = False
            st.rerun()
    
    cache_key = f"{st.session_state.chart_symbol}_{st.session_state.chart_interval}"
    
    if cache_key not in st.session_state.chart_data_cache:
        df = get_klines(st.session_state.chart_symbol, st.session_state.chart_interval, 200)
        st.session_state.chart_data_cache[cache_key] = df
    else:
        df = st.session_state.chart_data_cache[cache_key]
    
    render_tradingview_chart(df, st.session_state.chart_symbol, st.session_state.chart_interval)
    
    st.markdown("---")
    st.stop()

# ============================================
# CONTROL PANEL
# ============================================
st.markdown("---")
st.markdown("### 🎛️ Control Panel")

col1, col2, col3, col4 = st.columns([2, 2, 2, 2])

with col1:
    selected_symbol = st.selectbox(
        "📊 Symbol", 
        SYMBOLS, 
        index=SYMBOLS.index(st.session_state.selected_symbol) if st.session_state.selected_symbol in SYMBOLS else 0,
        key="symbol_select_main"
    )
    if selected_symbol != st.session_state.selected_symbol:
        st.session_state.selected_symbol = selected_symbol

with col2:
    timezone_options = ["Asia/Ho_Chi_Minh", "America/New_York", "Europe/London", "Asia/Tokyo"]
    timezone = st.selectbox(
        "🌍 Timezone",
        timezone_options,
        index=timezone_options.index(st.session_state.selected_timezone),
        key="timezone_select_main"
    )
    if timezone != st.session_state.selected_timezone:
        st.session_state.selected_timezone = timezone

with col3:
    st.write("")
    st.write("")
    if st.button(
        "🚀 Run Analysis", 
        type="primary", 
        use_container_width=True,
        key="run_analysis_main"
    ):
        st.session_state.trigger_analysis = True
        st.rerun()

with col4:
    st.write("")
    
    @st.fragment(run_every="0.5s")
    def price_display():
        current_ticker = get_ticker_realtime(st.session_state.selected_symbol)
        if current_ticker:
            change_pct = current_ticker['change_percent']
            is_up = change_pct >= 0
            price = current_ticker['price']
            formatted_price = format_price(price)
            price_class = "control-price-up" if is_up else "control-price-down"
            
            st.markdown(f"""
            <div class="control-price-box">
                <div class="control-symbol">{st.session_state.selected_symbol}</div>
                <div class="{price_class}">{formatted_price}</div>
                <div style="color: {'#27ae60' if is_up else '#e74c3c'}; font-size: 14px;">
                    {'▲' if is_up else '▼'} {abs(change_pct):.2f}%
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    price_display()

# ============================================
# RUN ANALYSIS
# ============================================
if st.session_state.trigger_analysis:
    st.session_state.trigger_analysis = False
    
    with st.spinner(f"🔄 Analyzing {st.session_state.selected_symbol}..."):
        try:
            predictor = AdvancedETHPredictor(timezone=st.session_state.selected_timezone)
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text("Fetching data...")
            progress_bar.progress(20)
            
            all_data, all_predictions = predictor.run_analysis(st.session_state.selected_symbol)
            
            progress_bar.progress(100)
            status_text.text("✅ Analysis complete!")
            
            predictor.all_predictions = all_predictions
            
            st.session_state.predictor = predictor
            st.session_state.predictions = all_predictions
            
            time.sleep(1)
            progress_bar.empty()
            status_text.empty()
            
            st.success(f"✅ Analysis completed for {st.session_state.selected_symbol}!")
            st.rerun()
            
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

# ============================================
# RESULTS DISPLAY
# ============================================
if st.session_state.predictor is not None and st.session_state.predictions is not None:
    predictor = st.session_state.predictor
    all_predictions = st.session_state.predictions
    
    if not hasattr(predictor, 'all_predictions'):
        predictor.all_predictions = all_predictions
    
    st.markdown("---")
    st.markdown("## 📊 Analysis Results")
    
    render_all_tabs(predictor, all_predictions)

# Footer with Admin button
st.markdown("---")

col1, col2, col3 = st.columns([1, 6, 1])

with col1:
    if st.button("👨‍💼 Admin", key="admin_btn"):
        st.session_state.page = "admin"
        st.rerun()

with col2:
    st.markdown(
        f"<div style='text-align: center; color: #7f8c8d; font-size: 14px;'>"
        f"🔮 Crypto Prediction | Last update: {datetime.now().strftime('%H:%M:%S')} | "
        f"Powered by Streamlit & Binance WebSocket"
        f"</div>",
        unsafe_allow_html=True
    )

# Cleanup WebSocket on app close
def cleanup():
    if 'ws_handler' in st.session_state:
        st.session_state.ws_handler.stop()
        print("🛑 WebSocket stopped")

atexit.register(cleanup)
