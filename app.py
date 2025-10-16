import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import requests
from datetime import datetime
import numpy as np
from streamlit_autorefresh import st_autorefresh

# Import WebSocket handler
from websocket_handler import BinanceWebSocket

# Import predictor
from eth import AdvancedETHPredictor

# Page config
st.set_page_config(
    page_title="🔮 Crypto Prediction",
    page_icon="🔮",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main { 
        background-color: #1e1e1e;
        font-size: 16px !important;
    }
    
    /* Beautiful Header */
    .header-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 30px;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 30px;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
    }
    
    .header-title {
        color: #ffffff;
        font-size: 48px;
        font-weight: bold;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .header-subtitle {
        color: #e0e0ff;
        font-size: 18px;
        margin-top: 10px;
        opacity: 0.9;
    }
    
    /* Control price box */
    .control-price-box {
        background-color: #2d2d2d;
        padding: 10px;
        border-radius: 5px;
        margin-top: 25px;
        border: 1px solid #3d3d3d;
    }
    
    .control-symbol {
        color: #ffffff !important;
        font-weight: bold !important;
        font-size: 14px !important;
    }
    
    .control-price-up {
        color: #27ae60 !important;
        font-weight: bold;
        font-size: 28px;
    }
    
    .control-price-down {
        color: #e74c3c !important;
        font-weight: bold;
        font-size: 28px;
    }
    
    /* Trading signal boxes */
    .signal-box {
        background-color: #2d2d2d;
        padding: 20px;
        border-radius: 8px;
        margin: 10px 0;
        border: 2px solid #3d3d3d;
    }
    
    .signal-long {
        border-left: 4px solid #27ae60;
    }
    
    .signal-short {
        border-left: 4px solid #e74c3c;
    }
    
    .signal-neutral {
        border-left: 4px solid #95a5a6;
    }
    
    p, span, div, label {
        color: #e0e0e0 !important;
        font-size: 16px;
    }
    
    h1, h2, h3 {
        color: #ffffff !important;
    }
    
    h1 {
        font-size: 42px !important;
    }
    
    h2 {
        font-size: 36px !important;
    }
    
    h3 {
        font-size: 28px !important;
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    [data-testid="stMetricValue"] {
        font-size: 22px;
        color: #e0e0e0 !important;
    }
    
    [data-testid="stMetricLabel"] {
        color: #ffffff !important;
    }
    
    .stButton button {
        font-size: 15px;
    }
    
    .stSelectbox > div > div {
        background-color: #2d2d2d !important;
        color: #e0e0e0 !important;
    }
    
    [data-testid="stDataFrame"] {
        background-color: #2d2d2d;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        background-color: #2d2d2d;
    }
    
    .stTabs [data-baseweb="tab"] {
        color: #e0e0e0 !important;
    }
    
    .stTabs [aria-selected="true"] {
        color: #ffffff !important;
    }
</style>
""", unsafe_allow_html=True)

# Symbols
SYMBOLS = ["ETHUSDT", "BTCUSDT", "BNBUSDT", "SOLUSDT", "ADAUSDT", "DOGEUSDT", "ARBUSDT", "PAXGUSDT"]

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

# Initialize WebSocket
if 'ws_handler' not in st.session_state:
    st.session_state.ws_handler = BinanceWebSocket()
    st.session_state.ws_handler.start(SYMBOLS)
    print("🚀 WebSocket initialized")

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
    
    # Check if data is fresh (< 5 seconds old)
    if data and (time.time() - data['timestamp']) < 5:
        return data
    
    # Fallback to REST API
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

def calculate_trading_signal(predictor, timeframe):
    """Calculate trading signal based on predictions"""
    if timeframe not in predictor.all_model_results:
        return None
    
    best_model = predictor.best_models.get(timeframe)
    if not best_model:
        return None
    
    result = predictor.all_model_results[timeframe][best_model]
    predictions = predictor.all_predictions.get(timeframe, {}).get(best_model, [])
    
    if not predictions or len(predictions) < 3:
        return None
    
    current_price = predictor.reference_price
    
    short_term = predictions[:3]
    mid_term = predictions[3:5] if len(predictions) > 3 else predictions[:3]
    
    avg_short = np.mean(short_term)
    avg_mid = np.mean(mid_term)
    
    short_change = ((avg_short / current_price) - 1) * 100
    mid_change = ((avg_mid / current_price) - 1) * 100
    
    if short_change > 2 and mid_change > 3:
        signal = "LONG"
        confidence = min(95, 60 + abs(short_change) * 5)
        bull_prob = min(95, 55 + abs(short_change) * 3)
        bear_prob = 100 - bull_prob
    elif short_change < -2 and mid_change < -3:
        signal = "SHORT"
        confidence = min(95, 60 + abs(short_change) * 5)
        bear_prob = min(95, 55 + abs(short_change) * 3)
        bull_prob = 100 - bear_prob
    else:
        signal = "NEUTRAL"
        confidence = 50
        bull_prob = 50
        bear_prob = 50
    
    if signal == "LONG":
        entry = current_price * 0.995
        stop_loss = entry * 0.97
        tp1 = entry * 1.02
        tp2 = entry * 1.05
        tp3 = entry * 1.10
    elif signal == "SHORT":
        entry = current_price * 1.005
        stop_loss = entry * 1.03
        tp1 = entry * 0.98
        tp2 = entry * 0.95
        tp3 = entry * 0.90
    else:
        entry = current_price
        stop_loss = current_price * 0.97
        tp1 = current_price * 1.02
        tp2 = current_price * 1.05
        tp3 = current_price * 1.08
    
    accuracy = result['direction_accuracy'] * 100
    r2_score = result['r2']
    
    return {
        'signal': signal,
        'confidence': confidence,
        'bull_prob': bull_prob,
        'bear_prob': bear_prob,
        'entry': entry,
        'stop_loss': stop_loss,
        'tp1': tp1,
        'tp2': tp2,
        'tp3': tp3,
        'accuracy': accuracy,
        'r2_score': r2_score,
        'current_price': current_price,
        'short_term_change': short_change,
        'mid_term_change': mid_change
    }

# ============================================
# BEAUTIFUL HEADER
# ============================================
st.markdown("""
<div class="header-container">
    <h1 class="header-title">🔮 Crypto Prediction</h1>
    <p class="header-subtitle">AI-Powered Real-time Market Analysis & Trading Signals</p>
</div>
""", unsafe_allow_html=True)

# ============================================
# FRAGMENT 1: TICKER CAROUSEL (Auto-refresh)
# ============================================
@st.fragment
def ticker_carousel():
    """Ticker carousel with auto-refresh"""
    count = st_autorefresh(interval=500, limit=None, key="ticker_refresh")
    
    st.markdown("---")
    st.markdown("### 📊 Live Market Prices")
    
    visible_count = 4
    start_idx = st.session_state.ticker_start_index
    
    nav_col1, *ticker_cols, nav_col2 = st.columns([1, 2, 2, 2, 2, 1])
    
    # Previous button
    with nav_col1:
        if st.button("◀", key=f"prev_btn_{count}", help="Previous symbols"):
            if st.session_state.ticker_start_index > 0:
                st.session_state.ticker_start_index -= visible_count
                st.rerun()
    
    # Display tickers
    for idx, col in enumerate(ticker_cols):
        symbol_idx = start_idx + idx
        if symbol_idx < len(SYMBOLS):
            symbol = SYMBOLS[symbol_idx]
            
            with col:
                ticker_data = get_ticker_realtime(symbol)
                
                if ticker_data:
                    change_pct = ticker_data['change_percent']
                    is_up = change_pct >= 0
                    
                    if st.button(
                        f"**{symbol}**\n\n${ticker_data['price']:,.2f}\n\n{'▲' if is_up else '▼'} {abs(change_pct):.2f}%",
                        key=f"ticker_{symbol}_{count}",
                        use_container_width=True,
                        type="secondary"
                    ):
                        st.session_state.show_chart = True
                        st.session_state.chart_symbol = symbol
                        st.rerun()
    
    # Next button
    with nav_col2:
        if st.button("▶", key=f"next_btn_{count}", help="Next symbols"):
            if st.session_state.ticker_start_index + visible_count < len(SYMBOLS):
                st.session_state.ticker_start_index += visible_count
                st.rerun()

# ============================================
# FRAGMENT 2: REAL-TIME PRICE DISPLAY (Auto-refresh)
# ============================================
@st.fragment
def realtime_price_display():
    """Display real-time price for selected symbol"""
    count = st_autorefresh(interval=500, limit=None, key="price_refresh")
    
    current_ticker = get_ticker_realtime(st.session_state.selected_symbol)
    if current_ticker:
        change_pct = current_ticker['change_percent']
        is_up = change_pct >= 0
        price_class = "control-price-up" if is_up else "control-price-down"
        
        st.markdown(f"""
        <div class="control-price-box">
            <div class="control-symbol">{st.session_state.selected_symbol}</div>
            <div class="{price_class}">${current_ticker['price']:,.2f}</div>
            <div style="color: {'#27ae60' if is_up else '#e74c3c'}; font-size: 14px;">
                {'▲' if is_up else '▼'} {abs(change_pct):.2f}%
            </div>
        </div>
        """, unsafe_allow_html=True)

# ============================================
# MAIN CONTROL PANEL (Static - No auto-refresh)
# ============================================
st.markdown("---")
st.markdown("### 🎛️ Control Panel")

col1, col2, col3, col4 = st.columns([2, 2, 2, 2])

with col1:
    selected_symbol = st.selectbox(
        "📊 Select Symbol", 
        SYMBOLS, 
        index=SYMBOLS.index(st.session_state.selected_symbol),
        key="symbol_select_main"
    )
    if selected_symbol != st.session_state.selected_symbol:
        st.session_state.selected_symbol = selected_symbol
        st.rerun()

with col2:
    timezone_options = ["Asia/Ho_Chi_Minh", "America/New_York", "Europe/London", "Asia/Tokyo"]
    timezone = st.selectbox(
        "🌍 Select Timezone",
        timezone_options,
        index=timezone_options.index(st.session_state.selected_timezone),
        key="timezone_select_main"
    )
    if timezone != st.session_state.selected_timezone:
        st.session_state.selected_timezone = timezone
        st.rerun()

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
    # Real-time price display (auto-refreshing fragment)
    realtime_price_display()

# Call ticker carousel fragment
ticker_carousel()

# ============================================
# CANDLESTICK CHART
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
            index=['15m', '1h', '4h', '1d'].index(st.session_state.chart_interval)
        )
        if interval != st.session_state.chart_interval:
            st.session_state.chart_interval = interval
    
    with col3:
        if st.button("❌ Close", type="primary"):
            st.session_state.show_chart = False
            st.rerun()
    
    df = get_klines(st.session_state.chart_symbol, st.session_state.chart_interval, 200)
    
    if df is not None and len(df) > 0:
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.7, 0.3],
            subplot_titles=(f'{st.session_state.chart_symbol} - {st.session_state.chart_interval.upper()}', 'Volume')
        )
        
        fig.add_trace(
            go.Candlestick(
                x=df['timestamp'],
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                name='Price',
                increasing_line_color='#26a69a',
                decreasing_line_color='#ef5350'
            ),
            row=1, col=1
        )
        
        colors = ['#26a69a' if row['close'] >= row['open'] else '#ef5350' 
                 for _, row in df.iterrows()]
        
        fig.add_trace(
            go.Bar(
                x=df['timestamp'],
                y=df['volume'],
                name='Volume',
                marker_color=colors,
                opacity=0.6
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            height=700,
            template='plotly_dark',
            xaxis_rangeslider_visible=False,
            showlegend=False,
            hovermode='x unified',
            dragmode='pan',
            modebar_add=['zoom2d', 'zoomIn2d', 'zoomOut2d', 'autoScale2d', 'resetScale2d']
        )
        
        fig.update_xaxes(fixedrange=False)
        fig.update_yaxes(fixedrange=False)
        
        fig.update_xaxes(title_text="Time", row=2, col=1)
        fig.update_yaxes(title_text="Price ($)", row=1, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1)
        
        st.plotly_chart(fig, use_container_width=True, config={
            'scrollZoom': True,
            'displayModeBar': True,
            'displaylogo': False,
            'modeBarButtonsToRemove': ['select2d', 'lasso2d']
        })
        
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
    
    st.markdown("---")
    st.stop()

# ============================================
# RUN ANALYSIS (Triggered by button)
# ============================================
if st.session_state.trigger_analysis:
    st.session_state.trigger_analysis = False  # Reset flag
    
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
    
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🎯 Trading Signals",
        "📈 Summary",
        "⏰ 4H Predictions",
        "📅 1D Predictions",
        "📆 1W Predictions",
        "🔮 Final Predictions"
    ])
    
    # TAB 1: Trading Signals
    with tab1:
        st.markdown("### 🎯 Trading Signals & Recommendations")
        
        for timeframe in ['4h', '1d', '1w']:
            signal_data = calculate_trading_signal(predictor, timeframe)
            
            if signal_data:
                signal = signal_data['signal']
                
                if signal == "LONG":
                    box_class = "signal-box signal-long"
                    signal_emoji = "📈"
                    signal_color = "#27ae60"
                elif signal == "SHORT":
                    box_class = "signal-box signal-short"
                    signal_emoji = "📉"
                    signal_color = "#e74c3c"
                else:
                    box_class = "signal-box signal-neutral"
                    signal_emoji = "➡️"
                    signal_color = "#95a5a6"
                
                st.markdown(f"""
                <div class="{box_class}">
                    <h3 style="color: {signal_color};">{signal_emoji} {timeframe.upper()} - {signal} Signal</h3>
                </div>
                """, unsafe_allow_html=True)
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("#### 📊 Market Sentiment")
                    st.metric("Confidence", f"{signal_data['confidence']:.1f}%")
                    st.metric("Bull Probability", f"{signal_data['bull_prob']:.1f}%", 
                             delta=f"{signal_data['bull_prob'] - 50:+.1f}%")
                    st.metric("Bear Probability", f"{signal_data['bear_prob']:.1f}%",
                             delta=f"{signal_data['bear_prob'] - 50:+.1f}%")
                
                with col2:
                    st.markdown("#### 💰 Entry & Risk Management")
                    st.metric("Current Price", f"${signal_data['current_price']:.2f}")
                    st.metric("Entry Price", f"${signal_data['entry']:.2f}",
                             delta=f"{((signal_data['entry']/signal_data['current_price']-1)*100):+.2f}%")
                    st.metric("Stop Loss", f"${signal_data['stop_loss']:.2f}",
                             delta=f"{((signal_data['stop_loss']/signal_data['entry']-1)*100):+.2f}%",
                             delta_color="inverse")
                
                with col3:
                    st.markdown("#### 🎯 Take Profit Targets")
                    st.metric("TP1 (Conservative)", f"${signal_data['tp1']:.2f}",
                             delta=f"{((signal_data['tp1']/signal_data['entry']-1)*100):+.2f}%")
                    st.metric("TP2 (Moderate)", f"${signal_data['tp2']:.2f}",
                             delta=f"{((signal_data['tp2']/signal_data['entry']-1)*100):+.2f}%")
                    st.metric("TP3 (Aggressive)", f"${signal_data['tp3']:.2f}",
                             delta=f"{((signal_data['tp3']/signal_data['entry']-1)*100):+.2f}%")
                
                st.markdown("#### 📈 Model Performance")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Direction Accuracy", f"{signal_data['accuracy']:.1f}%")
                with col2:
                    st.metric("R² Score", f"{signal_data['r2_score']:.4f}")
                with col3:
                    st.metric("Short-term Trend", f"{signal_data['short_term_change']:+.2f}%")
                with col4:
                    st.metric("Mid-term Trend", f"{signal_data['mid_term_change']:+.2f}%")
                
                st.markdown("---")
    
    # TAB 2: Summary
    with tab2:
        st.markdown("### 🏆 Best Models Performance")
        
        perf_data = []
        for timeframe in ['4h', '1d', '1w']:
            if timeframe in predictor.all_model_results:
                best_model = predictor.best_models.get(timeframe, '')
                if best_model and best_model in predictor.all_model_results[timeframe]:
                    result = predictor.all_model_results[timeframe][best_model]
                    perf_data.append({
                        'Timeframe': timeframe.upper(),
                        'Best Model': best_model,
                        'R² Score': f"{result['r2']:.4f}",
                        'MAE ($)': f"${result['mae']:.2f}",
                        'RMSE ($)': f"${result['rmse']:.2f}",
                        'Direction Acc': f"{result['direction_accuracy']:.2%}"
                    })
        
        if perf_data:
            df_perf = pd.DataFrame(perf_data)
            st.dataframe(df_perf, use_container_width=True, hide_index=True)
        
        st.markdown("### 📊 Performance Visualization")
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('R² Score by Timeframe', 'MAE Comparison', 
                          'Direction Accuracy', 'Best Model Distribution'),
            specs=[[{'type': 'bar'}, {'type': 'bar'}],
                   [{'type': 'bar'}, {'type': 'pie'}]]
        )
        
        timeframes = []
        r2_scores = []
        mae_values = []
        dir_acc = []
        
        for tf in ['4h', '1d', '1w']:
            if tf in predictor.best_models:
                best_model = predictor.best_models[tf]
                result = predictor.all_model_results[tf][best_model]
                timeframes.append(tf.upper())
                r2_scores.append(result['r2'])
                mae_values.append(result['mae'])
                dir_acc.append(result['direction_accuracy'] * 100)
        
        fig.add_trace(
            go.Bar(x=timeframes, y=r2_scores, name='R² Score',
                   marker_color=['#2ecc71', '#3498db', '#9b59b6']),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Bar(x=timeframes, y=mae_values, name='MAE',
                   marker_color=['#e74c3c', '#f39c12', '#1abc9c']),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Bar(x=timeframes, y=dir_acc, name='Accuracy (%)',
                   marker_color=['#16a085', '#27ae60', '#2980b9']),
            row=2, col=1
        )
        
        model_names = [predictor.best_models.get(tf, 'N/A') 
                      for tf in ['4h', '1d', '1w'] 
                      if tf in predictor.best_models]
        model_counts = {}
        for model in model_names:
            model_counts[model] = model_counts.get(model, 0) + 1
        
        fig.add_trace(
            go.Pie(labels=list(model_counts.keys()), 
                   values=list(model_counts.values())),
            row=2, col=2
        )
        
        fig.update_layout(height=800, showlegend=False, template='plotly_dark')
        st.plotly_chart(fig, use_container_width=True)
    
    # TAB 3-5: Timeframe Predictions
    for tab, timeframe in zip([tab3, tab4, tab5], ['4h', '1d', '1w']):
        with tab:
            if timeframe not in all_predictions:
                st.warning(f"No predictions for {timeframe}")
                continue
            
            st.markdown(f"### 🎯 {timeframe.upper()} Predictions")
            
            pred_data = []
            best_model = predictor.best_models.get(timeframe, '')
            
            if best_model and best_model in all_predictions[timeframe]:
                predictions = all_predictions[timeframe][best_model][:7]
                
                for i, price in enumerate(predictions):
                    if timeframe == '4h':
                        period = f"{(i+1)*4}h"
                    elif timeframe == '1d':
                        period = f"Day {i+1}"
                    else:
                        period = f"Week {i+1}"
                    
                    change = ((price / predictor.reference_price - 1) * 100)
                    
                    pred_data.append({
                        'Period': period,
                        'Predicted Price': f"${price:.2f}",
                        'Change': f"{change:+.2f}%",
                        'Trend': '📈' if change > 0 else '📉' if change < 0 else '➡️'
                    })
            
            if pred_data:
                df_pred = pd.DataFrame(pred_data)
                st.dataframe(df_pred, use_container_width=True, hide_index=True)
                
                fig = go.Figure()
                
                prices = [float(p['Predicted Price'].replace('$', '')) for p in pred_data]
                periods = [p['Period'] for p in pred_data]
                
                fig.add_trace(go.Scatter(
                    x=periods,
                    y=prices,
                    mode='lines+markers',
                    name='Prediction',
                    line=dict(color='#3498db', width=3),
                    marker=dict(size=10)
                ))
                
                fig.add_hline(
                    y=predictor.reference_price,
                    line_dash="dash",
                    line_color="red",
                    annotation_text=f"Current: ${predictor.reference_price:.2f}"
                )
                
                fig.update_layout(
                    title=f"{timeframe.upper()} Price Predictions",
                    xaxis_title="Period",
                    yaxis_title="Price ($)",
                    template='plotly_dark',
                    height=500,
                    dragmode='pan'
                )
                
                fig.update_xaxes(fixedrange=False)
                fig.update_yaxes(fixedrange=False)
                
                st.plotly_chart(fig, use_container_width=True, config={
                    'scrollZoom': True,
                    'displayModeBar': True,
                    'displaylogo': False
                })
    
    # TAB 6: Final Predictions
    with tab6:
        st.markdown("### 🎯 Final Predictions Summary")
        
        final_data = []
        
        for timeframe in ['4h', '1d', '1w']:
            if timeframe not in all_predictions:
                continue
            
            best_model = predictor.best_models.get(timeframe)
            if not best_model or best_model not in all_predictions[timeframe]:
                continue
            
            predictions = all_predictions[timeframe][best_model][:7]
            
            for i, price in enumerate(predictions):
                if timeframe == '4h':
                    period = f"{(i+1)*4} hours"
                elif timeframe == '1d':
                    period = f"Day {i+1}"
                else:
                    period = f"Week {i+1}"
                
                change = ((price / predictor.reference_price - 1) * 100)
                
                final_data.append({
                    'Timeframe': timeframe.upper(),
                    'Period': period,
                    'Predicted Price': f"${price:.2f}",
                    'Change': f"{change:+.2f}%",
                    'Trend': '📈' if change > 0 else '📉'
                })
        
        if final_data:
            df_final = pd.DataFrame(final_data)
            st.dataframe(df_final, use_container_width=True, hide_index=True)
            
            st.markdown("### 📊 All Timeframes Comparison")
            
            fig = go.Figure()
            
            for timeframe in ['4h', '1d', '1w']:
                if timeframe in all_predictions:
                    best_model = predictor.best_models.get(timeframe)
                    if best_model:
                        predictions = all_predictions[timeframe][best_model][:7]
                        x = list(range(1, len(predictions) + 1))
                        
                        fig.add_trace(go.Scatter(
                            x=x,
                            y=predictions,
                            mode='lines+markers',
                            name=timeframe.upper(),
                            line=dict(width=3),
                            marker=dict(size=8)
                        ))
            
            fig.add_hline(
                y=predictor.reference_price,
                line_dash="dash",
                line_color="purple",
                annotation_text=f"Current: ${predictor.reference_price:.2f}"
            )
            
            fig.update_layout(
                title="Final Price Predictions - All Timeframes",
                xaxis_title="Period",
                yaxis_title="Price ($)",
                template='plotly_dark',
                height=600,
                dragmode='pan'
            )
            
            fig.update_xaxes(fixedrange=False)
            fig.update_yaxes(fixedrange=False)
            
            st.plotly_chart(fig, use_container_width=True, config={
                'scrollZoom': True,
                'displayModeBar': True,
                'displaylogo': False
            })

# Footer
st.markdown("---")
st.markdown(
    f"<div style='text-align: center; color: #7f8c8d; font-size: 14px;'>"
    f"🔮 Crypto Prediction | Last update: {datetime.now().strftime('%H:%M:%S')} | "
    f"Powered by Streamlit & Binance WebSocket"
    f"</div>",
    unsafe_allow_html=True
)

# Cleanup WebSocket on app close
import atexit
def cleanup():
    if 'ws_handler' in st.session_state:
        st.session_state.ws_handler.stop()
        print("🛑 WebSocket stopped")

atexit.register(cleanup)
