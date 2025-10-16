import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import requests
from datetime import datetime

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
        background-color: #f5f6fa;
        font-size: 16px !important;
    }
    
    .ticker-container {
        background-color: white;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 10px;
    }
    
    .ticker-symbol {
        color: #000000 !important;
        font-weight: bold;
        font-size: 16px;
        margin-bottom: 5px;
    }
    
    .ticker-price-up {
        color: #27ae60 !important;
        font-weight: bold;
        font-size: 20px;
    }
    
    .ticker-price-down {
        color: #e74c3c !important;
        font-weight: bold;
        font-size: 20px;
    }
    
    .ticker-change-up {
        color: #27ae60 !important;
        font-size: 14px;
    }
    
    .ticker-change-down {
        color: #e74c3c !important;
        font-size: 14px;
    }
    
    .control-symbol {
        color: #000000 !important;
        font-weight: bold;
        font-size: 14px;
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
    
    p, span, div {
        font-size: 16px;
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
    }
    
    .stButton button {
        font-size: 15px;
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

@st.cache_data(ttl=1)
def get_ticker(symbol):
    """Get ticker from Binance"""
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

# ============================================
# HEADER
# ============================================
st.title("🔮 Crypto Prediction - Real-time")

col1, col2 = st.columns([5, 1])
with col1:
    st.markdown("### Real-time Market Data")
with col2:
    auto_refresh = st.checkbox("Auto-refresh (1s)", value=True)

# ============================================
# TICKER BAR
# ============================================
st.markdown("---")

for row in range(0, len(SYMBOLS), 4):
    ticker_cols = st.columns(4)
    
    for idx, symbol in enumerate(SYMBOLS[row:row+4]):
        with ticker_cols[idx]:
            ticker_data = get_ticker(symbol)
            if ticker_data:
                change_pct = ticker_data['change_percent']
                is_up = change_pct >= 0
                
                price_class = "ticker-price-up" if is_up else "ticker-price-down"
                change_class = "ticker-change-up" if is_up else "ticker-change-down"
                arrow = "▲" if is_up else "▼"
                
                html = f"""
                <div class="ticker-container">
                    <div class="ticker-symbol">{symbol}</div>
                    <div class="{price_class}">${ticker_data['price']:,.2f}</div>
                    <div class="{change_class}">{arrow} {abs(change_pct):.2f}%</div>
                </div>
                """
                st.markdown(html, unsafe_allow_html=True)
                
                if st.button(f"📊 Chart", key=f"chart_{symbol}_{row}"):
                    st.session_state.show_chart = True
                    st.session_state.chart_symbol = symbol
                    st.rerun()

# ============================================
# CANDLESTICK CHART - FIXED CLOSE BUTTON
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
        
        # ENABLE ZOOM & PAN
        fig.update_layout(
            height=700,
            template='plotly_dark',
            xaxis_rangeslider_visible=False,
            showlegend=False,
            hovermode='x unified',
            dragmode='zoom',  # Enable zoom by default
            modebar_add=['pan2d', 'zoom2d', 'zoomIn2d', 'zoomOut2d', 'autoScale2d', 'resetScale2d']
        )
        
        # Enable drag to zoom and scroll to zoom
        fig.update_xaxes(fixedrange=False)
        fig.update_yaxes(fixedrange=False)
        
        fig.update_xaxes(title_text="Time", row=2, col=1)
        fig.update_yaxes(title_text="Price ($)", row=1, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1)
        
        st.plotly_chart(fig, use_container_width=True, config={
            'scrollZoom': True,
            'displayModeBar': True,
            'modeBarButtonsToAdd': ['drawline', 'drawopenpath', 'eraseshape'],
            'displaylogo': False
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
    # STOP AUTO-REFRESH when chart is open
    st.stop()

# ============================================
# CONTROL PANEL
# ============================================
st.markdown("### 🎛️ Control Panel")

col1, col2, col3, col4 = st.columns([2, 2, 2, 2])

with col1:
    selected_symbol = st.selectbox("📊 Symbol", SYMBOLS)

with col2:
    timezone = st.selectbox(
        "🌍 Timezone",
        ["Asia/Ho_Chi_Minh", "America/New_York", "Europe/London", "Asia/Tokyo"],
        index=0
    )

with col3:
    st.write("")
    st.write("")
    run_analysis = st.button("🚀 Run Analysis", type="primary", use_container_width=True)

with col4:
    current_ticker = get_ticker(selected_symbol)
    if current_ticker:
        change_pct = current_ticker['change_percent']
        is_up = change_pct >= 0
        price_class = "control-price-up" if is_up else "control-price-down"
        
        st.markdown(f"""
        <div style="background: white; padding: 10px; border-radius: 5px; margin-top: 25px;">
            <div class="control-symbol">{selected_symbol}</div>
            <div class="{price_class}">${current_ticker['price']:,.2f}</div>
            <div style="color: {'#27ae60' if is_up else '#e74c3c'}; font-size: 14px;">
                {'▲' if is_up else '▼'} {abs(change_pct):.2f}%
            </div>
        </div>
        """, unsafe_allow_html=True)

# ============================================
# RUN ANALYSIS
# ============================================
if run_analysis:
    with st.spinner(f"🔄 Analyzing {selected_symbol}..."):
        try:
            predictor = AdvancedETHPredictor(timezone=timezone)
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text("Fetching data...")
            progress_bar.progress(20)
            
            all_data, all_predictions = predictor.run_analysis(selected_symbol)
            
            progress_bar.progress(100)
            status_text.text("✅ Analysis complete!")
            
            st.session_state.predictor = predictor
            st.session_state.predictions = all_predictions
            
            time.sleep(1)
            progress_bar.empty()
            status_text.empty()
            
            st.success(f"✅ Analysis completed for {selected_symbol}!")
            
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

# ============================================
# RESULTS DISPLAY - ONLY ONCE
# ============================================
if st.session_state.predictor is not None and st.session_state.predictions is not None:
    predictor = st.session_state.predictor
    all_predictions = st.session_state.predictions
    
    st.markdown("---")
    st.markdown("## 📊 Analysis Results")
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Summary",
        "⏰ 4H Predictions",
        "📅 1D Predictions",
        "📆 1W Predictions",
        "🎯 Final Predictions"
    ])
    
    with tab1:
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
        
        fig.update_layout(height=800, showlegend=False, template='plotly_white')
        st.plotly_chart(fig, use_container_width=True)
    
    for tab, timeframe in zip([tab2, tab3, tab4], ['4h', '1d', '1w']):
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
                
                # ENABLE ZOOM & PAN
                fig.update_layout(
                    title=f"{timeframe.upper()} Price Predictions",
                    xaxis_title="Period",
                    yaxis_title="Price ($)",
                    template='plotly_white',
                    height=500,
                    dragmode='zoom'
                )
                
                fig.update_xaxes(fixedrange=False)
                fig.update_yaxes(fixedrange=False)
                
                st.plotly_chart(fig, use_container_width=True, config={
                    'scrollZoom': True,
                    'displayModeBar': True,
                    'displaylogo': False
                })
    
    with tab5:
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
            
            # ENABLE ZOOM & PAN
            fig.update_layout(
                title="Final Price Predictions - All Timeframes",
                xaxis_title="Period",
                yaxis_title="Price ($)",
                template='plotly_white',
                height=600,
                dragmode='zoom'
            )
            
            fig.update_xaxes(fixedrange=False)
            fig.update_yaxes(fixedrange=False)
            
            st.plotly_chart(fig, use_container_width=True, config={
                'scrollZoom': True,
                'displayModeBar': True,
                'displaylogo': False
            })
    
    # STOP AUTO-REFRESH when showing results
    st.stop()

# ============================================
# AUTO-REFRESH
# ============================================
if auto_refresh:
    time.sleep(1)
    st.rerun()

# Footer
st.markdown("---")
st.markdown(
    f"<div style='text-align: center; color: #7f8c8d; font-size: 14px;'>"
    f"🔮 Crypto Prediction | Last update: {datetime.now().strftime('%H:%M:%S')} | "
    f"Powered by Streamlit & Binance API"
    f"</div>",
    unsafe_allow_html=True
)
