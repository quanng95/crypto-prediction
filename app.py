import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from datetime import datetime
import requests

# Import predictor
from eth import AdvancedETHPredictor

# Page config
st.set_page_config(
    page_title="🔮 Crypto Prediction",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        background-color: #f5f6fa;
    }
    .stMetric {
        background-color: white;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .ticker-positive {
        color: #27ae60;
        font-weight: bold;
    }
    .ticker-negative {
        color: #e74c3c;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'predictor' not in st.session_state:
    st.session_state.predictor = None
if 'predictions' not in st.session_state:
    st.session_state.predictions = None
if 'last_update' not in st.session_state:
    st.session_state.last_update = time.time()

# Symbols
SYMBOLS = ["ETHUSDT", "BTCUSDT", "PAXGUSDT", "BNBUSDT", 
           "SOLUSDT", "ADAUSDT", "DOGEUSDT", "ARBUSDT"]


def get_binance_ticker(symbol):
    """Lấy ticker data từ Binance"""
    try:
        url = f"https://api.binance.com/api/v3/ticker/24hr"
        response = requests.get(url, params={'symbol': symbol}, timeout=5)
        data = response.json()
        return {
            'price': float(data['lastPrice']),
            'change': float(data['priceChange']),
            'change_percent': float(data['priceChangePercent']),
            'high': float(data['highPrice']),
            'low': float(data['lowPrice']),
            'volume': float(data['volume'])
        }
    except:
        return None


def format_volume(volume):
    """Format volume"""
    if volume >= 1_000_000:
        return f"{volume/1_000_000:.2f}M"
    elif volume >= 1_000:
        return f"{volume/1_000:.2f}K"
    return f"{volume:.0f}"


# ============================================
# HEADER
# ============================================
st.title("🔮 Crypto Prediction - Real-time")

# Connection status
col1, col2, col3 = st.columns([3, 1, 1])
with col1:
    st.markdown("### Real-time Market Data")
with col2:
    if st.button("🔄 Refresh Now"):
        st.rerun()
with col3:
    auto_refresh = st.checkbox("Auto-refresh (5s)", value=True)

# ============================================
# TICKER BAR
# ============================================
st.markdown("---")
ticker_cols = st.columns(len(SYMBOLS))

for idx, symbol in enumerate(SYMBOLS):
    with ticker_cols[idx]:
        ticker_data = get_binance_ticker(symbol)
        if ticker_data:
            change_pct = ticker_data['change_percent']
            color_class = "ticker-positive" if change_pct >= 0 else "ticker-negative"
            arrow = "▲" if change_pct >= 0 else "▼"
            
            st.metric(
                label=symbol,
                value=f"${ticker_data['price']:,.2f}",
                delta=f"{arrow} {abs(change_pct):.2f}%"
            )

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
        index=0
    )

with col2:
    timezone = st.selectbox(
        "🌍 Timezone",
        ["Asia/Ho_Chi_Minh", "America/New_York", "Europe/London", "Asia/Tokyo"],
        index=0
    )

with col3:
    st.write("")  # Spacing
    st.write("")
    run_analysis = st.button("🚀 Run Analysis", use_container_width=True, type="primary")

with col4:
    # Current price display
    current_ticker = get_binance_ticker(selected_symbol)
    if current_ticker:
        st.metric(
            label="Current Price",
            value=f"${current_ticker['price']:,.2f}",
            delta=f"{current_ticker['change_percent']:+.2f}%"
        )

# ============================================
# RUN ANALYSIS
# ============================================
if run_analysis:
    with st.spinner(f"🔄 Analyzing {selected_symbol}..."):
        try:
            # Create predictor
            predictor = AdvancedETHPredictor(timezone=timezone)
            
            # Progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text("Fetching data...")
            progress_bar.progress(20)
            
            # Run analysis
            all_data, all_predictions = predictor.run_analysis(selected_symbol)
            
            progress_bar.progress(100)
            status_text.text("✅ Analysis complete!")
            
            # Store in session state
            st.session_state.predictor = predictor
            st.session_state.predictions = all_predictions
            
            time.sleep(1)
            progress_bar.empty()
            status_text.empty()
            
            st.success(f"✅ Analysis completed for {selected_symbol}!")
            
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

# ============================================
# RESULTS DISPLAY
# ============================================
if st.session_state.predictor and st.session_state.predictions:
    predictor = st.session_state.predictor
    all_predictions = st.session_state.predictions
    
    st.markdown("---")
    st.markdown("## 📊 Analysis Results")
    
    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Summary",
        "⏰ 4H Predictions",
        "📅 1D Predictions",
        "📆 1W Predictions",
        "🎯 Final Predictions"
    ])
    
    # TAB 1: SUMMARY
    with tab1:
        st.markdown("### 🏆 Best Models Performance")
        
        # Performance table
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
        
        # Charts
        st.markdown("### 📊 Performance Visualization")
        
        # Create subplots
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
        
        # R² Score
        fig.add_trace(
            go.Bar(x=timeframes, y=r2_scores, name='R² Score',
                   marker_color=['#2ecc71', '#3498db', '#9b59b6']),
            row=1, col=1
        )
        
        # MAE
        fig.add_trace(
            go.Bar(x=timeframes, y=mae_values, name='MAE',
                   marker_color=['#e74c3c', '#f39c12', '#1abc9c']),
            row=1, col=2
        )
        
        # Direction Accuracy
        fig.add_trace(
            go.Bar(x=timeframes, y=dir_acc, name='Accuracy (%)',
                   marker_color=['#16a085', '#27ae60', '#2980b9']),
            row=2, col=1
        )
        
        # Model Distribution
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
    
    # TAB 2-4: TIMEFRAME PREDICTIONS
    for tab, timeframe in zip([tab2, tab3, tab4], ['4h', '1d', '1w']):
        with tab:
            if timeframe not in all_predictions:
                st.warning(f"No predictions for {timeframe}")
                continue
            
            st.markdown(f"### 🎯 {timeframe.upper()} Predictions")
            
            # Predictions table
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
                
                # Chart
                fig = go.Figure()
                
                prices = [p['Predicted Price'].replace('$', '') for p in pred_data]
                prices = [float(p) for p in prices]
                periods = [p['Period'] for p in pred_data]
                
                fig.add_trace(go.Scatter(
                    x=periods,
                    y=prices,
                    mode='lines+markers',
                    name='Prediction',
                    line=dict(color='#3498db', width=3),
                    marker=dict(size=10)
                ))
                
                # Current price line
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
                    template='plotly_white',
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    # TAB 5: FINAL PREDICTIONS
    with tab5:
        st.markdown("### 🎯 Final Predictions Summary")
        
        # Consolidated predictions
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
            
            # Combined chart
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
            
            # Current price
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
                template='plotly_white',
                height=600
            )
            
            st.plotly_chart(fig, use_container_width=True)

# ============================================
# AUTO-REFRESH
# ============================================
if auto_refresh:
    time.sleep(5)
    st.rerun()

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #7f8c8d;'>"
    "🔮 Crypto Prediction App | Powered by Streamlit"
    "</div>",
    unsafe_allow_html=True
)
