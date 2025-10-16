"""
streamlit_app.py
Streamlit web application for crypto prediction
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from datetime import datetime
import pytz
from eth import AdvancedETHPredictor

# Page config
st.set_page_config(
    page_title="🔮 Crypto Prediction - Real-time",
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
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: #e9ecef;
        border-radius: 5px;
        padding: 10px 20px;
        font-weight: bold;
    }
    .stTabs [aria-selected="true"] {
        background-color: #27ae60;
        color: white;
    }
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .ticker-bar {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #dee2e6;
        margin-bottom: 20px;
    }
    .stButton>button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'predictor' not in st.session_state:
    st.session_state.predictor = None
if 'all_predictions' not in st.session_state:
    st.session_state.all_predictions = None
if 'prices' not in st.session_state:
    st.session_state.prices = {}
if 'last_price_update' not in st.session_state:
    st.session_state.last_price_update = 0
if 'analysis_running' not in st.session_state:
    st.session_state.analysis_running = False

# Symbols
SYMBOLS = ["ETHUSDT", "BTCUSDT", "PAXGUSDT", "BNBUSDT", 
           "SOLUSDT", "ADAUSDT", "DOGEUSDT", "ARBUSDT"]

# Cache predictor initialization
@st.cache_resource
def get_predictor(timezone='Asia/Ho_Chi_Minh'):
    """Get cached predictor instance"""
    return AdvancedETHPredictor(timezone=timezone)

# Get real-time prices (polling method - better for Streamlit Cloud)
def update_realtime_prices():
    """Update prices via REST API (more reliable than WebSocket on Streamlit Cloud)"""
    current_time = time.time()
    
    # Update every 3 seconds
    if current_time - st.session_state.last_price_update < 3:
        return
    
    predictor = get_predictor()
    
    for symbol in SYMBOLS:
        try:
            ticker = predictor.get_24h_ticker(symbol)
            if ticker:
                st.session_state.prices[symbol] = ticker
        except:
            pass
    
    st.session_state.last_price_update = current_time

# Display ticker bar
def display_ticker_bar():
    """Display real-time ticker bar"""
    st.markdown('<div class="ticker-bar">', unsafe_allow_html=True)
    
    # Update prices
    update_realtime_prices()
    
    # Create columns for each symbol
    cols = st.columns(len(SYMBOLS))
    
    for idx, symbol in enumerate(SYMBOLS):
        with cols[idx]:
            if symbol in st.session_state.prices:
                data = st.session_state.prices[symbol]
                price = data['price']
                change_pct = data['change_percent']
                
                # Format price
                if price >= 1000:
                    price_str = f"${price:,.2f}"
                else:
                    price_str = f"${price:.4f}"
                
                # Color based on change
                color = "#27ae60" if change_pct >= 0 else "#e74c3c"
                arrow = "▲" if change_pct >= 0 else "▼"
                
                st.markdown(f"""
                <div style='text-align: center; padding: 10px; background: white; border-radius: 5px; cursor: pointer;'>
                    <b>{symbol}</b><br>
                    <span style='font-size: 18px; font-weight: bold;'>{price_str}</span><br>
                    <span style='color: {color}; font-weight: bold;'>{arrow} {change_pct:+.2f}%</span>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style='text-align: center; padding: 10px; background: white; border-radius: 5px;'>
                    <b>{symbol}</b><br>
                    <span style='color: gray;'>Loading...</span>
                </div>
                """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Display current price widget
def display_price_widget(symbol):
    """Display detailed price information"""
    if symbol in st.session_state.prices:
        data = st.session_state.prices[symbol]
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric(
                label=f"💰 {symbol}",
                value=f"${data['price']:,.2f}",
                delta=f"{data['change_percent']:+.2f}%"
            )
        
        with col2:
            st.metric(
                label="📈 24h High",
                value=f"${data['high']:,.2f}"
            )
        
        with col3:
            st.metric(
                label="📉 24h Low",
                value=f"${data['low']:,.2f}"
            )
        
        with col4:
            volume = data['volume']
            if volume >= 1_000_000:
                volume_str = f"{volume/1_000_000:.2f}M"
            elif volume >= 1_000:
                volume_str = f"{volume/1_000:.2f}K"
            else:
                volume_str = f"{volume:,.0f}"
            
            st.metric(
                label="📊 24h Volume",
                value=volume_str
            )
        
        with col5:
            st.metric(
                label="⏰ Last Update",
                value=datetime.now().strftime("%H:%M:%S")
            )
    else:
        st.info(f"Loading {symbol} data...")

# Plot candlestick chart
@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_candlestick_data(symbol, interval, limit=200):
    """Fetch candlestick data with caching"""
    predictor = get_predictor()
    return predictor.fetch_kline_data(symbol, interval, limit)

def plot_candlestick(df, symbol, interval):
    """Plot interactive candlestick chart with Plotly"""
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.7, 0.3],
        subplot_titles=(f'{symbol} - {interval.upper()} Candlestick Chart', 'Volume')
    )
    
    # Candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=df['datetime'],
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
    
    # Volume bars
    colors = ['#26a69a' if row['close'] >= row['open'] else '#ef5350' 
              for idx, row in df.iterrows()]
    
    fig.add_trace(
        go.Bar(
            x=df['datetime'],
            y=df['volume'],
            name='Volume',
            marker_color=colors,
            opacity=0.6
        ),
        row=2, col=1
    )
    
    # Update layout
    fig.update_layout(
        height=700,
        template='plotly_white',
        xaxis_rangeslider_visible=False,
        showlegend=True,
        hovermode='x unified',
        margin=dict(l=50, r=50, t=80, b=50)
    )
    
    fig.update_xaxes(title_text="Time", row=2, col=1)
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    
    return fig

# Plot predictions
def plot_predictions(all_predictions, reference_price, timeframe):
    """Plot prediction comparison chart"""
    if timeframe not in all_predictions:
        return None
    
    fig = go.Figure()
    
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c', '#e67e22']
    
    for idx, (model_name, predictions) in enumerate(all_predictions[timeframe].items()):
        x = list(range(1, len(predictions[:7]) + 1))
        
        if timeframe == '4h':
            labels = [f'{i*4}h' for i in x]
        elif timeframe == '1d':
            labels = [f'D{i}' for i in x]
        else:
            labels = [f'W{i}' for i in x]
        
        fig.add_trace(go.Scatter(
            x=labels,
            y=predictions[:7],
            mode='lines+markers',
            name=model_name,
            line=dict(color=colors[idx % len(colors)], width=3),
            marker=dict(size=10)
        ))
    
    # Reference line
    fig.add_hline(
        y=reference_price,
        line_dash="dash",
        line_color="red",
        line_width=2,
        annotation_text=f"Current: ${reference_price:.2f}",
        annotation_position="right"
    )
    
    fig.update_layout(
        title=dict(
            text=f'{timeframe.upper()} - Price Predictions Comparison',
            font=dict(size=20, color='#2c3e50')
        ),
        xaxis_title='Period',
        yaxis_title='Price ($)',
        height=550,
        template='plotly_white',
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

# Plot model comparison
def plot_model_comparison(predictor, timeframe):
    """Plot R² scores comparison"""
    if timeframe not in predictor.all_model_results:
        return None
    
    results = predictor.all_model_results[timeframe]
    sorted_results = sorted(results.items(), key=lambda x: x[1]['r2'], reverse=True)[:8]
    
    model_names = [name for name, _ in sorted_results]
    r2_scores = [result['r2'] for _, result in sorted_results]
    
    # Color best model differently
    best_model = predictor.best_models.get(timeframe)
    colors_list = ['gold' if name == best_model else 'lightblue' for name in model_names]
    
    fig = go.Figure(data=[
        go.Bar(
            x=r2_scores,
            y=model_names,
            orientation='h',
            marker=dict(color=colors_list),
            text=[f'{score:.4f}' for score in r2_scores],
            textposition='outside'
        )
    ])
    
    fig.update_layout(
        title=f'{timeframe.upper()} - Model Performance (R² Score)',
        xaxis_title='R² Score',
        yaxis_title='Model',
        height=400,
        template='plotly_white',
        xaxis=dict(range=[0, 1])
    )
    
    return fig

# Main app
def main():
    st.title("🔮 Crypto Prediction - Real-time Analysis")
    st.markdown("Advanced cryptocurrency price prediction with machine learning")
    
    # Ticker bar
    display_ticker_bar()
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Control Panel")
        
        symbol = st.selectbox(
            "📊 Select Symbol",
            SYMBOLS,
            index=0,
            help="Choose cryptocurrency to analyze"
        )
        
        timezone = st.selectbox(
            "🌍 Timezone",
            ["Asia/Ho_Chi_Minh", "America/New_York", "Europe/London", 
             "Asia/Tokyo", "Asia/Shanghai"],
            index=0,
            help="Select your timezone"
        )
        
        st.markdown("---")
        
        # Analysis button
        run_analysis = st.button(
            "🚀 Run Analysis", 
            type="primary",
            disabled=st.session_state.analysis_running
        )
        
        if run_analysis:
            st.session_state.analysis_running = True
            
            with st.spinner(f"🔄 Analyzing {symbol}..."):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    # Create predictor
                    status_text.text("Initializing predictor...")
                    progress_bar.progress(10)
                    
                    predictor = AdvancedETHPredictor(timezone=timezone)
                    
                    # Run analysis
                    status_text.text("Fetching data...")
                    progress_bar.progress(30)
                    
                    all_data, all_predictions = predictor.run_analysis(symbol)
                    
                    progress_bar.progress(90)
                    
                    if all_data and all_predictions:
                        st.session_state.predictor = predictor
                        st.session_state.all_predictions = all_predictions
                        progress_bar.progress(100)
                        status_text.empty()
                        st.success("✅ Analysis complete!")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error("❌ Analysis failed - no data returned")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                finally:
                    st.session_state.analysis_running = False
        
        st.markdown("---")
        
        # Status
        st.markdown("### 📊 Status")
        if st.session_state.predictor:
            st.success("🟢 Model Ready")
            ref_price = st.session_state.predictor.reference_price
            if ref_price:
                st.info(f"💰 Ref Price: ${ref_price:.2f}")
        else:
            st.warning("🟡 No Analysis Yet")
        
        # Auto-refresh toggle
        st.markdown("---")
        auto_refresh = st.checkbox("🔄 Auto-refresh prices", value=True)
        
        if auto_refresh:
            st.caption("Prices update every 3 seconds")
    
    # Price widget
    st.markdown("---")
    display_price_widget(symbol)
    
    # Candlestick chart section
    st.markdown("---")
    st.markdown("## 📈 Live Chart")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        interval = st.selectbox(
            "Timeframe",
            ['15m', '1h', '4h', '1d', '1w'],
            index=2,
            key='chart_interval'
        )
    
    with col2:
        if st.button("🔄 Refresh Chart"):
            st.cache_data.clear()
            st.rerun()
    
    # Fetch and display candlestick data
    try:
        with st.spinner("Loading chart..."):
            df = get_candlestick_data(symbol, interval, limit=200)
            
            if df is not None and len(df) > 0:
                fig = plot_candlestick(df, symbol, interval)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No chart data available")
    except Exception as e:
        st.error(f"Error loading chart: {e}")
    
    # Analysis results
    if st.session_state.predictor and st.session_state.all_predictions:
        st.markdown("---")
        st.markdown("## 📊 Analysis Results")
        
        predictor = st.session_state.predictor
        all_predictions = st.session_state.all_predictions
        
        # Create tabs
        tab_names = ["📊 Summary", "⏰ 4H", "⏰ 1D", "⏰ 1W", "🏆 Comparison", "🎯 Final Predictions"]
        tabs = st.tabs(tab_names)
        
        # Tab 1: Summary
        with tabs[0]:
            st.markdown("### 🏆 Best Models Performance")
            
            summary_data = []
            for timeframe in ['4h', '1d', '1w']:
                if timeframe in predictor.best_models:
                    best_model = predictor.best_models[timeframe]
                    result = predictor.all_model_results[timeframe][best_model]
                    
                    summary_data.append({
                        'Timeframe': timeframe.upper(),
                        'Best Model': best_model,
                        'R² Score': f"{result['r2']:.4f}",
                        'MAE ($)': f"${result['mae']:.2f}",
                        'RMSE ($)': f"${result['rmse']:.2f}",
                        'Direction Acc': f"{result['direction_accuracy']:.2%}"
                    })
            
            if summary_data:
                df_summary = pd.DataFrame(summary_data)
                st.dataframe(
                    df_summary,
                    use_container_width=True,
                    hide_index=True
                )
                
                # Quick insights
                st.markdown("### 💡 Quick Insights")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    avg_r2 = np.mean([float(d['R² Score']) for d in summary_data])
                    st.metric("Average R² Score", f"{avg_r2:.4f}")
                
                with col2:
                    best_tf = max(summary_data, key=lambda x: float(x['R² Score']))
                    st.metric("Best Timeframe", best_tf['Timeframe'])
                
                with col3:
                    models_used = len(set(d['Best Model'] for d in summary_data))
                    st.metric("Unique Models", models_used)
        
        # Tab 2-4: Timeframe results
        for tab_idx, timeframe in enumerate(['4h', '1d', '1w'], start=1):
            with tabs[tab_idx]:
                if timeframe in predictor.all_model_results:
                    
                    # Model performance table
                    st.markdown(f"### 🎯 {timeframe.upper()} - Model Performance")
                    
                    results = predictor.all_model_results[timeframe]
                    best_model = predictor.best_models.get(timeframe)
                    
                    perf_data = []
                    for model_name, result in sorted(results.items(), 
                                                    key=lambda x: x[1]['r2'], 
                                                    reverse=True):
                        status = "🏆" if model_name == best_model else ""
                        
                        perf_data.append({
                            'Status': status,
                            'Model': model_name,
                            'R²': f"{result['r2']:.4f}",
                            'MAE ($)': f"${result['mae']:.2f}",
                            'RMSE ($)': f"${result['rmse']:.2f}",
                            'MAPE (%)': f"{result['mape']:.2f}%",
                            'Direction Acc': f"{result['direction_accuracy']:.2%}"
                        })
                    
                    df_perf = pd.DataFrame(perf_data)
                    st.dataframe(
                        df_perf,
                        use_container_width=True,
                        hide_index=True
                    )
                    
                    # Model comparison chart
                    st.markdown(f"### 📊 {timeframe.upper()} - Model Comparison")
                    fig_comp = plot_model_comparison(predictor, timeframe)
                    if fig_comp:
                        st.plotly_chart(fig_comp, use_container_width=True)
                    
                    # Predictions chart
                    st.markdown(f"### 📈 {timeframe.upper()} - Price Predictions")
                    fig_pred = plot_predictions(all_predictions, predictor.reference_price, timeframe)
                    if fig_pred:
                        st.plotly_chart(fig_pred, use_container_width=True)
                    
                    # Predictions table
                    if timeframe in all_predictions:
                        st.markdown(f"### 📋 {timeframe.upper()} - Detailed Predictions")
                        
                        pred_table_data = []
                        
                        for model_name, predictions in all_predictions[timeframe].items():
                            for i in range(min(len(predictions), 7)):
                                if timeframe == '4h':
                                    period = f"{(i+1)*4}h"
                                elif timeframe == '1d':
                                    period = f"Day {i+1}"
                                else:
                                    period = f"Week {i+1}"
                                
                                price = predictions[i]
                                change = ((price / predictor.reference_price - 1) * 100)
                                
                                pred_table_data.append({
                                    'Model': model_name,
                                    'Period': period,
                                    'Price': f"${price:.2f}",
                                    'Change %': f"{change:+.2f}%"
                                })
                        
                        if pred_table_data:
                            df_pred = pd.DataFrame(pred_table_data)
                            st.dataframe(
                                df_pred,
                                use_container_width=True,
                                hide_index=True
                            )
        
        # Tab 5: Comparison
        with tabs[4]:
            st.markdown("### 🏅 Model Rankings Across Timeframes")
            
            for timeframe in ['4h', '1d', '1w']:
                if timeframe in predictor.all_model_results:
                    st.markdown(f"#### {timeframe.upper()}")
                    
                    results = predictor.all_model_results[timeframe]
                    sorted_models = sorted(results.items(), 
                                         key=lambda x: x[1]['r2'], 
                                         reverse=True)[:5]
                    
                    rank_data = []
                    for rank, (model_name, result) in enumerate(sorted_models, 1):
                        medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else f"#{rank}"
                        
                        rank_data.append({
                            'Rank': medal,
                            'Model': model_name,
                            'R²': f"{result['r2']:.4f}",
                            'MAE': f"${result['mae']:.2f}",
                            'RMSE': f"${result['rmse']:.2f}",
                            'Direction Acc': f"{result['direction_accuracy']:.2%}"
                        })
                    
                    df_rank = pd.DataFrame(rank_data)
                    st.dataframe(
                        df_rank,
                        use_container_width=True,
                        hide_index=True
                    )
                    
                    st.markdown("---")
        
        # Tab 6: Final Predictions
        with tabs[5]:
            st.markdown("### 🎯 Final Predictions Summary (Best Models)")
            
            pred_data = []
            for timeframe in ['4h', '1d', '1w']:
                if timeframe not in all_predictions:
                    continue
                
                best_model = predictor.best_models.get(timeframe)
                if not best_model or best_model not in all_predictions[timeframe]:
                    continue
                
                predictions = all_predictions[timeframe][best_model]
                
                for i in range(min(len(predictions), 7)):
                    if timeframe == '4h':
                        period = f"{(i+1)*4} hours"
                    elif timeframe == '1d':
                        period = f"Day {i+1}"
                    else:
                        period = f"Week {i+1}"
                    
                    price = predictions[i]
                    change = ((price / predictor.reference_price - 1) * 100)
                    trend = "📈" if change > 0 else "📉" if change < 0 else "➡️"
                    
                    pred_data.append({
                        'Timeframe': timeframe.upper(),
                        'Period': period,
                        'Model': best_model,
                        'Predicted Price': f"${price:.2f}",
                        'Change': f"{change:+.2f}%",
                        'Trend': trend
                    })
            
            if pred_data:
                df_final = pd.DataFrame(pred_data)
                st.dataframe(
                    df_final,
                    use_container_width=True,
                    hide_index=True
                )
                
                # Download predictions
                csv = df_final.to_csv(index=False)
                st.download_button(
                    label="📥 Download Predictions (CSV)",
                    data=csv,
                    file_name=f"{symbol}_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #7f8c8d; padding: 20px;'>
        <p>🔮 <b>Crypto Prediction App</b> | Powered by Machine Learning</p>
        <p><small>⚠️ Disclaimer: This is for educational purposes only. Not financial advice.</small></p>
    </div>
    """, unsafe_allow_html=True)
    
    # Auto-refresh for real-time updates
    if 'auto_refresh' in locals() and auto_refresh:
        time.sleep(3)
        st.rerun()

if __name__ == "__main__":
    main()
