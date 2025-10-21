import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from methodology import render_methodology_tab

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
    
    if timeframe == '15m':
        short_term = predictions[:2]
        mid_term = predictions[2:4]
        
        avg_short = np.mean(short_term)
        avg_mid = np.mean(mid_term)
        
        short_change = ((avg_short / current_price) - 1) * 100
        mid_change = ((avg_mid / current_price) - 1) * 100
        
        if short_change > 0.5 and mid_change > 0.8:
            signal = "LONG"
            confidence = min(90, 50 + abs(short_change) * 10)
            bull_prob = min(90, 50 + abs(short_change) * 8)
            bear_prob = 100 - bull_prob
        elif short_change < -0.5 and mid_change < -0.8:
            signal = "SHORT"
            confidence = min(90, 50 + abs(short_change) * 10)
            bear_prob = min(90, 50 + abs(short_change) * 8)
            bull_prob = 100 - bear_prob
        else:
            signal = "NEUTRAL"
            confidence = 50
            bull_prob = 50
            bear_prob = 50
        
        if signal == "LONG":
            entry = current_price * 0.999
            stop_loss = entry * 0.995
            tp1 = entry * 1.005
            tp2 = entry * 1.010
            tp3 = entry * 1.015
        elif signal == "SHORT":
            entry = current_price * 1.001
            stop_loss = entry * 1.005
            tp1 = entry * 0.995
            tp2 = entry * 0.990
            tp3 = entry * 0.985
        else:
            entry = current_price
            stop_loss = current_price * 0.995
            tp1 = current_price * 1.005
            tp2 = current_price * 1.010
            tp3 = current_price * 1.015
    
    else:
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

def render_trading_signals_tab(predictor):
    """Render Trading Signals tab"""
    st.markdown("### 🎯 Trading Signals & Recommendations")
    
    for timeframe in ['15m', '4h', '1d', '1w']:
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
            
            timeframe_label = timeframe.upper()
            
            st.markdown(f"""
            <div class="{box_class}">
                <h3 style="color: {signal_color};">{signal_emoji} {timeframe_label} - {signal} Signal</h3>
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

def render_summary_tab(predictor):
    """Render Summary tab"""
    st.markdown("### 🏆 Best Models Performance")
    
    perf_data = []
    for timeframe in ['15m', '4h', '1d', '1w']:
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
    
    for tf in ['15m', '4h', '1d', '1w']:
        if tf in predictor.best_models:
            best_model = predictor.best_models[tf]
            result = predictor.all_model_results[tf][best_model]
            timeframes.append(tf.upper())
            r2_scores.append(result['r2'])
            mae_values.append(result['mae'])
            dir_acc.append(result['direction_accuracy'] * 100)
    
    fig.add_trace(
        go.Bar(x=timeframes, y=r2_scores, name='R² Score',
               marker_color=['#f39c12', '#2ecc71', '#3498db', '#9b59b6']),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Bar(x=timeframes, y=mae_values, name='MAE',
               marker_color=['#e67e22', '#e74c3c', '#f39c12', '#1abc9c']),
        row=1, col=2
    )
    
    fig.add_trace(
        go.Bar(x=timeframes, y=dir_acc, name='Accuracy (%)',
               marker_color=['#d35400', '#16a085', '#27ae60', '#2980b9']),
        row=2, col=1
    )
    
    model_names = [predictor.best_models.get(tf, 'N/A') 
                  for tf in ['15m', '4h', '1d', '1w'] 
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

def render_timeframe_predictions_tab(predictor, all_predictions, timeframe):
    """Render predictions for specific timeframe"""
    if timeframe not in all_predictions:
        st.warning(f"No predictions for {timeframe}")
        return
    
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

def render_final_predictions_tab(predictor, all_predictions):
    """Render Final Predictions tab"""
    st.markdown("### 🎯 Final Predictions Summary")
    
    final_data = []
    
    for timeframe in ['15m', '4h', '1d', '1w']:
        if timeframe not in all_predictions:
            continue
        
        best_model = predictor.best_models.get(timeframe)
        if not best_model or best_model not in all_predictions[timeframe]:
            continue
        
        predictions = all_predictions[timeframe][best_model][:7]
        
        for i, price in enumerate(predictions):
            if timeframe == '15m':
                period = f"{(i+1)*15} minutes"
            elif timeframe == '4h':
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
        
        colors = {
            '15m': '#f39c12',
            '4h': '#3498db',
            '1d': '#2ecc71',
            '1w': '#9b59b6'
        }
        
        for timeframe in ['15m', '4h', '1d', '1w']:
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
                        line=dict(width=3, color=colors.get(timeframe, '#ffffff')),
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

def render_all_tabs(predictor, all_predictions):
    """Render all tabs"""
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "🎯 Trading Signals",
        "📈 Summary",
        "⏰ 4H Predictions",
        "📅 1D Predictions",
        "📆 1W Predictions",
        "🔮 Final Predictions",
        "📚 Methodology"
    ])
    
    with tab1:
        render_trading_signals_tab(predictor)
    
    with tab2:
        render_summary_tab(predictor)
    
    with tab3:
        render_timeframe_predictions_tab(predictor, all_predictions, '4h')
    
    with tab4:
        render_timeframe_predictions_tab(predictor, all_predictions, '1d')
    
    with tab5:
        render_timeframe_predictions_tab(predictor, all_predictions, '1w')
    
    with tab6:
        render_final_predictions_tab(predictor, all_predictions)
    
    with tab7:
        render_methodology_tab(predictor)
