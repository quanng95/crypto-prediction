import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def render_methodology_tab(predictor):
    """Render comprehensive methodology explanation tab"""
    
    st.markdown("### 📚 Methodology & Technical Documentation")
    
    st.markdown("""
    This section provides a comprehensive overview of the machine learning models, 
    technical indicators, and mathematical formulas used in our cryptocurrency price prediction system.
    """)
    
    # ============================================
    # SECTION 1: OVERVIEW
    # ============================================
    st.markdown("---")
    st.markdown("## 🎯 1. System Overview")
    
    st.markdown("""
    Our prediction system employs an **Ensemble Machine Learning** approach, combining multiple algorithms 
    to generate accurate price forecasts across different timeframes (15M, 4H, 1D, 1W).
    
    **Key Components:**
    - **10 Machine Learning Models** (Tree-based & Linear/Neural)
    - **70+ Technical Indicators** (Trend, Momentum, Volatility, Volume)
    - **Multi-Timeframe Analysis** (15-minute to 1-week predictions)
    - **Recursive Prediction** (Each step recalculates all indicators)
    - **Risk Management System** (Entry, Stop Loss, Take Profit calculations)
    """)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Models", "10", help="Different ML algorithms")
    with col2:
        st.metric("Technical Indicators", "70+", help="Features for prediction")
    with col3:
        st.metric("Timeframes", "4", help="15M, 4H, 1D, 1W")
    
    # ============================================
    # SECTION 2: MACHINE LEARNING MODELS
    # ============================================
    st.markdown("---")
    st.markdown("## 🤖 2. Machine Learning Models")
    
    st.markdown("### 2.1 Tree-Based Models (No Scaling Required)")
    
    models_tree = pd.DataFrame({
        'Model': ['Random Forest', 'Gradient Boosting', 'Extra Trees', 'AdaBoost'],
        'Type': ['Ensemble', 'Boosting', 'Ensemble', 'Boosting'],
        'N Estimators': [200, 200, 200, 100],
        'Max Depth': [15, 8, 15, 'N/A'],
        'Key Strength': [
            'Robust to outliers, captures non-linear patterns',
            'Sequential learning from errors, high accuracy',
            'Random splits, reduces overfitting',
            'Focuses on hard-to-predict samples'
        ]
    })
    
    st.dataframe(models_tree, use_container_width=True, hide_index=True)
    
    st.markdown("""
    **Random Forest Formula:**
    """)
    st.latex(r"""
    \hat{y} = \frac{1}{B} \sum_{b=1}^{B} T_b(x)
    """)
    st.markdown("""
    Where:
    - $$B$$ = Number of trees (200)
    - $$T_b(x)$$ = Prediction from tree $$b$$
    - $$\hat{y}$$ = Final averaged prediction
    """)
    
    st.markdown("### 2.2 Linear & Neural Models (Scaling Required)")
    
    models_linear = pd.DataFrame({
        'Model': ['Linear Regression', 'Ridge', 'Lasso', 'ElasticNet', 'SVR', 'Neural Network'],
        'Regularization': ['None', 'L2 (α=1.0)', 'L1 (α=1.0)', 'L1+L2 (α=1.0)', 'RBF Kernel', 'Dropout'],
        'Scaler': ['StandardScaler'] * 6,
        'Key Strength': [
            'Fast, interpretable baseline',
            'Prevents overfitting via L2',
            'Feature selection via L1',
            'Combines L1 and L2 benefits',
            'Non-linear kernel mapping',
            'Deep learning, complex patterns'
        ]
    })
    
    st.dataframe(models_linear, use_container_width=True, hide_index=True)
    
    st.markdown("""
    **Ridge Regression Formula:**
    """)
    st.latex(r"""
    \min_{\beta} \left\{ \sum_{i=1}^{n} (y_i - X_i\beta)^2 + \alpha \sum_{j=1}^{p} \beta_j^2 \right\}
    """)
    st.markdown("""
    Where:
    - $$\alpha$$ = Regularization strength (1.0)
    - $$\beta$$ = Model coefficients
    - $$X_i$$ = Feature vector
    - $$y_i$$ = Target price
    """)
    
    st.markdown("""
    **Neural Network Architecture:**
    """)
    st.latex(r"""
    \text{Input}(70+) \rightarrow \text{Hidden}_1(100) \rightarrow \text{Hidden}_2(50) \rightarrow \text{Output}(1)
    """)
    st.markdown("""
    - Activation: ReLU (Rectified Linear Unit)
    - Solver: Adam optimizer
    - Early stopping: Prevents overfitting
    """)
    
    # ============================================
    # SECTION 3: TECHNICAL INDICATORS
    # ============================================
    st.markdown("---")
    st.markdown("## 📊 3. Technical Indicators (70+ Features)")
    
    st.markdown("### 3.1 Trend Indicators")
    
    trend_indicators = pd.DataFrame({
        'Indicator': ['SMA', 'EMA', 'MACD', 'ADX'],
        'Periods': ['5, 10, 20, 50', '12, 26, 50', 'Signal, Histogram', '14'],
        'Formula': [
            'Simple Moving Average',
            'Exponential Moving Average',
            'Moving Average Convergence Divergence',
            'Average Directional Index'
        ],
        'Purpose': [
            'Identify trend direction',
            'Weighted recent prices',
            'Momentum & trend changes',
            'Trend strength measurement'
        ]
    })
    
    st.dataframe(trend_indicators, use_container_width=True, hide_index=True)
    
    st.markdown("""
    **Simple Moving Average (SMA):**
    """)
    st.latex(r"""
    SMA_n = \frac{1}{n} \sum_{i=0}^{n-1} P_{t-i}
    """)
    
    st.markdown("""
    **Exponential Moving Average (EMA):**
    """)
    st.latex(r"""
    EMA_t = \alpha \cdot P_t + (1-\alpha) \cdot EMA_{t-1}
    """)
    st.latex(r"""
    \text{where } \alpha = \frac{2}{n+1}
    """)
    
    st.markdown("""
    **MACD (Moving Average Convergence Divergence):**
    """)
    st.latex(r"""
    MACD = EMA_{12} - EMA_{26}
    """)
    st.latex(r"""
    Signal = EMA_9(MACD)
    """)
    st.latex(r"""
    Histogram = MACD - Signal
    """)
    
    st.markdown("### 3.2 Momentum Indicators")
    
    momentum_indicators = pd.DataFrame({
        'Indicator': ['RSI', 'Stochastic', 'ROC'],
        'Periods': ['6, 14, 24', '%K, %D', '5, 10'],
        'Range': ['0-100', '0-100', 'Unlimited'],
        'Interpretation': [
            'Overbought (>70), Oversold (<30)',
            'Overbought (>80), Oversold (<20)',
            'Rate of price change'
        ]
    })
    
    st.dataframe(momentum_indicators, use_container_width=True, hide_index=True)
    
    st.markdown("""
    **Relative Strength Index (RSI):**
    """)
    st.latex(r"""
    RSI = 100 - \frac{100}{1 + RS}
    """)
    st.latex(r"""
    \text{where } RS = \frac{\text{Average Gain}}{\text{Average Loss}}
    """)
    
    st.markdown("""
    **Stochastic Oscillator:**
    """)
    st.latex(r"""
    \%K = \frac{C - L_{14}}{H_{14} - L_{14}} \times 100
    """)
    st.latex(r"""
    \%D = SMA_3(\%K)
    """)
    st.markdown("""
    Where:
    - $$C$$ = Current close price
    - $$L_{14}$$ = Lowest low in 14 periods
    - $$H_{14}$$ = Highest high in 14 periods
    """)
    
    st.markdown("### 3.3 Volatility Indicators")
    
    volatility_indicators = pd.DataFrame({
        'Indicator': ['Bollinger Bands', 'ATR', 'Standard Deviation'],
        'Calculation': ['SMA ± 2σ', 'True Range Average', 'Price Dispersion'],
        'Purpose': [
            'Price boundaries & volatility',
            'Market volatility measurement',
            'Risk assessment'
        ]
    })
    
    st.dataframe(volatility_indicators, use_container_width=True, hide_index=True)
    
    st.markdown("""
    **Bollinger Bands:**
    """)
    st.latex(r"""
    \text{Upper Band} = SMA_{20} + 2\sigma
    """)
    st.latex(r"""
    \text{Middle Band} = SMA_{20}
    """)
    st.latex(r"""
    \text{Lower Band} = SMA_{20} - 2\sigma
    """)
    st.latex(r"""
    \text{BB Width} = \frac{\text{Upper} - \text{Lower}}{\text{Close}}
    """)
    st.latex(r"""
    \text{BB Position} = \frac{\text{Close} - \text{Lower}}{\text{Upper} - \text{Lower}}
    """)
    
    st.markdown("""
    **Average True Range (ATR):**
    """)
    st.latex(r"""
    TR = \max(H - L, |H - C_{prev}|, |L - C_{prev}|)
    """)
    st.latex(r"""
    ATR = \frac{1}{14} \sum_{i=1}^{14} TR_i
    """)
    
    st.markdown("### 3.4 Volume Indicators")
    
    st.markdown("""
    **On-Balance Volume (OBV):**
    """)
    st.latex(r"""
    OBV_t = OBV_{t-1} + \begin{cases}
    +V_t & \text{if } C_t > C_{t-1} \\
    -V_t & \text{if } C_t < C_{t-1} \\
    0 & \text{if } C_t = C_{t-1}
    \end{cases}
    """)
    
    st.markdown("""
    **Volume Ratio:**
    """)
    st.latex(r"""
    \text{Volume Ratio} = \frac{V_t}{SMA_{20}(V)}
    """)
    
    # ============================================
    # SECTION 4: PREDICTION METHODOLOGY
    # ============================================
    st.markdown("---")
    st.markdown("## 🔮 4. Prediction Methodology")
    
    st.markdown("""
    ### 4.1 Recursive Prediction Process
    
    Our system uses a **recursive prediction** approach, where each prediction step:
    1. Recalculates ALL 70+ technical indicators
    2. Uses the predicted price to generate synthetic OHLCV data
    3. Feeds this new data back into the model for the next prediction
    
    This creates a **chain of predictions** that maintain indicator continuity.
    """)
    
    st.markdown("""
    **Prediction Loop (Pseudocode):**
    """)
    
    st.code("""
for period in range(1, n_periods):
    # Step 1: Calculate indicators from historical + predicted data
    indicators = calculate_all_indicators(historical_data)
    
    # Step 2: Extract features from latest candle
    features = extract_features(indicators[-1])
    
    # Step 3: Scale features (if needed)
    if model_requires_scaling:
        features = scaler.transform(features)
    
    # Step 4: Predict next price
    predicted_price = model.predict(features)
    
    # Step 5: Apply constraints
    if abs(predicted_price / current_price - 1) > max_change:
        predicted_price = constrain(predicted_price, max_change)
    
    # Step 6: Generate synthetic OHLCV candle
    new_candle = create_ohlcv(predicted_price, volatility)
    
    # Step 7: Append to historical data
    historical_data = append(historical_data, new_candle)
    
    # Step 8: Store prediction
    predictions.append(predicted_price)
    """, language="python")
    
    st.markdown("### 4.2 Price Constraints by Timeframe")
    
    constraints_df = pd.DataFrame({
        'Timeframe': ['15M', '4H', '1D', '1W'],
        'Max Change Per Period': ['5%', '10%', '20%', '20%'],
        'Noise Factor': ['0.05%', '0.1%', '0.1%', '0.1%'],
        'Rationale': [
            'Scalping - tight range',
            'Intraday - moderate range',
            'Daily - wider range',
            'Weekly - wider range'
        ]
    })
    
    st.dataframe(constraints_df, use_container_width=True, hide_index=True)
    
    st.markdown("""
    **Constraint Formula:**
    """)
    st.latex(r"""
    P_{constrained} = \begin{cases}
    P_{current} \times (1 + \Delta_{max}) & \text{if } P_{pred} > P_{current} \times (1 + \Delta_{max}) \\
    P_{current} \times (1 - \Delta_{max}) & \text{if } P_{pred} < P_{current} \times (1 - \Delta_{max}) \\
    P_{pred} & \text{otherwise}
    \end{cases}
    """)
    
    st.markdown("""
    **Noise Addition:**
    """)
    st.latex(r"""
    P_{final} = P_{constrained} \times (1 + \mathcal{N}(0, \sigma_{noise}))
    """)
    st.markdown("""
    Where $$\mathcal{N}(0, \sigma_{noise})$$ is Gaussian noise with mean 0 and standard deviation $$\sigma_{noise}$$
    """)
    
    # ============================================
    # SECTION 5: TRADING SIGNAL GENERATION
    # ============================================
    st.markdown("---")
    st.markdown("## 🎯 5. Trading Signal Generation")
    
    st.markdown("### 5.1 Signal Logic by Timeframe")
    
    signal_logic_df = pd.DataFrame({
        'Timeframe': ['15M', '4H / 1D / 1W'],
        'Short-term Window': ['2 periods', '3 periods'],
        'Mid-term Window': ['2 periods (3-4)', '2 periods (4-5)'],
        'Long Threshold': ['+0.5% & +0.8%', '+2% & +3%'],
        'Short Threshold': ['-0.5% & -0.8%', '-2% & -3%'],
        'Neutral Range': ['Between thresholds', 'Between thresholds']
    })
    
    st.dataframe(signal_logic_df, use_container_width=True, hide_index=True)
    
    st.markdown("""
    **Signal Calculation Formula:**
    """)
    st.latex(r"""
    \text{Short-term Change} = \left(\frac{\text{Avg}(P_1, P_2, ...)}{\text{Current Price}} - 1\right) \times 100
    """)
    st.latex(r"""
    \text{Mid-term Change} = \left(\frac{\text{Avg}(P_{n+1}, P_{n+2}, ...)}{\text{Current Price}} - 1\right) \times 100
    """)
    
    st.markdown("""
    **Signal Decision Tree:**
    """)
    st.latex(r"""
    \text{Signal} = \begin{cases}
    \text{LONG} & \text{if } \Delta_{short} > T_{long}^{short} \text{ AND } \Delta_{mid} > T_{long}^{mid} \\
    \text{SHORT} & \text{if } \Delta_{short} < T_{short}^{short} \text{ AND } \Delta_{mid} < T_{short}^{mid} \\
    \text{NEUTRAL} & \text{otherwise}
    \end{cases}
    """)
    
    st.markdown("### 5.2 Confidence & Probability Calculation")
    
    st.markdown("""
    **Confidence Score:**
    """)
    st.latex(r"""
    \text{Confidence} = \min\left(95, \text{Base} + |\Delta_{short}| \times \text{Multiplier}\right)
    """)
    st.markdown("""
    Where:
    - Base = 50 (for 15M) or 60 (for others)
    - Multiplier = 10 (for 15M) or 5 (for others)
    """)
    
    st.markdown("""
    **Bull/Bear Probability:**
    """)
    st.latex(r"""
    P_{bull} = \min\left(95, 50 + |\Delta_{short}| \times \text{Factor}\right)
    """)
    st.latex(r"""
    P_{bear} = 100 - P_{bull}
    """)
    st.markdown("""
    Where Factor = 8 (for 15M) or 3 (for others)
    """)
    
    st.markdown("### 5.3 Entry, Stop Loss & Take Profit")
    
    st.markdown("""
    **For LONG Signals:**
    """)
    st.latex(r"""
    \text{Entry} = P_{current} \times \begin{cases}
    0.999 & \text{15M} \\
    0.995 & \text{Others}
    \end{cases}
    """)
    st.latex(r"""
    \text{Stop Loss} = \text{Entry} \times \begin{cases}
    0.995 & \text{15M (0.5\% loss)} \\
    0.970 & \text{Others (3\% loss)}
    \end{cases}
    """)
    st.latex(r"""
    \text{TP1} = \text{Entry} \times \begin{cases}
    1.005 & \text{15M (0.5\% profit)} \\
    1.020 & \text{Others (2\% profit)}
    \end{cases}
    """)
    st.latex(r"""
    \text{TP2} = \text{Entry} \times \begin{cases}
    1.010 & \text{15M (1\% profit)} \\
    1.050 & \text{Others (5\% profit)}
    \end{cases}
    """)
    st.latex(r"""
    \text{TP3} = \text{Entry} \times \begin{cases}
    1.015 & \text{15M (1.5\% profit)} \\
    1.100 & \text{Others (10\% profit)}
    \end{cases}
    """)
    
    st.markdown("""
    **For SHORT Signals:**
    """)
    st.latex(r"""
    \text{Entry} = P_{current} \times \begin{cases}
    1.001 & \text{15M} \\
    1.005 & \text{Others}
    \end{cases}
    """)
    st.latex(r"""
    \text{Stop Loss} = \text{Entry} \times \begin{cases}
    1.005 & \text{15M (0.5\% loss)} \\
    1.030 & \text{Others (3\% loss)}
    \end{cases}
    """)
    st.latex(r"""
    \text{TP1} = \text{Entry} \times \begin{cases}
    0.995 & \text{15M (0.5\% profit)} \\
    0.980 & \text{Others (2\% profit)}
    \end{cases}
    """)
    st.latex(r"""
    \text{TP2} = \text{Entry} \times \begin{cases}
    0.990 & \text{15M (1\% profit)} \\
    0.950 & \text{Others (5\% profit)}
    \end{cases}
    """)
    st.latex(r"""
    \text{TP3} = \text{Entry} \times \begin{cases}
    0.985 & \text{15M (1.5\% profit)} \\
    0.900 & \text{Others (10\% profit)}
    \end{cases}
    """)
    
    # ============================================
    # SECTION 6: MODEL EVALUATION METRICS
    # ============================================
    st.markdown("---")
    st.markdown("## 📈 6. Model Evaluation Metrics")
    
    st.markdown("""
    ### 6.1 Performance Metrics
    
    We use multiple metrics to evaluate model performance:
    """)
    
    metrics_df = pd.DataFrame({
        'Metric': ['R² Score', 'MAE', 'RMSE', 'MAPE', 'Direction Accuracy'],
        'Formula': [
            'Coefficient of Determination',
            'Mean Absolute Error',
            'Root Mean Squared Error',
            'Mean Absolute Percentage Error',
            'Directional Accuracy'
        ],
        'Range': ['(-∞, 1]', '[0, ∞)', '[0, ∞)', '[0, ∞)', '[0, 1]'],
        'Best Value': ['1.0', '0', '0', '0', '1.0'],
        'Interpretation': [
            'Variance explained by model',
            'Average prediction error ($)',
            'Penalizes large errors more',
            'Percentage error',
            'Correct trend prediction rate'
        ]
    })
    
    st.dataframe(metrics_df, use_container_width=True, hide_index=True)
    
    st.markdown("""
    **R² Score (Coefficient of Determination):**
    """)
    st.latex(r"""
    R^2 = 1 - \frac{\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}{\sum_{i=1}^{n}(y_i - \bar{y})^2}
    """)
    st.markdown("""
    Where:
    - $$y_i$$ = Actual price
    - $$\hat{y}_i$$ = Predicted price
    - $$\bar{y}$$ = Mean of actual prices
    """)
    
    st.markdown("""
    **Mean Absolute Error (MAE):**
    """)
    st.latex(r"""
    MAE = \frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i|
    """)
    
    st.markdown("""
    **Root Mean Squared Error (RMSE):**
    """)
    st.latex(r"""
    RMSE = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}
    """)
    
    st.markdown("""
    **Mean Absolute Percentage Error (MAPE):**
    """)
    st.latex(r"""
    MAPE = \frac{100\%}{n}\sum_{i=1}^{n}\left|\frac{y_i - \hat{y}_i}{y_i}\right|
    """)
    
    st.markdown("""
    **Direction Accuracy:**
    """)
    st.latex(r"""
    \text{Dir Acc} = \frac{1}{n-1}\sum_{i=2}^{n}\mathbb{1}\left[\text{sign}(y_i - y_{i-1}) = \text{sign}(\hat{y}_i - \hat{y}_{i-1})\right]
    """)
    st.markdown("""
    Where $$\mathbb{1}[\cdot]$$ is the indicator function (1 if true, 0 if false)
    """)
    
    # ============================================
    # SECTION 7: CURRENT MODEL PERFORMANCE
    # ============================================
    st.markdown("---")
    st.markdown("## 🏆 7. Current Model Performance")
    
    if predictor and hasattr(predictor, 'all_model_results'):
        st.markdown("### 7.1 Best Models by Timeframe")
        
        performance_data = []
        for timeframe in ['15m', '4h', '1d', '1w']:
            if timeframe in predictor.best_models:
                best_model = predictor.best_models[timeframe]
                if best_model in predictor.all_model_results[timeframe]:
                    result = predictor.all_model_results[timeframe][best_model]
                    performance_data.append({
                        'Timeframe': timeframe.upper(),
                        'Best Model': best_model,
                        'R² Score': f"{result['r2']:.4f}",
                        'MAE ($)': f"${result['mae']:.2f}",
                        'RMSE ($)': f"${result['rmse']:.2f}",
                        'MAPE (%)': f"{result['mape']:.2f}%",
                        'Direction Accuracy': f"{result['direction_accuracy']:.2%}"
                    })
        
        if performance_data:
            df_performance = pd.DataFrame(performance_data)
            st.dataframe(df_performance, use_container_width=True, hide_index=True)
            
            st.markdown("""
            **Interpretation Guide:**
            - **R² > 0.90**: Excellent predictive power
            - **R² 0.70-0.90**: Good predictive power
            - **R² 0.50-0.70**: Moderate predictive power
            - **Direction Accuracy > 60%**: Better than random
            - **Direction Accuracy > 70%**: Strong directional prediction
            """)
        
        st.markdown("### 7.2 Model Comparison Across Timeframes")
        
        # Create comparison chart
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('R² Score Comparison', 'Direction Accuracy Comparison'),
            specs=[[{'type': 'bar'}, {'type': 'bar'}]]
        )
        
        timeframes = []
        r2_scores = []
        dir_accs = []
        
        for tf in ['15m', '4h', '1d', '1w']:
            if tf in predictor.best_models:
                best_model = predictor.best_models[tf]
                result = predictor.all_model_results[tf][best_model]
                timeframes.append(tf.upper())
                r2_scores.append(result['r2'])
                dir_accs.append(result['direction_accuracy'] * 100)
        
        fig.add_trace(
            go.Bar(
                x=timeframes,
                y=r2_scores,
                name='R² Score',
                marker_color=['#f39c12', '#3498db', '#2ecc71', '#9b59b6'],
                text=[f"{r2:.3f}" for r2 in r2_scores],
                textposition='outside'
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Bar(
                x=timeframes,
                y=dir_accs,
                name='Direction Accuracy',
                marker_color=['#e67e22', '#3498db', '#27ae60', '#8e44ad'],
                text=[f"{acc:.1f}%" for acc in dir_accs],
                textposition='outside'
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            height=400,
            showlegend=False,
            template='plotly_dark'
        )
        
        fig.update_yaxes(title_text="R² Score", row=1, col=1, range=[0, 1])
        fig.update_yaxes(title_text="Accuracy (%)", row=1, col=2, range=[0, 100])
        
        st.plotly_chart(fig, use_container_width=True)
    
    # ============================================
    # SECTION 8: RISK DISCLOSURE
    # ============================================
    st.markdown("---")
    st.markdown("## ⚠️ 8. Risk Disclosure & Limitations")
    
    st.warning("""
    **Important Disclaimers:**
    
    1. **Past Performance ≠ Future Results**: Historical accuracy does not guarantee future performance.
    
    2. **Market Volatility**: Cryptocurrency markets are highly volatile and unpredictable.
    
    3. **Technical Analysis Limitations**: 
       - Does not account for fundamental factors
       - Does not predict black swan events
       - Assumes historical patterns repeat
    
    4. **Model Limitations**:
       - Trained on historical data only
       - Cannot predict unexpected news/events
       - Maximum constraint may miss breakouts
       - Lag in indicator calculation
    
    5. **Not Financial Advice**: This system is for informational purposes only. 
       Always do your own research and consult with financial advisors.
    
    6. **Risk Management**: Never invest more than you can afford to lose. 
       Always use stop losses and proper position sizing.
    """)
    
    # ============================================
    # SECTION 9: REFERENCES
    # ============================================
    st.markdown("---")
    st.markdown("## 📚 9. References & Further Reading")
    
    st.markdown("""
    ### Academic Papers:
    1. Breiman, L. (2001). "Random Forests". Machine Learning, 45(1), 5-32.
    2. Friedman, J. H. (2001). "Greedy Function Approximation: A Gradient Boosting Machine".
    3. Wilder, J. W. (1978). "New Concepts in Technical Trading Systems".
    4. Bollinger, J. (1992). "Using Bollinger Bands".
    
    ### Libraries Used:
    - **scikit-learn**: Machine learning algorithms
    - **ta (Technical Analysis)**: Technical indicators calculation
    - **pandas**: Data manipulation
    - **numpy**: Numerical computations
    - **plotly**: Interactive visualizations
    
    ### Data Source:
    - **Binance API**: Real-time and historical cryptocurrency data
    - **WebSocket**: Live price streaming
    """)
    
    st.markdown("---")
    st.info("""
    **System Version**: 2.0.0  
    **Last Updated**: 2025-01-20  
    **Developed by**: Crypto Prediction Team  
    **Contact**: support@cryptoprediction.ai
    """)

