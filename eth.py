import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
import ta
from datetime import datetime, timedelta
import pytz
import warnings
import yfinance as yf  # THÊM IMPORT
warnings.filterwarnings('ignore')

class AdvancedETHPredictor:
    def __init__(self, timezone='Asia/Ho_Chi_Minh', log_callback=None):
        self.base_url = "https://api.binance.com/api/v3"
        self.models = {}
        self.scalers = {}
        self.feature_names = {}
        self.all_model_results = {}
        self.best_models = {}
        self.reference_price = None
        self.reference_time = None
        self.reference_time_local = None
        self.timezone = timezone
        self.log_callback = log_callback
        
    def log(self, message):
        """Log message to console and GUI"""
        print(message)
        if self.log_callback:
            self.log_callback(message)
    
    def is_forex_symbol(self, symbol):
        """Check if symbol is forex (not crypto)"""
        forex_symbols = ['XAUUSD', 'EURUSD', 'GBPUSD', 'USDJPY']
        return symbol in forex_symbols
    
    def get_yfinance_symbol(self, symbol):
        """Convert symbol to yfinance format"""
        symbol_map = {
            'XAUUSD': 'GC=F',  # Gold Futures
            'EURUSD': 'EURUSD=X',
            'GBPUSD': 'GBPUSD=X',
            'USDJPY': 'USDJPY=X'
        }
        return symbol_map.get(symbol, symbol)
    
    def get_current_price(self, symbol="ETHUSDT"):
        """Get current real-time price"""
        try:
            if self.is_forex_symbol(symbol):
                # Use yfinance for forex
                yf_symbol = self.get_yfinance_symbol(symbol)
                ticker = yf.Ticker(yf_symbol)
                data = ticker.history(period='1d', interval='1m')
                if not data.empty:
                    return float(data['Close'].iloc[-1])
                return None
            else:
                # Use Binance for crypto
                url = f"{self.base_url}/ticker/price"
                params = {'symbol': symbol}
                response = requests.get(url, params=params, timeout=5)
                response.raise_for_status()
                data = response.json()
                return float(data['price'])
        except Exception as e:
            self.log(f"Error getting current price for {symbol}: {e}")
            return None

    def get_24h_ticker(self, symbol="ETHUSDT"):
        """Get 24h ticker data"""
        try:
            if self.is_forex_symbol(symbol):
                # Use yfinance for forex
                yf_symbol = self.get_yfinance_symbol(symbol)
                ticker = yf.Ticker(yf_symbol)
                
                # Get 2 days of data to calculate 24h change
                data = ticker.history(period='2d', interval='1h')
                
                if len(data) < 2:
                    return None
                
                current_price = float(data['Close'].iloc[-1])
                prev_price = float(data['Close'].iloc[0])
                high_24h = float(data['High'].tail(24).max())
                low_24h = float(data['Low'].tail(24).min())
                volume_24h = float(data['Volume'].tail(24).sum())
                
                change = current_price - prev_price
                change_percent = (change / prev_price * 100) if prev_price != 0 else 0
                
                return {
                    'price': current_price,
                    'change': change,
                    'change_percent': change_percent,
                    'high': high_24h,
                    'low': low_24h,
                    'volume': volume_24h
                }
            else:
                # Use Binance for crypto
                url = f"{self.base_url}/ticker/24hr"
                params = {'symbol': symbol}
                response = requests.get(url, params=params, timeout=5)
                response.raise_for_status()
                data = response.json()
                return {
                    'price': float(data['lastPrice']),
                    'change': float(data['priceChange']),
                    'change_percent': float(data['priceChangePercent']),
                    'high': float(data['highPrice']),
                    'low': float(data['lowPrice']),
                    'volume': float(data['volume'])
                }
        except Exception as e:
            self.log(f"Error getting 24h ticker for {symbol}: {e}")
            return None
        
    def fetch_kline_data(self, symbol="ETHUSDT", interval="1d", limit=500):
        """Fetch historical price data"""
        self.log(f"Fetching {symbol} {interval} data...")
        
        try:
            if self.is_forex_symbol(symbol):
                # Use yfinance for forex
                yf_symbol = self.get_yfinance_symbol(symbol)
                
                # Map interval to yfinance period
                period_map = {
                    '4h': '60d',
                    '1d': '2y',
                    '1w': '5y'
                }
                interval_map = {
                    '4h': '1h',
                    '1d': '1d',
                    '1w': '1wk'
                }
                
                period = period_map.get(interval, '2y')
                yf_interval = interval_map.get(interval, '1d')
                
                ticker = yf.Ticker(yf_symbol)
                data = ticker.history(period=period, interval=yf_interval)
                
                if data.empty:
                    self.log(f"❌ No data for {symbol}")
                    return None
                
                # Convert to Binance-like format
                df = pd.DataFrame({
                    'open_time': [0] * len(data),
                    'open': data['Open'].values,
                    'high': data['High'].values,
                    'low': data['Low'].values,
                    'close': data['Close'].values,
                    'volume': data['Volume'].values,
                    'close_time': [0] * len(data),
                    'quote_asset_volume': [0] * len(data),
                    'number_of_trades': [0] * len(data),
                    'taker_buy_base_asset_volume': [0] * len(data),
                    'taker_buy_quote_asset_volume': [0] * len(data),
                    'ignore': [0] * len(data),
                    'datetime': data.index
                })
                
                # Resample to get exact number of candles
                if interval == '4h':
                    df = df.tail(300)
                elif interval == '1d':
                    df = df.tail(500)
                else:
                    df = df.tail(150)
                
                df = df.reset_index(drop=True)
                
                self.log(f"✅ Loaded {len(df)} candles for {symbol}")
                return df
                
            else:
                # Use Binance for crypto
                url = f"{self.base_url}/klines"
                params = {'symbol': symbol, 'interval': interval, 'limit': limit}
                
                response = requests.get(url, params=params)
                response.raise_for_status()
                data = response.json()
                
                df = pd.DataFrame(data, columns=[
                    'open_time', 'open', 'high', 'low', 'close', 'volume',
                    'close_time', 'quote_asset_volume', 'number_of_trades',
                    'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
                ])
                
                numeric_columns = ['open', 'high', 'low', 'close', 'volume']
                for col in numeric_columns:
                    df[col] = pd.to_numeric(df[col])
                
                df['datetime'] = pd.to_datetime(df['open_time'], unit='ms')
                df = df.sort_values('datetime').reset_index(drop=True)
                
                self.log(f"✅ Loaded {len(df)} candles")
                return df
                
        except Exception as e:
            self.log(f"❌ Error fetching data for {symbol}: {e}")
            return None
    
    def fetch_all_data(self, symbol="ETHUSDT"):
        """Fetch all timeframes and store reference price"""
        self.log(f"\n🔄 Fetching all data for {symbol}...")
        
        timeframes = {'4h': 300, '1d': 500, '1w': 150}
        all_data = {}
        
        for timeframe, limit in timeframes.items():
            df = self.fetch_kline_data(symbol, timeframe, limit)
            if df is not None:
                all_data[timeframe] = df
        
        if '4h' in all_data:
            self.reference_price = all_data['4h']['close'].iloc[-1]
            self.reference_time = all_data['4h']['datetime'].iloc[-1]
            
            utc_time = self.reference_time.tz_localize('UTC')
            local_tz = pytz.timezone(self.timezone)
            self.reference_time_local = utc_time.astimezone(local_tz)
            
            self.log(f"\n💰 Reference Price: ${self.reference_price:.2f}")
            self.log(f"⏰ Reference Time (UTC): {self.reference_time}")
            self.log(f"🌏 Reference Time ({self.timezone}): {self.reference_time_local}")
        
        return all_data
    
    def calculate_indicators(self, df, timeframe):
        """Calculate comprehensive technical indicators"""
        df = df.copy()
        
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in required_cols:
            if col not in df.columns:
                return df
        
        close = df['close']
        high = df['high']
        low = df['low']
        volume = df['volume']
        
        # Trend Indicators
        df['sma_5'] = ta.trend.sma_indicator(close, window=5)
        df['sma_10'] = ta.trend.sma_indicator(close, window=10)
        df['sma_20'] = ta.trend.sma_indicator(close, window=20)
        df['sma_50'] = ta.trend.sma_indicator(close, window=50)
        df['ema_12'] = ta.trend.ema_indicator(close, window=12)
        df['ema_26'] = ta.trend.ema_indicator(close, window=26)
        df['ema_50'] = ta.trend.ema_indicator(close, window=50)
        
        # MACD
        df['macd'] = ta.trend.macd_diff(close)
        df['macd_signal'] = ta.trend.macd_signal(close)
        df['macd_hist'] = ta.trend.macd(close) - ta.trend.macd_signal(close)
        
        # RSI
        df['rsi'] = ta.momentum.rsi(close, window=14)
        df['rsi_6'] = ta.momentum.rsi(close, window=6)
        df['rsi_24'] = ta.momentum.rsi(close, window=24)
        
        # Stochastic
        df['stoch'] = ta.momentum.stoch(high, low, close)
        df['stoch_signal'] = ta.momentum.stoch_signal(high, low, close)
        
        # Bollinger Bands
        df['bb_high'] = ta.volatility.bollinger_hband(close)
        df['bb_mid'] = ta.volatility.bollinger_mavg(close)
        df['bb_low'] = ta.volatility.bollinger_lband(close)
        df['bb_width'] = (df['bb_high'] - df['bb_low']) / close
        df['bb_position'] = (close - df['bb_low']) / (df['bb_high'] - df['bb_low'])
        
        # ATR (Average True Range)
        df['atr'] = ta.volatility.average_true_range(high, low, close)
        
        # ADX (Average Directional Index)
        df['adx'] = ta.trend.adx(high, low, close)
        
        # Volume Indicators
        df['volume_sma'] = volume.rolling(window=20).mean()
        df['volume_ratio'] = volume / df['volume_sma']
        df['obv'] = ta.volume.on_balance_volume(close, volume)
        
        # Price Action Features
        df['price_change'] = close.pct_change()
        df['price_change_5'] = close.pct_change(5)
        df['price_change_10'] = close.pct_change(10)
        df['volatility'] = df['price_change'].rolling(window=10).std()
        df['volatility_20'] = df['price_change'].rolling(window=20).std()
        
        # High-Low Range
        df['hl_ratio'] = (high - low) / close
        df['oc_ratio'] = (close - df['open']) / df['open']
        
        # Lag Features
        for lag in [1, 2, 3, 5, 7]:
            df[f'close_lag_{lag}'] = close.shift(lag)
            df[f'volume_lag_{lag}'] = volume.shift(lag)
        
        # Rolling Statistics
        df['close_rolling_mean_5'] = close.rolling(window=5).mean()
        df['close_rolling_std_5'] = close.rolling(window=5).std()
        df['close_rolling_mean_20'] = close.rolling(window=20).mean()
        df['close_rolling_std_20'] = close.rolling(window=20).std()
        
        # Price momentum
        df['momentum_5'] = close - close.shift(5)
        df['momentum_10'] = close - close.shift(10)
        df['momentum_20'] = close - close.shift(20)
        
        # Rate of Change
        df['roc_5'] = ta.momentum.roc(close, window=5)
        df['roc_10'] = ta.momentum.roc(close, window=10)
        
        return df
    
    def prepare_features(self, df, timeframe):
        """Prepare features for training"""
        exclude_columns = [
            'open_time', 'close_time', 'ignore', 'datetime', 
            'open', 'high', 'low', 'close', 'volume',
            'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume'
        ]
        
        feature_columns = [col for col in df.columns if col not in exclude_columns]
        df['target'] = df['close'].shift(-1)
        
        df_clean = df.dropna()
        if len(df_clean) == 0:
            return None, None, None
        
        X = df_clean[feature_columns]
        y = df_clean['target']
        
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.median())
        
        self.feature_names[timeframe] = feature_columns
        return X, y, df_clean
    
    def train_models(self, X, y, timeframe):
        """Train multiple models with different algorithms"""
        self.log(f"\n{'='*60}")
        self.log(f"Training models for {timeframe.upper()}")
        self.log(f"{'='*60}")
        
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        # Multiple scalers
        scalers = {
            'RobustScaler': RobustScaler(),
            'StandardScaler': StandardScaler(),
            'MinMaxScaler': MinMaxScaler()
        }
        
        # Define models with optimized parameters
        models_config = {
            'Random Forest': {
                'model': RandomForestRegressor(
                    n_estimators=200,
                    max_depth=15,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=-1
                ),
                'use_scaler': False,
                'scaler_type': None
            },
            'Gradient Boosting': {
                'model': GradientBoostingRegressor(
                    n_estimators=200,
                    max_depth=8,
                    learning_rate=0.05,
                    min_samples_split=5,
                    random_state=42
                ),
                'use_scaler': False,
                'scaler_type': None
            },
            'Extra Trees': {
                'model': ExtraTreesRegressor(
                    n_estimators=200,
                    max_depth=15,
                    min_samples_split=5,
                    random_state=42,
                    n_jobs=-1
                ),
                'use_scaler': False,
                'scaler_type': None
            },
            'AdaBoost': {
                'model': AdaBoostRegressor(
                    n_estimators=100,
                    learning_rate=0.1,
                    random_state=42
                ),
                'use_scaler': False,
                'scaler_type': None
            },
            'Linear Regression': {
                'model': LinearRegression(),
                'use_scaler': True,
                'scaler_type': 'StandardScaler'
            },
            'Ridge': {
                'model': Ridge(alpha=1.0),
                'use_scaler': True,
                'scaler_type': 'StandardScaler'
            },
            'Lasso': {
                'model': Lasso(alpha=1.0, max_iter=5000),
                'use_scaler': True,
                'scaler_type': 'StandardScaler'
            },
            'ElasticNet': {
                'model': ElasticNet(alpha=1.0, l1_ratio=0.5, max_iter=5000),
                'use_scaler': True,
                'scaler_type': 'StandardScaler'
            },
            'SVR': {
                'model': SVR(kernel='rbf', C=100, gamma='scale', epsilon=0.1),
                'use_scaler': True,
                'scaler_type': 'StandardScaler'
            },
            'Neural Network': {
                'model': MLPRegressor(
                    hidden_layer_sizes=(100, 50),
                    activation='relu',
                    solver='adam',
                    max_iter=1000,
                    random_state=42,
                    early_stopping=True
                ),
                'use_scaler': True,
                'scaler_type': 'StandardScaler'
            }
        }
        
        results = {}
        
        for name, config in models_config.items():
            try:
                self.log(f"\n🔧 Training {name}...")
                
                model = config['model']
                
                if config['use_scaler']:
                    scaler = scalers[config['scaler_type']]
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_test_scaled = scaler.transform(X_test)
                    
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)
                else:
                    scaler = None
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                
                # Calculate metrics
                r2 = r2_score(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                mape = mean_absolute_percentage_error(y_test, y_pred) * 100
                
                # Direction accuracy
                if len(y_test) > 1:
                    actual_dir = (y_test.values[1:] > y_test.values[:-1]).astype(int)
                    pred_dir = (y_pred[1:] > y_pred[:-1]).astype(int)
                    dir_acc = (actual_dir == pred_dir).mean()
                else:
                    dir_acc = 0
                
                results[name] = {
                    'model': model,
                    'scaler': scaler,
                    'r2': r2,
                    'mae': mae,
                    'rmse': rmse,
                    'mape': mape,
                    'direction_accuracy': dir_acc,
                    'y_pred': y_pred,
                    'y_test': y_test
                }
                
                self.log(f"   ✅ R²={r2:.4f} | MAE=${mae:.2f} | RMSE=${rmse:.2f} | MAPE={mape:.2f}% | Dir={dir_acc:.2%}")
                
            except Exception as e:
                self.log(f"   ❌ Failed: {e}")
                continue
        
        # Store all results for this timeframe
        self.all_model_results[timeframe] = results
        
        if results:
            # Select best model based on R² score
            best_model_name = max(results.keys(), key=lambda k: results[k]['r2'])
            self.models[timeframe] = results[best_model_name]['model']
            self.scalers[timeframe] = results[best_model_name]['scaler']
            self.best_models[timeframe] = best_model_name
            
            self.log(f"\n🏆 Best Model: {best_model_name} (R²={results[best_model_name]['r2']:.4f})")
        
        return results
    
    def predict_future(self, df_clean, timeframe, model_name=None, periods=7):
        """
        Predict future prices with proper indicator recalculation
        If model_name is None, use the best model
        """
        if timeframe not in self.all_model_results:
            return None
        
        # Select model
        if model_name is None:
            model_name = self.best_models.get(timeframe)
            if model_name is None:
                return None
        
        if model_name not in self.all_model_results[timeframe]:
            return None
        
        model_data = self.all_model_results[timeframe][model_name]
        model = model_data['model']
        scaler = model_data['scaler']
        feature_names = self.feature_names[timeframe]
        
        predictions = []
        current_data = df_clean.copy()
        
        for i in range(periods):
            try:
                # Recalculate indicators
                temp_data = self.calculate_indicators(current_data, timeframe)
                
                # Prepare features
                exclude_columns = [
                    'open_time', 'close_time', 'ignore', 'datetime', 
                    'open', 'high', 'low', 'close', 'volume',
                    'quote_asset_volume', 'number_of_trades',
                    'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume',
                    'target'
                ]
                
                available_features = [col for col in temp_data.columns 
                                     if col not in exclude_columns and col in feature_names]
                
                temp_data_clean = temp_data[available_features].replace([np.inf, -np.inf], np.nan)
                temp_data_clean = temp_data_clean.fillna(temp_data_clean.median())
                
                if len(temp_data_clean) == 0:
                    break
                
                features = temp_data_clean.iloc[-1:][feature_names].values
                
                # Scale if needed
                if scaler:
                    features = scaler.transform(features)
                
                # Predict
                pred_price = model.predict(features)[0]
                
                # Apply realistic constraints
                current_price = current_data['close'].iloc[-1]
                max_change = 0.20  # 20% max change
                
                if abs(pred_price / current_price - 1) > max_change:
                    if pred_price > current_price:
                        pred_price = current_price * (1 + max_change)
                    else:
                        pred_price = current_price * (1 - max_change)
                
                # Add minimal noise for variance
                noise_factor = np.random.normal(0, 0.001)
                pred_price = pred_price * (1 + noise_factor)
                
                predictions.append(pred_price)
                
                # Create new OHLCV data
                volatility = current_data['close'].pct_change().std()
                if pd.isna(volatility) or volatility == 0:
                    volatility = 0.01
                
                high_price = pred_price * (1 + volatility * 0.5)
                low_price = pred_price * (1 - volatility * 0.5)
                open_price = current_price + (pred_price - current_price) * 0.3
                
                avg_volume = current_data['volume'].tail(20).mean()
                new_volume = avg_volume * np.random.uniform(0.8, 1.2)
                
                # Calculate next datetime
                last_datetime = current_data['datetime'].iloc[-1]
                if timeframe == '4h':
                    next_datetime = last_datetime + pd.Timedelta(hours=4)
                elif timeframe == '1d':
                    next_datetime = last_datetime + pd.Timedelta(days=1)
                else:
                    next_datetime = last_datetime + pd.Timedelta(weeks=1)
                
                # Create new row
                new_row = pd.DataFrame({
                    'open_time': [0],
                    'open': [open_price],
                    'high': [high_price],
                    'low': [low_price],
                    'close': [pred_price],
                    'volume': [new_volume],
                    'close_time': [0],
                    'quote_asset_volume': [0],
                    'number_of_trades': [0],
                    'taker_buy_base_asset_volume': [0],
                    'taker_buy_quote_asset_volume': [0],
                    'ignore': [0],
                    'datetime': [next_datetime]
                })
                
                current_data = pd.concat([current_data, new_row], ignore_index=True)
                
            except Exception as e:
                self.log(f"Error at period {i+1}: {e}")
                break
        
        return predictions
    
    def get_all_predictions(self, all_data, periods=7):
        """Get predictions from all models for all timeframes"""
        all_predictions = {}
        
        for timeframe in ['4h', '1d', '1w']:
            if timeframe not in self.all_model_results or timeframe not in all_data:
                continue
            
            all_predictions[timeframe] = {}
            
            for model_name in self.all_model_results[timeframe].keys():
                predictions = self.predict_future(all_data[timeframe], timeframe, model_name, periods)
                if predictions:
                    all_predictions[timeframe][model_name] = predictions
        
        return all_predictions
    
    def run_analysis(self, symbol="ETHUSDT"):
        """Run complete analysis"""
        self.log("🚀 Starting Advanced ETH Prediction Analysis")
        self.log("="*60)
        
        # Fetch data
        all_raw_data = self.fetch_all_data(symbol)
        
        if not all_raw_data:
            self.log("❌ No data fetched!")
            return None, None
        
        all_processed_data = {}
        
        for timeframe, df in all_raw_data.items():
            self.log(f"\n📊 Processing {timeframe.upper()}...")
            
            # Calculate indicators
            df_with_indicators = self.calculate_indicators(df, timeframe)
            
            # Prepare features
            result = self.prepare_features(df_with_indicators, timeframe)
            if result[0] is None:
                continue
            X, y, df_clean = result
            
            # Train models
            self.train_models(X, y, timeframe)
            all_processed_data[timeframe] = df_clean
        
        # Get all predictions
        all_predictions = self.get_all_predictions(all_processed_data, periods=10)
        
        self.log(f"\n{'='*60}")
        self.log("✅ ANALYSIS COMPLETE")
        self.log(f"{'='*60}")
        
        return all_processed_data, all_predictions
