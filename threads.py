"""
threads.py
Background threads for price updates and analysis
"""

from PyQt5.QtCore import QThread, pyqtSignal
from eth import AdvancedETHPredictor
import traceback
import time


class MultiPriceUpdateThread(QThread):
    """Thread for updating multiple symbols at once"""
    prices_updated = pyqtSignal(dict)  # {symbol: ticker_data}
    
    def __init__(self, symbols):
        super().__init__()
        self.symbols = symbols
        self.running = True
        self.predictor = AdvancedETHPredictor()
    
    def run(self):
        while self.running:
            try:
                all_prices = {}
                for symbol in self.symbols:
                    ticker_data = self.predictor.get_24h_ticker(symbol)
                    if ticker_data:
                        all_prices[symbol] = ticker_data
                    
                    # Small delay between requests to avoid rate limiting
                    time.sleep(0.1)  # Giảm từ 0.5s xuống 0.1s
                
                if all_prices:
                    self.prices_updated.emit(all_prices)
            except Exception as e:
                print(f"Multi-price update error: {e}")
            
            self.msleep(2000)  # Update every 2 seconds
    
    def stop(self):
        self.running = False


class AnalysisThread(QThread):
    """Background thread for running analysis"""
    finished = pyqtSignal(object, object)
    progress = pyqtSignal(str)
    error = pyqtSignal(str)
    log_message = pyqtSignal(str)
    
    def __init__(self, symbol, timezone):
        super().__init__()
        self.symbol = symbol
        self.timezone = timezone
    
    def log_callback(self, message):
        """Callback for logging from predictor"""
        self.log_message.emit(message)
    
    def run(self):
        try:
            self.progress.emit("Initializing predictor...")
            predictor = AdvancedETHPredictor(timezone=self.timezone, log_callback=self.log_callback)
            
            self.progress.emit("Running analysis...")
            all_data, all_predictions = predictor.run_analysis(self.symbol)
            
            if all_data and all_predictions:
                self.finished.emit(predictor, all_predictions)
            else:
                self.error.emit("Analysis failed - no data returned")
        except Exception as e:
            self.error.emit(f"Error: {str(e)}\n{traceback.format_exc()}")


class CandlestickDataThread(QThread):
    """Thread for fetching candlestick data"""
    data_ready = pyqtSignal(object)  # pandas DataFrame
    error = pyqtSignal(str)
    
    def __init__(self, symbol, interval='1h', limit=100):
        super().__init__()
        self.symbol = symbol
        self.interval = interval
        self.limit = limit
    
    def run(self):
        try:
            # Import binance client directly
            from binance.client import Client
            import pandas as pd
            
            # Create new client instance
            client = Client()
            
            # Fetch klines data from Binance
            klines = client.get_klines(
                symbol=self.symbol,
                interval=self.interval,
                limit=self.limit
            )
            
            if klines:
                df = pd.DataFrame(klines, columns=[
                    'timestamp', 'open', 'high', 'low', 'close', 'volume',
                    'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                    'taker_buy_quote', 'ignore'
                ])
                
                # Convert to proper types
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    df[col] = df[col].astype(float)
                
                self.data_ready.emit(df)
            else:
                self.error.emit("No data received from Binance")
                
        except Exception as e:
            self.error.emit(f"Error fetching candlestick data: {str(e)}\n{traceback.format_exc()}")


class CandlestickUpdateThread(QThread):
    """Thread for real-time candlestick updates"""
    data_ready = pyqtSignal(object)
    
    def __init__(self, symbol, interval='1h'):
        super().__init__()
        self.symbol = symbol
        self.interval = interval
        self.running = True
    
    def run(self):
        while self.running:
            try:
                from binance.client import Client
                import pandas as pd
                
                client = Client()
                
                # Fetch latest klines
                klines = client.get_klines(
                    symbol=self.symbol,
                    interval=self.interval,
                    limit=200
                )
                
                if klines:
                    df = pd.DataFrame(klines, columns=[
                        'timestamp', 'open', 'high', 'low', 'close', 'volume',
                        'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                        'taker_buy_quote', 'ignore'
                    ])
                    
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    for col in ['open', 'high', 'low', 'close', 'volume']:
                        df[col] = df[col].astype(float)
                    
                    self.data_ready.emit(df)
                
            except Exception as e:
                print(f"Real-time update error: {e}")
            
            # Update interval based on timeframe
            if self.interval == '15m':
                self.msleep(5000)  # 5 seconds
            elif self.interval == '1h':
                self.msleep(10000)  # 10 seconds
            else:
                self.msleep(15000)  # 15 seconds
    
    def stop(self):
        self.running = False
