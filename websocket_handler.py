import json
import threading
import time
from collections import deque
from websocket import create_connection, WebSocketConnectionClosedException
import pandas as pd

class BinanceWebSocket:
    """WebSocket handler for real-time price updates"""
    def __init__(self):
        self.prices = {}
        self.ws = None
        self.running = False
        self.thread = None
        self.lock = threading.Lock()
        
    def start(self, symbols):
        """Start WebSocket connection for multiple symbols"""
        if self.running:
            return
            
        self.running = True
        streams = [f"{symbol.lower()}@ticker" for symbol in symbols]
        url = f"wss://stream.binance.com:9443/stream?streams={'/'.join(streams)}"
        
        self.thread = threading.Thread(target=self._run, args=(url,), daemon=True)
        self.thread.start()
        
    def _run(self, url):
        """Run WebSocket connection in background"""
        retry_count = 0
        max_retries = 10
        
        while self.running and retry_count < max_retries:
            try:
                self.ws = create_connection(url, timeout=10)
                retry_count = 0
                print(f"✅ Price WebSocket connected")
                
                while self.running:
                    try:
                        result = self.ws.recv()
                        data = json.loads(result)
                        
                        if 'data' in data:
                            ticker = data['data']
                            symbol = ticker['s']
                            
                            with self.lock:
                                self.prices[symbol] = {
                                    'price': float(ticker['c']),
                                    'change_percent': float(ticker['P']),
                                    'high': float(ticker['h']),
                                    'low': float(ticker['l']),
                                    'volume': float(ticker['v']),
                                    'timestamp': time.time()
                                }
                                
                    except WebSocketConnectionClosedException:
                        print("⚠️ Price WebSocket connection closed")
                        break
                    except json.JSONDecodeError:
                        continue
                    except Exception as e:
                        print(f"❌ Error processing price message: {e}")
                        continue
                        
            except Exception as e:
                retry_count += 1
                print(f"❌ Price WebSocket error (attempt {retry_count}/{max_retries}): {e}")
                time.sleep(min(2 ** retry_count, 30))
            finally:
                if self.ws:
                    try:
                        self.ws.close()
                    except:
                        pass
        
        print("❌ Price WebSocket stopped")
        
    def get_price(self, symbol):
        """Get latest price for symbol (thread-safe)"""
        with self.lock:
            return self.prices.get(symbol)
    
    def get_all_prices(self):
        """Get all prices (thread-safe)"""
        with self.lock:
            return self.prices.copy()
    
    def stop(self):
        """Stop WebSocket connection"""
        self.running = False
        if self.ws:
            try:
                self.ws.close()
            except:
                pass


class ChartWebSocket:
    """WebSocket handler for real-time candlestick chart updates"""
    def __init__(self, symbol, interval, max_candles=200):
        self.symbol = symbol.lower()
        self.interval = interval
        self.max_candles = max_candles
        self.candles = deque(maxlen=max_candles)
        self.current_candle = None
        self.ws = None
        self.thread = None
        self.running = False
        self.callbacks = []
        self.lock = threading.Lock()
        
    def add_callback(self, callback):
        """Add callback function to be called on updates"""
        self.callbacks.append(callback)
        
    def _on_message(self, message):
        """Handle incoming WebSocket messages"""
        try:
            data = json.loads(message)
            
            if 'k' in data:
                kline = data['k']
                
                candle = {
                    'timestamp': kline['t'],
                    'open': float(kline['o']),
                    'high': float(kline['h']),
                    'low': float(kline['l']),
                    'close': float(kline['c']),
                    'volume': float(kline['v']),
                    'is_closed': kline['x']
                }
                
                with self.lock:
                    if kline['x']:
                        if self.current_candle:
                            self.candles.append(self.current_candle)
                        self.current_candle = candle
                    else:
                        self.current_candle = candle
                
                for callback in self.callbacks:
                    try:
                        callback(self.get_dataframe())
                    except Exception as e:
                        print(f"Error in callback: {e}")
                        
        except json.JSONDecodeError:
            pass
        except Exception as e:
            print(f"Error processing chart message: {e}")
    
    def _run(self):
        """Run WebSocket connection in background"""
        url = f"wss://stream.binance.com:9443/ws/{self.symbol}@kline_{self.interval}"
        retry_count = 0
        max_retries = 10
        
        while self.running and retry_count < max_retries:
            try:
                self.ws = create_connection(url, timeout=10)
                retry_count = 0
                print(f"✅ Chart WebSocket connected: {self.symbol} {self.interval}")
                
                while self.running:
                    try:
                        result = self.ws.recv()
                        self._on_message(result)
                        
                    except WebSocketConnectionClosedException:
                        print(f"⚠️ Chart WebSocket closed: {self.symbol}")
                        break
                    except Exception as e:
                        print(f"❌ Error receiving chart message: {e}")
                        continue
                        
            except Exception as e:
                retry_count += 1
                print(f"❌ Chart WebSocket error (attempt {retry_count}/{max_retries}): {e}")
                time.sleep(min(2 ** retry_count, 30))
            finally:
                if self.ws:
                    try:
                        self.ws.close()
                    except:
                        pass
        
        print(f"❌ Chart WebSocket stopped: {self.symbol} {self.interval}")
    
    def get_dataframe(self):
        """Convert candles to DataFrame format (thread-safe)"""
        with self.lock:
            all_candles = list(self.candles)
            if self.current_candle:
                all_candles.append(self.current_candle)
        
        if not all_candles:
            return None
            
        df = pd.DataFrame(all_candles)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df
    
    def get_latest_price(self):
        """Get latest close price (thread-safe)"""
        with self.lock:
            if self.current_candle:
                return self.current_candle['close']
            elif self.candles:
                return self.candles[-1]['close']
        return None
    
    def start(self, initial_data=None):
        """Start WebSocket connection with optional initial data"""
        if self.running:
            return
        
        if initial_data is not None:
            with self.lock:
                for _, row in initial_data.iterrows():
                    self.candles.append({
                        'timestamp': int(row['timestamp'].timestamp() * 1000),
                        'open': row['open'],
                        'high': row['high'],
                        'low': row['low'],
                        'close': row['close'],
                        'volume': row['volume'],
                        'is_closed': True
                    })
        
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
    
    def stop(self):
        """Stop WebSocket connection"""
        self.running = False
        if self.ws:
            try:
                self.ws.close()
            except:
                pass
        if self.thread:
            self.thread.join(timeout=1)


class WebSocketManager:
    """Manager for multiple WebSocket connections"""
    def __init__(self):
        self.price_ws = None
        self.chart_ws_dict = {}
        self.lock = threading.Lock()
    
    def start_price_stream(self, symbols):
        """Start price WebSocket for multiple symbols"""
        if self.price_ws is None:
            self.price_ws = BinanceWebSocket()
            self.price_ws.start(symbols)
        return self.price_ws
    
    def start_chart_stream(self, symbol, interval, initial_data=None):
        """Start chart WebSocket for a specific symbol and interval"""
        key = f"{symbol}_{interval}"
        
        with self.lock:
            if key not in self.chart_ws_dict:
                chart_ws = ChartWebSocket(symbol, interval)
                chart_ws.start(initial_data)
                self.chart_ws_dict[key] = chart_ws
                time.sleep(0.3)
            
            return self.chart_ws_dict[key]
    
    def stop_chart_stream(self, symbol, interval):
        """Stop chart WebSocket for a specific symbol and interval"""
        key = f"{symbol}_{interval}"
        
        with self.lock:
            if key in self.chart_ws_dict:
                self.chart_ws_dict[key].stop()
                del self.chart_ws_dict[key]
                print(f"🛑 Stopped chart stream: {key}")
    
    def get_price(self, symbol):
        """Get latest price from price WebSocket"""
        if self.price_ws:
            return self.price_ws.get_price(symbol)
        return None
    
    def get_chart_data(self, symbol, interval):
        """Get chart DataFrame"""
        key = f"{symbol}_{interval}"
        
        with self.lock:
            if key in self.chart_ws_dict:
                return self.chart_ws_dict[key].get_dataframe()
        return None
    
    def stop_all(self):
        """Stop all WebSocket connections"""
        if self.price_ws:
            self.price_ws.stop()
            print("🛑 Price WebSocket stopped")
        
        with self.lock:
            for key, chart_ws in list(self.chart_ws_dict.items()):
                chart_ws.stop()
                print(f"🛑 Chart WebSocket stopped: {key}")
            
            self.chart_ws_dict.clear()
