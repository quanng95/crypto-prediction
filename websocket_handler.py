import json
import threading
import time
from websocket import create_connection, WebSocketConnectionClosedException

class BinanceWebSocket:
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
                retry_count = 0  # Reset on successful connection
                print(f"✅ WebSocket connected")
                
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
                        print("⚠️ WebSocket connection closed")
                        break
                    except json.JSONDecodeError:
                        continue
                    except Exception as e:
                        print(f"❌ Error processing message: {e}")
                        continue
                        
            except Exception as e:
                retry_count += 1
                print(f"❌ WebSocket error (attempt {retry_count}/{max_retries}): {e}")
                time.sleep(min(2 ** retry_count, 30))  # Exponential backoff, max 30s
            finally:
                if self.ws:
                    try:
                        self.ws.close()
                    except:
                        pass
        
        print("❌ WebSocket stopped")
        
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
