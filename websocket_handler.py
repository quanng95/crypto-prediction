import websocket
import json
import threading
import time

class BinanceWebSocket:
    def __init__(self):
        self.ws_spot = None
        self.ws_future = None
        self.thread_spot = None
        self.thread_future = None
        self.prices = {}
        self.running = False
    
    def start(self, symbols):
        """Start WebSocket for both Spot and Future symbols"""
        self.running = True
        
        # Separate Spot and Future symbols
        spot_symbols = [s.lower() for s in symbols if 'PERP' not in s.upper()]
        future_symbols = [s.lower() for s in symbols if 'PERP' in s.upper()]
        
        # Start Spot WebSocket
        if spot_symbols:
            streams_spot = '/'.join([f"{s}@ticker" for s in spot_symbols])
            url_spot = f"wss://stream.binance.com:9443/stream?streams={streams_spot}"
            
            self.thread_spot = threading.Thread(
                target=self._run_websocket,
                args=(url_spot, 'spot'),
                daemon=True
            )
            self.thread_spot.start()
        
        # Start Future WebSocket
        if future_symbols:
            streams_future = '/'.join([f"{s}@ticker" for s in future_symbols])
            url_future = f"wss://fstream.binance.com/stream?streams={streams_future}"
            
            self.thread_future = threading.Thread(
                target=self._run_websocket,
                args=(url_future, 'future'),
                daemon=True
            )
            self.thread_future.start()
    
    def _run_websocket(self, url, ws_type):
        """Run WebSocket connection"""
        def on_message(ws, message):
            data = json.loads(message)
            if 'data' in data:
                ticker = data['data']
                symbol = ticker['s'].upper()
                
                self.prices[symbol] = {
                    'price': float(ticker['c']),
                    'change_percent': float(ticker['P']),
                    'high': float(ticker['h']),
                    'low': float(ticker['l']),
                    'volume': float(ticker['v']),
                    'timestamp': time.time()
                }
        
        def on_error(ws, error):
            print(f"WebSocket error ({ws_type}): {error}")
        
        def on_close(ws, close_status_code, close_msg):
            print(f"WebSocket closed ({ws_type})")
        
        def on_open(ws):
            print(f"✅ WebSocket connected ({ws_type})")
        
        ws = websocket.WebSocketApp(
            url,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close,
            on_open=on_open
        )
        
        if ws_type == 'spot':
            self.ws_spot = ws
        else:
            self.ws_future = ws
        
        ws.run_forever()
    
    def get_price(self, symbol):
        """Get price for a symbol"""
        return self.prices.get(symbol.upper())
    
    def stop(self):
        """Stop all WebSocket connections"""
        self.running = False
        
        if self.ws_spot:
            self.ws_spot.close()
        
        if self.ws_future:
            self.ws_future.close()
