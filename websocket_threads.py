"""
websocket_threads.py
WebSocket threads for real-time price updates
"""

from PyQt5.QtCore import QThread, pyqtSignal
import json
import websocket
import threading
import time


class BinanceWebSocketThread(QThread):
    """Real-time price updates using Binance WebSocket"""
    prices_updated = pyqtSignal(dict)  # {symbol: ticker_data}
    connection_status = pyqtSignal(bool)  # True=connected, False=disconnected
    
    def __init__(self, symbols):
        super().__init__()
        self.symbols = symbols
        self.running = True
        self.ws = None
        self.prices = {}
        self.connected = False
        
        # Convert symbols to lowercase for WebSocket
        self.streams = [f"{symbol.lower()}@ticker" for symbol in symbols]
    
    def run(self):
        """Main thread loop"""
        while self.running:
            try:
                self.connect_websocket()
            except Exception as e:
                print(f"WebSocket connection error: {e}")
                self.connected = False
                self.connection_status.emit(False)
                
                if self.running:
                    print("Reconnecting in 5 seconds...")
                    time.sleep(5)
    
    def connect_websocket(self):
        """Connect to Binance WebSocket"""
        # Binance WebSocket URL for multiple streams
        stream_names = '/'.join(self.streams)
        ws_url = f"wss://stream.binance.com:9443/stream?streams={stream_names}"
        
        print(f"Connecting to WebSocket: {ws_url}")
        
        def on_message(ws, message):
            try:
                data = json.loads(message)
                if 'data' in data:
                    ticker = data['data']
                    symbol = ticker['s']  # Symbol name
                    
                    # Update prices dictionary
                    self.prices[symbol] = {
                        'price': float(ticker['c']),  # Current price
                        'change': float(ticker['p']),  # Price change
                        'change_percent': float(ticker['P']),  # Price change percent
                        'high': float(ticker['h']),  # 24h high
                        'low': float(ticker['l']),  # 24h low
                        'volume': float(ticker['v'])  # 24h volume
                    }
                    
                    # Emit updated prices
                    self.prices_updated.emit(self.prices.copy())
                    
            except Exception as e:
                print(f"WebSocket message error: {e}")
        
        def on_error(ws, error):
            print(f"WebSocket error: {error}")
            self.connected = False
            self.connection_status.emit(False)
        
        def on_close(ws, close_status_code, close_msg):
            print(f"WebSocket closed: {close_status_code} - {close_msg}")
            self.connected = False
            self.connection_status.emit(False)
        
        def on_open(ws):
            print("✅ WebSocket connected - Real-time updates active")
            self.connected = True
            self.connection_status.emit(True)
        
        # Create WebSocket connection
        self.ws = websocket.WebSocketApp(
            ws_url,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close,
            on_open=on_open
        )
        
        # Run WebSocket (blocking call)
        self.ws.run_forever(
            ping_interval=20,  # Send ping every 20 seconds
            ping_timeout=10    # Timeout after 10 seconds
        )
    
    def stop(self):
        """Stop WebSocket thread"""
        print("Stopping WebSocket thread...")
        self.running = False
        if self.ws:
            self.ws.close()


class MultiSymbolWebSocketThread(QThread):
    """
    Alternative WebSocket implementation with better error handling
    """
    prices_updated = pyqtSignal(dict)
    connection_status = pyqtSignal(bool)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, symbols):
        super().__init__()
        self.symbols = symbols
        self.running = True
        self.ws = None
        self.prices = {}
        self.reconnect_delay = 5
        self.max_reconnect_delay = 60
    
    def run(self):
        """Main thread loop with exponential backoff"""
        while self.running:
            try:
                self.connect_and_listen()
                self.reconnect_delay = 5  # Reset delay on successful connection
            except Exception as e:
                error_msg = f"WebSocket error: {str(e)}"
                print(error_msg)
                self.error_occurred.emit(error_msg)
                self.connection_status.emit(False)
                
                if self.running:
                    print(f"Reconnecting in {self.reconnect_delay} seconds...")
                    time.sleep(self.reconnect_delay)
                    
                    # Exponential backoff
                    self.reconnect_delay = min(
                        self.reconnect_delay * 2,
                        self.max_reconnect_delay
                    )
    
    def connect_and_listen(self):
        """Connect to WebSocket and listen for messages"""
        streams = [f"{symbol.lower()}@ticker" for symbol in self.symbols]
        stream_names = '/'.join(streams)
        ws_url = f"wss://stream.binance.com:9443/stream?streams={stream_names}"
        
        self.ws = websocket.create_connection(ws_url)
        self.connection_status.emit(True)
        print("✅ WebSocket connected")
        
        while self.running:
            try:
                message = self.ws.recv()
                if message:
                    self.process_message(message)
            except websocket.WebSocketTimeoutException:
                continue
            except Exception as e:
                raise e
    
    def process_message(self, message):
        """Process incoming WebSocket message"""
        try:
            data = json.loads(message)
            if 'data' in data:
                ticker = data['data']
                symbol = ticker['s']
                
                self.prices[symbol] = {
                    'price': float(ticker['c']),
                    'change': float(ticker['p']),
                    'change_percent': float(ticker['P']),
                    'high': float(ticker['h']),
                    'low': float(ticker['l']),
                    'volume': float(ticker['v'])
                }
                
                self.prices_updated.emit(self.prices.copy())
        except Exception as e:
            print(f"Error processing message: {e}")
    
    def stop(self):
        """Stop thread"""
        self.running = False
        if self.ws:
            try:
                self.ws.close()
            except:
                pass
