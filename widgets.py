"""
widgets.py
Custom widgets for the application
"""

from PyQt5.QtWidgets import (QWidget, QHBoxLayout, QVBoxLayout, QLabel, 
                             QPushButton, QDialog, QComboBox, QGroupBox,
                             QTextEdit, QTableWidget, QTableWidgetItem,
                             QScrollArea, QMessageBox)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont, QColor, QTextCursor
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pandas as pd


class ChartCanvas(FigureCanvas):
    """Canvas for matplotlib charts"""
    def __init__(self, parent=None, width=12, height=6, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        super().__init__(self.fig)
        self.setParent(parent)


class TickerBar(QWidget):
    """Horizontal ticker bar showing multiple symbols"""
    symbol_clicked = pyqtSignal(str)
    
    def __init__(self, symbols):
        super().__init__()
        self.symbols = symbols
        self.current_index = 0
        self.ticker_labels = {}
        self.init_ui()
    
    def init_ui(self):
        layout = QHBoxLayout()
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(15)
        
        # Create labels for first 5 symbols
        self.visible_count = 5
        for i in range(min(self.visible_count, len(self.symbols))):
            symbol = self.symbols[i]
            label = self.create_ticker_label(symbol)
            self.ticker_labels[symbol] = label
            layout.addWidget(label)
        
        layout.addStretch()
        
        # Next button
        if len(self.symbols) > self.visible_count:
            self.next_button = QPushButton(">")
            self.next_button.setMaximumWidth(35)
            self.next_button.setMaximumHeight(35)
            self.next_button.setStyleSheet("""
                QPushButton {
                    background-color: #95a5a6;
                    color: white;
                    border: none;
                    padding: 5px;
                    font-size: 18px;
                    font-weight: bold;
                    border-radius: 4px;
                }
                QPushButton:hover {
                    background-color: #7f8c8d;
                }
            """)
            self.next_button.clicked.connect(self.show_next_symbols)
            layout.addWidget(self.next_button)
        
        self.setLayout(layout)
        self.setMaximumHeight(55)
        self.setStyleSheet("""
            QWidget {
                background-color: #f8f9fa;
                border: 1px solid #dee2e6;
                border-radius: 8px;
            }
        """)
    
    def create_ticker_label(self, symbol):
        """Create clickable ticker label"""
        label = QLabel(f"{symbol}: $0.00 (--)")
        label.setFont(QFont("Arial", 11, QFont.Weight.Bold))
        label.setStyleSheet("""
            QLabel {
                color: #2c3e50;
                padding: 8px 15px;
                border-radius: 5px;
                background-color: white;
                border: 1px solid #dee2e6;
            }
            QLabel:hover {
                background-color: #e9ecef;
                border: 1px solid #adb5bd;
                cursor: pointer;
            }
        """)
        label.mousePressEvent = lambda event: self.symbol_clicked.emit(symbol)
        return label
    
    def update_ticker(self, symbol, ticker_data):
        """Update ticker display for a symbol"""
        if symbol not in self.ticker_labels:
            return
        
        price = ticker_data['price']
        change_percent = ticker_data['change_percent']
        
        # Format text with HTML for colored percentage
        if change_percent >= 0:
            color = "#27ae60"
            sign = "+"
        else:
            color = "#e74c3c"
            sign = ""
        
        # Format price display
        if price >= 1000:
            price_str = f"${price:,.2f}"
        else:
            price_str = f"${price:.4f}"
        
        text = f"<b>{symbol}</b>: {price_str} <span style='color: {color};'>({sign}{change_percent:.2f}%)</span>"
        
        self.ticker_labels[symbol].setText(text)
    
    def show_next_symbols(self):
        """Show next set of symbols"""
        # Clear current layout
        layout = self.layout()
        for i in reversed(range(layout.count())):
            widget = layout.itemAt(i).widget()
            if widget and widget != self.next_button:
                widget.setParent(None)
        
        # Update index
        self.current_index = (self.current_index + self.visible_count) % len(self.symbols)
        
        # Show next symbols
        self.ticker_labels.clear()
        for i in range(self.visible_count):
            idx = (self.current_index + i) % len(self.symbols)
            symbol = self.symbols[idx]
            label = self.create_ticker_label(symbol)
            self.ticker_labels[symbol] = label
            layout.insertWidget(i, label)


class RealtimePriceWidget(QWidget):
    """Real-time price display for selected symbol"""
    symbol_clicked = pyqtSignal(str)
    
    def __init__(self):
        super().__init__()
        self.current_symbol = "ETHUSDT"
        self.init_ui()
    
    def init_ui(self):
        layout = QHBoxLayout()
        layout.setSpacing(20)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # Symbol label - CLICKABLE
        self.symbol_display = QLabel("ETHUSDT")
        self.symbol_display.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        self.symbol_display.setStyleSheet("""
            QLabel {
                color: #2c3e50;
                padding: 8px 15px;
                border-radius: 6px;
                background-color: #ecf0f1;
                border: 2px solid #bdc3c7;
            }
            QLabel:hover {
                background-color: #d5dbdb;
                border: 2px solid #95a5a6;
                cursor: pointer;
            }
        """)
        # Make it clickable
        self.symbol_display.mousePressEvent = self.on_symbol_clicked
        
        # Current price
        self.current_price_label = QLabel("$0.00")
        self.current_price_label.setFont(QFont("Arial", 24, QFont.Weight.Bold))
        self.current_price_label.setStyleSheet("color: #27ae60;")
        
        # 24h change
        self.change_label = QLabel("--")
        self.change_label.setFont(QFont("Arial", 13, QFont.Weight.Bold))
        
        # Stats container
        stats_widget = QWidget()
        stats_layout = QHBoxLayout()
        stats_layout.setSpacing(20)
        stats_layout.setContentsMargins(0, 0, 0, 0)
        
        # 24h high
        self.high_label = QLabel("High: $0.00")
        self.high_label.setFont(QFont("Arial", 12))
        self.high_label.setStyleSheet("color: #555;")
        
        # 24h low
        self.low_label = QLabel("Low: $0.00")
        self.low_label.setFont(QFont("Arial", 12))
        self.low_label.setStyleSheet("color: #555;")
        
        # 24h volume
        self.volume_label = QLabel("Volume: 0")
        self.volume_label.setFont(QFont("Arial", 12))
        self.volume_label.setStyleSheet("color: #555;")
        
        stats_layout.addWidget(self.high_label)
        stats_layout.addWidget(self.low_label)
        stats_layout.addWidget(self.volume_label)
        stats_widget.setLayout(stats_layout)
        
        # Last update time
        self.update_time_label = QLabel("Last update: --")
        self.update_time_label.setFont(QFont("Arial", 10))
        self.update_time_label.setStyleSheet("color: #888;")
        
        layout.addWidget(self.symbol_display)
        layout.addWidget(self.current_price_label)
        layout.addWidget(self.change_label)
        layout.addSpacing(30)
        layout.addWidget(stats_widget)
        layout.addStretch()
        layout.addWidget(self.update_time_label)
        
        self.setLayout(layout)
        self.setMaximumHeight(80)
        self.setStyleSheet("""
            QWidget {
                background-color: white;
                border: 1px solid #ddd;
                border-radius: 8px;
            }
        """)
    
    def on_symbol_clicked(self, event):
        """Handle symbol label click"""
        self.symbol_clicked.emit(self.current_symbol)
    
    def update_symbol(self, symbol):
        """Update displayed symbol"""
        self.current_symbol = symbol
        self.symbol_display.setText(symbol)
    
    def update_price(self, ticker_data):
        """Update price display"""
        price = ticker_data['price']
        change = ticker_data['change']
        change_percent = ticker_data['change_percent']
        high = ticker_data['high']
        low = ticker_data['low']
        volume = ticker_data['volume']
        
        # Update price
        self.current_price_label.setText(f"${price:,.2f}")
        
        # Update change with color and arrow
        if change >= 0:
            self.current_price_label.setStyleSheet("color: #27ae60;")
            arrow = "📈"
            self.change_label.setText(f"{arrow} ${change:.2f} (+{change_percent:.2f}%)")
            self.change_label.setStyleSheet("color: #27ae60; font-weight: bold;")
        else:
            self.current_price_label.setStyleSheet("color: #e74c3c;")
            arrow = "📉"
            self.change_label.setText(f"{arrow} ${abs(change):.2f} ({change_percent:.2f}%)")
            self.change_label.setStyleSheet("color: #e74c3c; font-weight: bold;")
        
        # Update other info
        self.high_label.setText(f"High: ${high:,.2f}")
        self.low_label.setText(f"Low: ${low:,.2f}")
        
        # Format volume
        if volume >= 1_000_000:
            volume_str = f"{volume/1_000_000:.2f}M"
        elif volume >= 1_000:
            volume_str = f"{volume/1_000:.2f}K"
        else:
            volume_str = f"{volume:,.0f}"
        self.volume_label.setText(f"Volume: {volume_str}")
        
        # Update time
        from datetime import datetime
        self.update_time_label.setText(f"Last update: {datetime.now().strftime('%H:%M:%S')}")


class CandlestickDialog(QDialog):
    """Dialog showing candlestick chart with real-time updates"""
    
    def __init__(self, symbol, parent=None):
        super().__init__(parent)
        self.symbol = symbol
        self.df = None
        self.current_interval = '1h'
        self.update_thread = None
        self.init_ui()
    
    def init_ui(self):
        self.setWindowTitle(f"{self.symbol} - Real-time Candlestick Chart")
        self.setGeometry(100, 100, 1400, 800)
        
        layout = QVBoxLayout()
        
        # Control panel
        control_panel = QWidget()
        control_layout = QHBoxLayout()
        
        interval_label = QLabel("Timeframe:")
        interval_label.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        
        self.interval_combo = QComboBox()
        self.interval_combo.addItems(['15m', '1h', '4h', '1d', '1w'])
        self.interval_combo.setCurrentText('1h')
        self.interval_combo.currentTextChanged.connect(self.on_interval_changed)
        self.interval_combo.setStyleSheet("""
            QComboBox {
                padding: 5px 10px;
                font-size: 11px;
                font-weight: bold;
                border: 2px solid #3498db;
                border-radius: 4px;
                background-color: white;
            }
        """)
        
        # Auto-update checkbox
        self.auto_update_checkbox = QPushButton("🔄 Real-time: ON")
        self.auto_update_checkbox.setCheckable(True)
        self.auto_update_checkbox.setChecked(True)
        self.auto_update_checkbox.clicked.connect(self.toggle_auto_update)
        self.auto_update_checkbox.setStyleSheet("""
            QPushButton {
                background-color: #27ae60;
                color: white;
                padding: 5px 15px;
                font-size: 11px;
                font-weight: bold;
                border-radius: 4px;
                border: none;
            }
            QPushButton:checked {
                background-color: #27ae60;
            }
            QPushButton:!checked {
                background-color: #95a5a6;
            }
            QPushButton:hover {
                opacity: 0.8;
            }
        """)
        
        self.status_label = QLabel("Loading...")
        self.status_label.setStyleSheet("color: blue; font-size: 11px;")
        
        # Current price display
        self.current_price_display = QLabel("--")
        self.current_price_display.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        self.current_price_display.setStyleSheet("color: #27ae60; padding: 5px 10px; background-color: #f0f0f0; border-radius: 4px;")
        
        control_layout.addWidget(interval_label)
        control_layout.addWidget(self.interval_combo)
        control_layout.addSpacing(20)
        control_layout.addWidget(self.auto_update_checkbox)
        control_layout.addSpacing(20)
        control_layout.addWidget(QLabel("Current Price:"))
        control_layout.addWidget(self.current_price_display)
        control_layout.addSpacing(20)
        control_layout.addWidget(self.status_label)
        control_layout.addStretch()
        
        control_panel.setLayout(control_layout)
        layout.addWidget(control_panel)
        
        # Chart canvas with interactive features
        self.canvas = TradingChartCanvas(self, width=14, height=10)
        layout.addWidget(self.canvas)
        
        # Info label for crosshair
        self.info_label = QLabel("")
        self.info_label.setFont(QFont("Courier New", 9))
        self.info_label.setStyleSheet("background-color: #2c3e50; color: white; padding: 5px; border-radius: 3px;")
        self.info_label.setWordWrap(True)
        layout.addWidget(self.info_label)
        
        # Instructions
        instructions = QLabel("💡 Left Click: Set price line | Right/Middle Click + Drag: Pan chart | Scroll: Zoom in/out")
        instructions.setFont(QFont("Arial", 9))
        instructions.setStyleSheet("color: #7f8c8d; padding: 3px;")
        layout.addWidget(instructions)
        
        self.setLayout(layout)
        
        # Load initial data
        self.load_data('1h')
        
        # Start auto-update
        self.start_auto_update()
    
    def start_auto_update(self):
        """Start real-time updates"""
        if self.update_thread and self.update_thread.isRunning():
            return
        
        from threads import CandlestickUpdateThread
        self.update_thread = CandlestickUpdateThread(self.symbol, self.current_interval)
        self.update_thread.data_ready.connect(self.on_realtime_update)
        self.update_thread.start()
    
    def stop_auto_update(self):
        """Stop real-time updates"""
        if self.update_thread and self.update_thread.isRunning():
            self.update_thread.stop()
            self.update_thread.wait()
    
    def toggle_auto_update(self, checked):
        """Toggle auto-update"""
        if checked:
            self.auto_update_checkbox.setText("🔄 Real-time: ON")
            self.start_auto_update()
        else:
            self.auto_update_checkbox.setText("⏸ Real-time: OFF")
            self.stop_auto_update()
    
    def load_data(self, interval):
        """Load candlestick data"""
        from threads import CandlestickDataThread
        
        self.current_interval = interval
        self.status_label.setText(f"Loading {interval} data...")
        self.status_label.setStyleSheet("color: blue; font-size: 11px;")
        
        self.thread = CandlestickDataThread(self.symbol, interval, limit=200)
        self.thread.data_ready.connect(self.on_data_ready)
        self.thread.error.connect(self.on_data_error)
        self.thread.start()
    
    def on_data_ready(self, df):
        """Handle data ready"""
        self.df = df
        self.status_label.setText(f"✅ Loaded {len(df)} candles")
        self.status_label.setStyleSheet("color: green; font-size: 11px;")
        
        # Update current price
        if len(df) > 0:
            current_price = df.iloc[-1]['close']
            self.current_price_display.setText(f"${current_price:,.2f}")
            if df.iloc[-1]['close'] >= df.iloc[-1]['open']:
                self.current_price_display.setStyleSheet("color: #27ae60; padding: 5px 10px; background-color: #d5f4e6; border-radius: 4px; font-weight: bold;")
            else:
                self.current_price_display.setStyleSheet("color: #e74c3c; padding: 5px 10px; background-color: #fadbd8; border-radius: 4px; font-weight: bold;")
        
        self.plot_candlestick()
    
    def on_realtime_update(self, df):
        """Handle real-time update"""
        self.df = df
        
        # Update current price
        if len(df) > 0:
            current_price = df.iloc[-1]['close']
            self.current_price_display.setText(f"${current_price:,.2f}")
            if df.iloc[-1]['close'] >= df.iloc[-1]['open']:
                self.current_price_display.setStyleSheet("color: #27ae60; padding: 5px 10px; background-color: #d5f4e6; border-radius: 4px; font-weight: bold;")
            else:
                self.current_price_display.setStyleSheet("color: #e74c3c; padding: 5px 10px; background-color: #fadbd8; border-radius: 4px; font-weight: bold;")
        
        self.plot_candlestick()
    
    def on_data_error(self, error_msg):
        """Handle data error"""
        self.status_label.setText(f"❌ Error")
        self.status_label.setStyleSheet("color: red; font-size: 11px;")
        QMessageBox.warning(self, "Error", error_msg)
    
    def on_interval_changed(self, interval):
        """Handle interval change"""
        self.stop_auto_update()
        self.load_data(interval)
        if self.auto_update_checkbox.isChecked():
            self.start_auto_update()
    
    def plot_candlestick(self):
        """Plot candlestick chart with volume"""
        if self.df is None or len(self.df) == 0:
            return
        
        self.canvas.plot_candlestick(self.df, self.symbol, self.current_interval, self.info_label)
    
    def closeEvent(self, event):
        """Handle dialog close"""
        self.stop_auto_update()
        event.accept()
    
    def start_auto_update(self):
        """Start real-time updates"""
        if self.update_thread and self.update_thread.isRunning():
            return
        
        from threads import CandlestickUpdateThread
        self.update_thread = CandlestickUpdateThread(self.symbol, self.current_interval)
        self.update_thread.data_ready.connect(self.on_realtime_update)
        self.update_thread.start()
    
    def stop_auto_update(self):
        """Stop real-time updates"""
        if self.update_thread and self.update_thread.isRunning():
            self.update_thread.stop()
            self.update_thread.wait()
    
    def toggle_auto_update(self, checked):
        """Toggle auto-update"""
        if checked:
            self.auto_update_checkbox.setText("🔄 Real-time: ON")
            self.start_auto_update()
        else:
            self.auto_update_checkbox.setText("⏸ Real-time: OFF")
            self.stop_auto_update()
    
    def load_data(self, interval):
        """Load candlestick data"""
        from threads import CandlestickDataThread
        
        self.current_interval = interval
        self.status_label.setText(f"Loading {interval} data...")
        self.status_label.setStyleSheet("color: blue; font-size: 11px;")
        
        self.thread = CandlestickDataThread(self.symbol, interval, limit=200)
        self.thread.data_ready.connect(self.on_data_ready)
        self.thread.error.connect(self.on_data_error)
        self.thread.start()
    
    def on_data_ready(self, df):
        """Handle data ready"""
        self.df = df
        self.status_label.setText(f"✅ Loaded {len(df)} candles")
        self.status_label.setStyleSheet("color: green; font-size: 11px;")
        
        # Update current price
        if len(df) > 0:
            current_price = df.iloc[-1]['close']
            self.current_price_display.setText(f"${current_price:,.2f}")
            if df.iloc[-1]['close'] >= df.iloc[-1]['open']:
                self.current_price_display.setStyleSheet("color: #27ae60; padding: 5px 10px; background-color: #d5f4e6; border-radius: 4px; font-weight: bold;")
            else:
                self.current_price_display.setStyleSheet("color: #e74c3c; padding: 5px 10px; background-color: #fadbd8; border-radius: 4px; font-weight: bold;")
        
        self.plot_candlestick()
    
    def on_realtime_update(self, df):
        """Handle real-time update"""
        self.df = df
        
        # Update current price
        if len(df) > 0:
            current_price = df.iloc[-1]['close']
            self.current_price_display.setText(f"${current_price:,.2f}")
            if df.iloc[-1]['close'] >= df.iloc[-1]['open']:
                self.current_price_display.setStyleSheet("color: #27ae60; padding: 5px 10px; background-color: #d5f4e6; border-radius: 4px; font-weight: bold;")
            else:
                self.current_price_display.setStyleSheet("color: #e74c3c; padding: 5px 10px; background-color: #fadbd8; border-radius: 4px; font-weight: bold;")
        
        self.plot_candlestick()
    
    def on_data_error(self, error_msg):
        """Handle data error"""
        self.status_label.setText(f"❌ Error: {error_msg}")
        self.status_label.setStyleSheet("color: red; font-size: 11px;")
        QMessageBox.warning(self, "Error", error_msg)
    
    def on_interval_changed(self, interval):
        """Handle interval change"""
        self.stop_auto_update()
        self.load_data(interval)
        if self.auto_update_checkbox.isChecked():
            self.start_auto_update()
    
    def plot_candlestick(self):
        """Plot candlestick chart with volume"""
        if self.df is None or len(self.df) == 0:
            return
        
        self.canvas.plot_candlestick(self.df, self.symbol, self.current_interval, self.info_label)
    
    def closeEvent(self, event):
        """Handle dialog close"""
        self.stop_auto_update()
        event.accept()


class TradingChartCanvas(FigureCanvas):
    """Advanced trading chart canvas with interactive features"""
    
    def __init__(self, parent=None, width=14, height=10, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi, facecolor='#1e1e1e')
        super().__init__(self.fig)
        self.setParent(parent)
        
        # Interactive elements
        self.crosshair_lines = []
        self.price_line = None
        self.price_text = None
        self.df = None
        self.ax_candle = None
        self.ax_volume = None
        self.info_label = None
        
        # Pan/drag variables
        self.is_panning = False
        self.pan_start_x = None
        self.pan_start_xlim = None
        
        # Connect mouse events
        self.mpl_connect('motion_notify_event', self.on_mouse_move)
        self.mpl_connect('button_press_event', self.on_mouse_press)
        self.mpl_connect('button_release_event', self.on_mouse_release)
        self.mpl_connect('scroll_event', self.on_scroll)
        
        # Set cursor style
        self.setCursor(Qt.CursorShape.CrossCursor)
    
    def plot_candlestick(self, df, symbol, interval, info_label):
        """Plot candlestick chart with volume and interactive features"""
        self.df = df
        self.info_label = info_label
        
        self.fig.clear()
        
        # Create subplots: candlestick (70%) and volume (30%)
        gs = self.fig.add_gridspec(2, 1, height_ratios=[7, 3], hspace=0.05)
        self.ax_candle = self.fig.add_subplot(gs[0])
        self.ax_volume = self.fig.add_subplot(gs[1], sharex=self.ax_candle)
        
        # Style
        self.ax_candle.set_facecolor('#1e1e1e')
        self.ax_volume.set_facecolor('#1e1e1e')
        self.fig.patch.set_facecolor('#1e1e1e')
        
        # Plot candlesticks
        for idx, row in df.iterrows():
            open_price = row['open']
            close_price = row['close']
            high_price = row['high']
            low_price = row['low']
            
            # Determine color
            if close_price >= open_price:
                color = '#26a69a'  # Green (bullish)
                edge_color = '#26a69a'
            else:
                color = '#ef5350'  # Red (bearish)
                edge_color = '#ef5350'
            
            # Draw high-low line (wick)
            self.ax_candle.plot([idx, idx], [low_price, high_price], 
                               color=color, linewidth=1, alpha=0.8)
            
            # Draw body rectangle
            body_height = abs(close_price - open_price)
            body_bottom = min(open_price, close_price)
            
            if body_height < 0.01:  # Doji candle
                body_height = high_price * 0.001
            
            rect = Rectangle((idx - 0.4, body_bottom), 0.8, body_height,
                           facecolor=color, edgecolor=edge_color, 
                           linewidth=1.5, alpha=0.9)
            self.ax_candle.add_patch(rect)
        
        # Plot volume bars
        for idx, row in df.iterrows():
            if row['close'] >= row['open']:
                color = '#26a69a'
            else:
                color = '#ef5350'
            
            self.ax_volume.bar(idx, row['volume'], color=color, alpha=0.6, width=0.8)
        
        # Current price line (last close)
        current_price = df.iloc[-1]['close']
        self.price_line = self.ax_candle.axhline(y=current_price, color='#ffd700', 
                                                 linestyle='--', linewidth=2, 
                                                 alpha=0.8, label=f'Current: ${current_price:.2f}')
        
        # Price text on right side
        self.price_text = self.ax_candle.text(len(df), current_price, 
                                              f' ${current_price:.2f} ',
                                              verticalalignment='center',
                                              bbox=dict(boxstyle='round', 
                                                       facecolor='#ffd700', 
                                                       alpha=0.8),
                                              fontsize=10, fontweight='bold',
                                              color='#1e1e1e')
        
        # Format candlestick chart
        self.ax_candle.set_ylabel('Price ($)', fontweight='bold', color='white', fontsize=11)
        self.ax_candle.set_title(f'{symbol} - {interval.upper()} Candlestick Chart', 
                                fontweight='bold', fontsize=14, color='white', pad=15)
        self.ax_candle.grid(True, alpha=0.2, color='gray', linestyle='--')
        self.ax_candle.tick_params(colors='white', labelsize=9)
        
        # Hide x-axis labels for candlestick chart
        self.ax_candle.tick_params(axis='x', labelbottom=False)
        
        # Format volume chart
        self.ax_volume.set_xlabel('Time', fontweight='bold', color='white', fontsize=11)
        self.ax_volume.set_ylabel('Volume', fontweight='bold', color='white', fontsize=10)
        self.ax_volume.grid(True, alpha=0.2, color='gray', linestyle='--')
        self.ax_volume.tick_params(colors='white', labelsize=9)
        
        # Set x-axis labels (show timestamps)
        step = max(1, len(df) // 10)
        x_ticks = list(range(0, len(df), step))
        x_labels = [df.iloc[i]['timestamp'].strftime('%m-%d\n%H:%M') for i in x_ticks]
        self.ax_volume.set_xticks(x_ticks)
        self.ax_volume.set_xticklabels(x_labels, rotation=0, ha='center', color='white')
        
        # Format volume y-axis
        max_volume = df['volume'].max()
        if max_volume >= 1_000_000:
            self.ax_volume.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e6:.1f}M'))
        elif max_volume >= 1_000:
            self.ax_volume.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e3:.1f}K'))
        
        self.fig.tight_layout()
        self.draw()
    
    def on_mouse_move(self, event):
        """Handle mouse move - show crosshair and price"""
        if self.df is None:
            return
        
        # Handle panning
        if self.is_panning and event.inaxes == self.ax_candle:
            if self.pan_start_x is not None and event.xdata is not None:
                dx = self.pan_start_x - event.xdata
                new_xlim = [self.pan_start_xlim[0] + dx, self.pan_start_xlim[1] + dx]
                
                # Limit panning
                if new_xlim[0] >= 0 and new_xlim[1] <= len(self.df):
                    self.ax_candle.set_xlim(new_xlim)
                    self.draw()
            return
        
        if event.inaxes not in [self.ax_candle, self.ax_volume]:
            # Clear crosshair when mouse leaves
            for line in self.crosshair_lines:
                line.set_visible(False)
            self.draw()
            return
        
        # Get mouse position
        x, y = event.xdata, event.ydata
        
        if x is None or y is None:
            return
        
        # Clear old crosshair lines
        for line in self.crosshair_lines:
            line.remove()
        self.crosshair_lines.clear()
        
        # Draw new crosshair
        vline = self.ax_candle.axvline(x=x, color='#00bcd4', 
                                       linestyle=':', linewidth=1, alpha=0.7)
        hline = self.ax_candle.axhline(y=y, color='#00bcd4', 
                                       linestyle=':', linewidth=1, alpha=0.7)
        
        self.crosshair_lines.extend([vline, hline])
        
        # Get candle info at cursor position
        idx = int(round(x))
        if 0 <= idx < len(self.df):
            candle = self.df.iloc[idx]
            time_str = candle['timestamp'].strftime('%Y-%m-%d %H:%M')
            
            # Calculate change
            change = candle['close'] - candle['open']
            change_pct = (change / candle['open']) * 100 if candle['open'] != 0 else 0
            
            info_text = (f"📅 {time_str}  |  "
                        f"O: ${candle['open']:.2f}  "
                        f"H: ${candle['high']:.2f}  "
                        f"L: ${candle['low']:.2f}  "
                        f"C: ${candle['close']:.2f}  "
                        f"Vol: {candle['volume']:,.0f}  |  "
                        f"Change: ${change:.2f} ({change_pct:+.2f}%)  |  "
                        f"Cursor: ${y:.2f}")
            
            if self.info_label:
                self.info_label.setText(info_text)
        
        self.draw()
    
    def on_mouse_press(self, event):
        """Handle mouse press - start panning or set price line"""
        if event.inaxes != self.ax_candle or self.df is None:
            return
        
        # Right click or middle click - start panning
        if event.button in [2, 3]:  # Middle or right mouse button
            self.is_panning = True
            self.pan_start_x = event.xdata
            self.pan_start_xlim = self.ax_candle.get_xlim()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
        
        # Left click - set price line
        elif event.button == 1:
            y = event.ydata
            if y is None:
                return
            
            # Remove old price line
            if self.price_line:
                try:
                    self.price_line.remove()
                except:
                    pass
            if self.price_text:
                try:
                    self.price_text.remove()
                except:
                    pass
            
            # Draw new price line
            self.price_line = self.ax_candle.axhline(y=y, color='#ff9800', 
                                                     linestyle='--', linewidth=2, alpha=0.8)
            self.price_text = self.ax_candle.text(len(self.df), y, 
                                                  f' ${y:.2f} ',
                                                  verticalalignment='center',
                                                  bbox=dict(boxstyle='round', 
                                                           facecolor='#ff9800', 
                                                           alpha=0.8),
                                                  fontsize=10, fontweight='bold',
                                                  color='white')
            self.draw()
    
    def on_mouse_release(self, event):
        """Handle mouse release - stop panning"""
        if self.is_panning:
            self.is_panning = False
            self.pan_start_x = None
            self.pan_start_xlim = None
            self.setCursor(Qt.CursorShape.CrossCursor)
    
    def on_scroll(self, event):
        """Handle scroll - zoom in/out (FIXED: reversed direction)"""
        if event.inaxes not in [self.ax_candle, self.ax_volume]:
            return
        
        # Zoom factor - REVERSED
        scale_factor = 0.8 if event.button == 'up' else 1.2  # Đảo ngược
        
        # Get current limits
        xlim = self.ax_candle.get_xlim()
        
        # Calculate new limits (zoom around mouse position)
        xdata = event.xdata if event.xdata else (xlim[0] + xlim[1]) / 2
        
        new_width = (xlim[1] - xlim[0]) * scale_factor
        
        # Center zoom around mouse position
        left_ratio = (xdata - xlim[0]) / (xlim[1] - xlim[0])
        new_xlim = [xdata - new_width * left_ratio,
                   xdata + new_width * (1 - left_ratio)]
        
        # Limit zoom
        if new_xlim[1] - new_xlim[0] < 10:  # Min 10 candles
            return
        if new_xlim[1] - new_xlim[0] > len(self.df):  # Max all candles
            new_xlim = [0, len(self.df)]
        
        # Apply new limits
        self.ax_candle.set_xlim(new_xlim)
        
        self.draw()

class ConsoleWidget(QWidget):
    """Collapsible console log widget"""
    
    def __init__(self):
        super().__init__()
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Toggle button
        self.toggle_button = QPushButton("▼ Show Console Log")
        self.toggle_button.setStyleSheet("""
            QPushButton {
                background-color: #34495e;
                color: white;
                border: none;
                padding: 6px;
                font-size: 12px;
                border-radius: 3px;
                font-weight: bold;
                text-align: left;
            }
            QPushButton:hover {
                background-color: #2c3e50;
            }
        """)
        self.toggle_button.clicked.connect(self.toggle)
        layout.addWidget(self.toggle_button)
        
        # Console text area
        self.console_log = QTextEdit()
        self.console_log.setReadOnly(True)
        self.console_log.setMaximumHeight(200)
        self.console_log.setStyleSheet("""
            QTextEdit {
                background-color: #2c3e50;
                color: #ecf0f1;
                font-family: 'Courier New', monospace;
                font-size: 12px;
                border: 1px solid #34495e;
                border-radius: 3px;
            }
        """)
        self.console_log.setVisible(False)
        layout.addWidget(self.console_log)
        
        self.setLayout(layout)
        self.setMaximumHeight(250)
    
    def toggle(self):
        """Toggle console visibility"""
        is_visible = self.console_log.isVisible()
        self.console_log.setVisible(not is_visible)
        
        if is_visible:
            self.toggle_button.setText("▼ Show Console Log")
        else:
            self.toggle_button.setText("▲ Hide Console Log")
    
    def append(self, message):
        """Append message to console"""
        self.console_log.append(message)
        self.console_log.moveCursor(QTextCursor.MoveOperation.End)
    
    def clear(self):
        """Clear console"""
        self.console_log.clear()
    
    def show_console(self):
        """Show console"""
        self.console_log.setVisible(True)
        self.toggle_button.setText("▲ Hide Console Log")
    
    def hide_console(self):
        """Hide console"""
        self.console_log.setVisible(False)
        self.toggle_button.setText("▼ Show Console Log")
