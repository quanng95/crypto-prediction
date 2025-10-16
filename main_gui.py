"""
main_gui.py
Main GUI application with WebSocket real-time updates (Optimized)
"""

import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QTableWidget, QTableWidgetItem, QPushButton,
                             QLabel, QTabWidget, QTextEdit, QComboBox, QGroupBox,
                             QProgressBar, QMessageBox, QScrollArea)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont, QColor
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from threads import AnalysisThread, CandlestickDataThread, CandlestickUpdateThread
from websocket_threads import BinanceWebSocketThread
from widgets import (ChartCanvas, TickerBar, CandlestickDialog, ConsoleWidget,
                    RealtimePriceWidget)


class ETHPredictorGUI(QMainWindow):
    """Main application window with WebSocket real-time updates"""
    
    def __init__(self):
        super().__init__()
        self.predictor = None
        self.all_predictions = None
        self.ws_thread = None  # WebSocket thread
        
        # Symbol list
        self.symbols = ["ETHUSDT", "BTCUSDT", "PAXGUSDT", "BNBUSDT", 
                       "SOLUSDT", "ADAUSDT", "DOGEUSDT", "ARBUSDT"]
        
        self.current_symbol = "ETHUSDT"  # Track current symbol
        
        # Connection status
        self.is_connected = False
        
        self.init_ui()
        
        # Start WebSocket monitoring AFTER UI is ready
        QTimer.singleShot(500, self.start_websocket_monitoring)
    
    def start_websocket_monitoring(self):
        """Start WebSocket monitoring for all symbols"""
        if self.ws_thread and self.ws_thread.isRunning():
            self.ws_thread.stop()
            self.ws_thread.wait()
        
        self.ws_thread = BinanceWebSocketThread(self.symbols)
        self.ws_thread.prices_updated.connect(self.on_prices_updated)
        self.ws_thread.connection_status.connect(self.on_connection_status)
        self.ws_thread.start()
    
    def on_connection_status(self, connected):
        """Handle WebSocket connection status (silent)"""
        self.is_connected = connected
        # Chỉ log ra console, không hiển thị trên UI
        if connected:
            print("✅ WebSocket connected")
        else:
            print("⚠️ WebSocket disconnected - reconnecting...")
    
    def on_prices_updated(self, all_prices):
        """Handle real-time price updates from WebSocket"""
        # Update ticker bar
        for symbol, ticker_data in all_prices.items():
            self.ticker_bar.update_ticker(symbol, ticker_data)
        
        # Update price widget if current symbol is in the update
        if self.current_symbol in all_prices:
            self.price_widget.update_price(all_prices[self.current_symbol])
    
    def init_ui(self):
        """Initialize user interface"""
        self.setWindowTitle("🔮 Crypto Prediction - Real-time")
        self.setGeometry(50, 50, 1800, 1000)
        
        # Main widget
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        main_layout.setSpacing(5)
        main_layout.setContentsMargins(10, 10, 10, 10)
        
        # Ticker bar (TOP)
        self.ticker_bar = TickerBar(self.symbols)
        self.ticker_bar.symbol_clicked.connect(self.on_ticker_symbol_clicked)
        main_layout.addWidget(self.ticker_bar)
        
        # Real-time price widget
        self.price_widget = RealtimePriceWidget()
        self.price_widget.symbol_clicked.connect(self.on_price_widget_symbol_clicked)
        main_layout.addWidget(self.price_widget)
        
        # Control panel
        control_panel = self.create_control_panel()
        main_layout.addWidget(control_panel)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setMaximumHeight(20)
        main_layout.addWidget(self.progress_bar)
        
        # Status label
        self.status_label = QLabel("Ready to analyze")
        self.status_label.setStyleSheet("color: blue; font-size: 12px; padding: 2px;")
        main_layout.addWidget(self.status_label)
        
        # Console log
        self.console = ConsoleWidget()
        main_layout.addWidget(self.console)
        
        # Tab widget for results
        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)
        
        # Apply styles
        self.apply_styles()
    
    def apply_styles(self):
        """Apply application styles"""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f5f6fa;
            }
            QGroupBox {
                font-weight: bold;
                border: 1px solid #ddd;
                border-radius: 8px;
                margin-top: 8px;
                padding-top: 10px;
                background-color: white;
                font-size: 12px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 15px;
                padding: 0 8px 0 8px;
            }
            QPushButton {
                background-color: #27ae60;
                color: white;
                border: none;
                padding: 10px 20px;
                font-size: 13px;
                border-radius: 6px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #229954;
            }
            QPushButton:disabled {
                background-color: #bdc3c7;
            }
            QTableWidget {
                background-color: white;
                alternate-background-color: #f8f9fa;
                gridline-color: #dee2e6;
                border: 1px solid #dee2e6;
                border-radius: 5px;
            }
            QHeaderView::section {
                background-color: #27ae60;
                color: white;
                padding: 8px;
                font-weight: bold;
                border: none;
            }
            QTabWidget::pane {
                border: 1px solid #dee2e6;
                background-color: white;
                border-radius: 5px;
            }
            QTabBar::tab {
                background-color: #e9ecef;
                padding: 10px 20px;
                margin-right: 3px;
                border-top-left-radius: 5px;
                border-top-right-radius: 5px;
                color: #495057;
                font-weight: bold;
            }
            QTabBar::tab:selected {
                background-color: #27ae60;
                color: white;
            }
            QTabBar::tab:hover:!selected {
                background-color: #dee2e6;
            }
        """)
    
    def create_control_panel(self):
        """Create control panel"""
        panel = QGroupBox("Control Panel")
        layout = QHBoxLayout()
        
        # Symbol input
        symbol_label = QLabel("Symbol:")
        symbol_label.setFont(QFont("Arial", 11, QFont.Weight.Bold))
        
        self.symbol_input = QComboBox()
        self.symbol_input.addItems(self.symbols)
        self.symbol_input.setFont(QFont("Arial", 11, QFont.Weight.Bold))
        self.symbol_input.setStyleSheet("""
            QComboBox {
                background-color: white;
                border: 2px solid #3498db;
                border-radius: 5px;
                padding: 8px 12px;
                min-width: 120px;
                color: #2c3e50;
            }
            QComboBox:hover {
                background-color: #ecf0f1;
                border: 2px solid #2980b9;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox::down-arrow {
                image: none;
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 6px solid #3498db;
                margin-right: 8px;
            }
            QComboBox QAbstractItemView {
                background-color: white;
                border: 2px solid #3498db;
                selection-background-color: #3498db;
                selection-color: white;
                padding: 5px;
            }
        """)
        self.symbol_input.currentTextChanged.connect(self.on_symbol_changed)
        
        # Timezone input
        timezone_label = QLabel("Timezone:")
        timezone_label.setFont(QFont("Arial", 11, QFont.Weight.Bold))
        self.timezone_input = QComboBox()
        self.timezone_input.addItems([
            "Asia/Ho_Chi_Minh",
            "America/New_York",
            "Europe/London",
            "Asia/Tokyo",
            "Asia/Shanghai"
        ])
        self.timezone_input.setFont(QFont("Arial", 10))
        self.timezone_input.setStyleSheet("""
            QComboBox {
                background-color: white;
                border: 1px solid #bdc3c7;
                border-radius: 4px;
                padding: 6px 10px;
                min-width: 150px;
            }
            QComboBox:hover {
                border: 1px solid #95a5a6;
            }
            QComboBox QAbstractItemView {
                background-color: white;
                border: 1px solid #bdc3c7;
                selection-background-color: #ecf0f1;
                selection-color: #2c3e50;
            }
        """)
        
        # Run button
        self.run_button = QPushButton("🚀 Run Analysis")
        self.run_button.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        self.run_button.clicked.connect(self.run_analysis)
        
        layout.addWidget(symbol_label)
        layout.addWidget(self.symbol_input)
        layout.addSpacing(30)
        layout.addWidget(timezone_label)
        layout.addWidget(self.timezone_input)
        layout.addSpacing(30)
        layout.addWidget(self.run_button)
        layout.addStretch()
        
        panel.setLayout(layout)
        panel.setMaximumHeight(90)
        return panel
    
    def on_symbol_changed(self, symbol):
        """Handle symbol change in control panel"""
        self.current_symbol = symbol
        self.price_widget.update_symbol(symbol)
    
    def on_ticker_symbol_clicked(self, symbol):
        """Handle ticker symbol click - show candlestick chart"""
        # Update current symbol
        self.current_symbol = symbol
        self.symbol_input.setCurrentText(symbol)
        self.price_widget.update_symbol(symbol)
        
        # Show candlestick dialog
        dialog = CandlestickDialog(symbol, self)
        dialog.exec()
    
    def on_price_widget_symbol_clicked(self, symbol):
        """Handle click on price widget symbol - show candlestick chart"""
        dialog = CandlestickDialog(symbol, self)
        dialog.exec()
    
    def run_analysis(self):
        """Start analysis"""
        self.run_button.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)
        self.status_label.setText("Running analysis...")
        self.status_label.setStyleSheet("color: orange; font-size: 11px;")
        
        self.tab_widget.clear()
        self.console.clear()
        self.console.show_console()
        
        symbol = self.symbol_input.currentText()
        timezone = self.timezone_input.currentText()
        
        self.thread = AnalysisThread(symbol, timezone)
        self.thread.finished.connect(self.on_analysis_finished)
        self.thread.progress.connect(self.on_progress_update)
        self.thread.error.connect(self.on_analysis_error)
        self.thread.log_message.connect(self.console.append)
        self.thread.start()
    
    def on_progress_update(self, message):
        """Update progress"""
        self.status_label.setText(message)
    
    def on_analysis_finished(self, predictor, all_predictions):
        """Handle analysis completion"""
        self.predictor = predictor
        self.all_predictions = all_predictions
        
        self.progress_bar.setVisible(False)
        self.run_button.setEnabled(True)
        self.status_label.setText("✅ Analysis complete!")
        self.status_label.setStyleSheet("color: green; font-size: 12px;")
        
        QTimer.singleShot(2000, self.console.hide_console)
        
        self.display_results()
    
    def on_analysis_error(self, error_msg):
        """Handle analysis error"""
        self.progress_bar.setVisible(False)
        self.run_button.setEnabled(True)
        self.status_label.setText("❌ Analysis failed")
        self.status_label.setStyleSheet("color: red; font-size: 11px;")
        
        self.console.append(f"\n❌ ERROR: {error_msg}")
        QMessageBox.critical(self, "Error", error_msg)
    
    def display_results(self):
        """Display analysis results"""
        if not self.predictor or not self.all_predictions:
            return
        
        self.create_summary_tab()
        
        for timeframe in ['4h', '1d', '1w']:
            if timeframe in self.all_predictions:
                self.create_timeframe_tab(timeframe)
        
        self.create_comparison_tab()
        self.create_final_predictions_tab()
    
    def create_summary_tab(self):
        """Create summary tab"""
        tab = QWidget()
        main_layout = QVBoxLayout(tab)
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_widget = QWidget()
        layout = QVBoxLayout(scroll_widget)
        
        # Performance table
        perf_group = QGroupBox("🏆 Best Models Performance")
        perf_layout = QVBoxLayout()
        
        table = QTableWidget()
        table.setColumnCount(6)
        table.setHorizontalHeaderLabels(['Timeframe', 'Best Model', 'R² Score', 'MAE ($)', 'RMSE ($)', 'Direction Acc'])
        
        row = 0
        for timeframe in ['4h', '1d', '1w']:
            if timeframe in self.predictor.all_model_results:
                best_model = self.predictor.best_models.get(timeframe, '')
                if best_model and best_model in self.predictor.all_model_results[timeframe]:
                    result = self.predictor.all_model_results[timeframe][best_model]
                    
                    table.insertRow(row)
                    table.setItem(row, 0, QTableWidgetItem(timeframe.upper()))
                    table.setItem(row, 1, QTableWidgetItem(best_model))
                    table.setItem(row, 2, QTableWidgetItem(f"{result['r2']:.4f}"))
                    table.setItem(row, 3, QTableWidgetItem(f"${result['mae']:.2f}"))
                    table.setItem(row, 4, QTableWidgetItem(f"${result['rmse']:.2f}"))
                    table.setItem(row, 5, QTableWidgetItem(f"{result['direction_accuracy']:.2%}"))
                    
                    r2_item = table.item(row, 2)
                    if result['r2'] >= 0.8:
                        r2_item.setBackground(QColor(144, 238, 144))
                    elif result['r2'] >= 0.6:
                        r2_item.setBackground(QColor(255, 255, 153))
                    else:
                        r2_item.setBackground(QColor(255, 182, 193))
                    
                    row += 1
        
        table.resizeColumnsToContents()
        table.setMaximumHeight(150)
        perf_layout.addWidget(table)
        perf_group.setLayout(perf_layout)
        layout.addWidget(perf_group)
        
        # Charts
        charts_group = QGroupBox("📈 Performance Visualization")
        charts_layout = QVBoxLayout()
        
        chart_canvas = ChartCanvas(self, width=14, height=8)
        self.plot_summary_charts(chart_canvas)
        charts_layout.addWidget(chart_canvas)
        
        charts_group.setLayout(charts_layout)
        layout.addWidget(charts_group)
        
        scroll.setWidget(scroll_widget)
        main_layout.addWidget(scroll)
        
        self.tab_widget.addTab(tab, "📊 Summary")
    
    def plot_summary_charts(self, canvas):
        """Plot summary charts"""
        canvas.fig.clear()
        
        gs = canvas.fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        
        # Chart 1: R² Score
        ax1 = canvas.fig.add_subplot(gs[0, 0])
        timeframes = []
        r2_scores = []
        for tf in ['4h', '1d', '1w']:
            if tf in self.predictor.best_models:
                best_model = self.predictor.best_models[tf]
                result = self.predictor.all_model_results[tf][best_model]
                timeframes.append(tf.upper())
                r2_scores.append(result['r2'])
        
        bars = ax1.bar(timeframes, r2_scores, color=['#2ecc71', '#3498db', '#9b59b6'])
        ax1.set_ylabel('R² Score', fontweight='bold')
        ax1.set_title('Model Accuracy by Timeframe', fontweight='bold')
        ax1.set_ylim([0, 1])
        ax1.grid(axis='y', alpha=0.3)
        
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Chart 2: MAE
        ax2 = canvas.fig.add_subplot(gs[0, 1])
        mae_values = []
        for tf in ['4h', '1d', '1w']:
            if tf in self.predictor.best_models:
                best_model = self.predictor.best_models[tf]
                result = self.predictor.all_model_results[tf][best_model]
                mae_values.append(result['mae'])
        
        bars = ax2.bar(timeframes, mae_values, color=['#e74c3c', '#f39c12', '#1abc9c'])
        ax2.set_ylabel('MAE ($)', fontweight='bold')
        ax2.set_title('Mean Absolute Error', fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)
        
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'${height:.2f}', ha='center', va='bottom', fontweight='bold')
        
        # Chart 3: Direction Accuracy
        ax3 = canvas.fig.add_subplot(gs[1, 0])
        dir_acc = []
        for tf in ['4h', '1d', '1w']:
            if tf in self.predictor.best_models:
                best_model = self.predictor.best_models[tf]
                result = self.predictor.all_model_results[tf][best_model]
                dir_acc.append(result['direction_accuracy'] * 100)
        
        bars = ax3.bar(timeframes, dir_acc, color=['#16a085', '#27ae60', '#2980b9'])
        ax3.set_ylabel('Accuracy (%)', fontweight='bold')
        ax3.set_title('Direction Prediction Accuracy', fontweight='bold')
        ax3.set_ylim([0, 100])
        ax3.grid(axis='y', alpha=0.3)
        
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # Chart 4: Model Distribution
        ax4 = canvas.fig.add_subplot(gs[1, 1])
        model_names = [self.predictor.best_models.get(tf, 'N/A') for tf in ['4h', '1d', '1w'] if tf in self.predictor.best_models]
        model_counts = {}
        for model in model_names:
            model_counts[model] = model_counts.get(model, 0) + 1
        
        colors = plt.cm.Set3(range(len(model_counts)))
        ax4.pie(model_counts.values(), labels=model_counts.keys(), autopct='%1.0f%%',
               colors=colors, startangle=90)
        ax4.set_title('Best Model Distribution', fontweight='bold')
        
        canvas.draw()
    
    def create_timeframe_tab(self, timeframe):
        """Create timeframe-specific tab"""
        tab = QWidget()
        main_layout = QVBoxLayout(tab)
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_widget = QWidget()
        layout = QVBoxLayout(scroll_widget)
        
        # Model performance table
        perf_group = QGroupBox(f"🎯 {timeframe.upper()} - All Models Performance")
        perf_layout = QVBoxLayout()
        
        table = QTableWidget()
        table.setColumnCount(7)
        table.setHorizontalHeaderLabels(['Model', 'R²', 'MAE ($)', 'RMSE ($)', 'MAPE (%)', 'Direction Acc', 'Status'])
        
        if timeframe in self.predictor.all_model_results:
            results = self.predictor.all_model_results[timeframe]
            best_model = self.predictor.best_models.get(timeframe, '')
            
            row = 0
            for model_name, result in sorted(results.items(), key=lambda x: x[1]['r2'], reverse=True):
                table.insertRow(row)
                
                is_best = (model_name == best_model)
                status = "🏆 BEST" if is_best else ""
                
                table.setItem(row, 0, QTableWidgetItem(model_name))
                table.setItem(row, 1, QTableWidgetItem(f"{result['r2']:.4f}"))
                table.setItem(row, 2, QTableWidgetItem(f"${result['mae']:.2f}"))
                table.setItem(row, 3, QTableWidgetItem(f"${result['rmse']:.2f}"))
                table.setItem(row, 4, QTableWidgetItem(f"{result['mape']:.2f}%"))
                table.setItem(row, 5, QTableWidgetItem(f"{result['direction_accuracy']:.2%}"))
                table.setItem(row, 6, QTableWidgetItem(status))
                
                if is_best:
                    for col in range(7):
                        table.item(row, col).setBackground(QColor(255, 215, 0, 100))
                        table.item(row, col).setFont(QFont("Arial", 9, QFont.Weight.Bold))
                
                row += 1
        
        table.resizeColumnsToContents()
        table.setMaximumHeight(250)
        perf_layout.addWidget(table)
        perf_group.setLayout(perf_layout)
        layout.addWidget(perf_group)
        
        # Predictions table
        pred_group = QGroupBox(f"📈 {timeframe.upper()} - Predictions (First 7 Periods)")
        pred_layout = QVBoxLayout()
        
        if timeframe in self.all_predictions:
            pred_table = QTableWidget()
            models = list(self.all_predictions[timeframe].keys())
            pred_table.setColumnCount(len(models) + 1)
            pred_table.setHorizontalHeaderLabels(['Period'] + models)
            
            max_periods = max(len(preds) for preds in self.all_predictions[timeframe].values())
            
            for i in range(min(max_periods, 7)):
                pred_table.insertRow(i)
                
                if timeframe == '4h':
                    period = f"{(i+1)*4}h"
                elif timeframe == '1d':
                    period = f"Day {i+1}"
                else:
                    period = f"Week {i+1}"
                
                pred_table.setItem(i, 0, QTableWidgetItem(period))
                
                for col, model_name in enumerate(models, 1):
                    predictions = self.all_predictions[timeframe][model_name]
                    if i < len(predictions):
                        price = predictions[i]
                        change = ((price / self.predictor.reference_price - 1) * 100)
                        
                        item = QTableWidgetItem(f"${price:.2f} ({change:+.2f}%)")
                        
                        if change > 0:
                            item.setBackground(QColor(144, 238, 144, 100))
                        elif change < 0:
                            item.setBackground(QColor(255, 182, 193, 100))
                        
                        pred_table.setItem(i, col, item)
            
            pred_table.resizeColumnsToContents()
            pred_table.setMaximumHeight(250)
            pred_layout.addWidget(pred_table)
        
        pred_group.setLayout(pred_layout)
        layout.addWidget(pred_group)
        
        # Charts
        charts_group = QGroupBox(f"📊 {timeframe.upper()} - Visualization")
        charts_layout = QVBoxLayout()
        
        chart_canvas = ChartCanvas(self, width=14, height=6)
        self.plot_timeframe_charts(chart_canvas, timeframe)
        charts_layout.addWidget(chart_canvas)
        
        charts_group.setLayout(charts_layout)
        layout.addWidget(charts_group)
        
        scroll.setWidget(scroll_widget)
        main_layout.addWidget(scroll)
        
        self.tab_widget.addTab(tab, f"⏰ {timeframe.upper()}")
    
    def plot_timeframe_charts(self, canvas, timeframe):
        """Plot timeframe charts"""
        canvas.fig.clear()
        
        if timeframe not in self.all_predictions:
            return
        
        ax1 = canvas.fig.add_subplot(121)
        ax2 = canvas.fig.add_subplot(122)
        
        # Chart 1: Predictions comparison
        models = list(self.all_predictions[timeframe].keys())
        x = np.arange(7)
        width = 0.8 / len(models)
        
        for i, model_name in enumerate(models):
            predictions = self.all_predictions[timeframe][model_name][:7]
            offset = (i - len(models)/2) * width + width/2
            ax1.bar(x + offset, predictions, width, label=model_name, alpha=0.8)
        
        ax1.axhline(y=self.predictor.reference_price, color='r', linestyle='--', 
                   linewidth=2, label=f'Current: ${self.predictor.reference_price:.2f}')
        
        if timeframe == '4h':
            labels = [f'{(i+1)*4}h' for i in range(7)]
        elif timeframe == '1d':
            labels = [f'D{i+1}' for i in range(7)]
        else:
            labels = [f'W{i+1}' for i in range(7)]
        
        ax1.set_xlabel('Period', fontweight='bold')
        ax1.set_ylabel('Price ($)', fontweight='bold')
        ax1.set_title(f'{timeframe.upper()} - Price Predictions', fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels)
        ax1.legend(loc='best', fontsize=8)
        ax1.grid(axis='y', alpha=0.3)
        
        # Chart 2: Model metrics
        if timeframe in self.predictor.all_model_results:
            results = self.predictor.all_model_results[timeframe]
            sorted_results = sorted(results.items(), key=lambda x: x[1]['r2'], reverse=True)[:5]
            
            model_names = [name for name, _ in sorted_results]
            r2_scores = [result['r2'] for _, result in sorted_results]
            
            colors = ['gold' if name == self.predictor.best_models.get(timeframe) else 'skyblue' 
                     for name in model_names]
            
            bars = ax2.barh(model_names, r2_scores, color=colors)
            ax2.set_xlabel('R² Score', fontweight='bold')
            ax2.set_title(f'{timeframe.upper()} - Top 5 Models', fontweight='bold')
            ax2.set_xlim([0, 1])
            ax2.grid(axis='x', alpha=0.3)
            
            for i, (bar, score) in enumerate(zip(bars, r2_scores)):
                ax2.text(score, i, f' {score:.4f}', va='center', fontweight='bold')
        
        canvas.fig.tight_layout()
        canvas.draw()
    
    def create_comparison_tab(self):
        """Create comparison tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        
        for timeframe in ['4h', '1d', '1w']:
            if timeframe not in self.predictor.all_model_results:
                continue
            
            group = QGroupBox(f"🏅 {timeframe.upper()} - Model Rankings")
            group_layout = QVBoxLayout()
            
            results = self.predictor.all_model_results[timeframe]
            sorted_models = sorted(results.items(), key=lambda x: x[1]['r2'], reverse=True)
            
            text = QTextEdit()
            text.setReadOnly(True)
            text.setMaximumHeight(200)
            
            content = f"<h3>Top Models for {timeframe.upper()}</h3>"
            content += "<table border='1' cellpadding='5' style='border-collapse: collapse; width: 100%;'>"
            content += "<tr style='background-color: #27ae60; color: white;'>"
            content += "<th>Rank</th><th>Model</th><th>R²</th><th>MAE</th><th>RMSE</th><th>Direction Acc</th></tr>"
            
            for rank, (model_name, result) in enumerate(sorted_models[:5], 1):
                medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else f"{rank}"
                bg_color = "#fff9e6" if rank == 1 else "#f0f0f0" if rank % 2 == 0 else "white"
                content += f"<tr style='background-color: {bg_color};'>"
                content += f"<td style='text-align: center;'><b>{medal}</b></td>"
                content += f"<td><b>{model_name}</b></td>"
                content += f"<td>{result['r2']:.4f}</td>"
                content += f"<td>${result['mae']:.2f}</td>"
                content += f"<td>${result['rmse']:.2f}</td>"
                content += f"<td>{result['direction_accuracy']:.2%}</td>"
                content += f"</tr>"
            
            content += "</table>"
            text.setHtml(content)
            
            group_layout.addWidget(text)
            group.setLayout(group_layout)
            scroll_layout.addWidget(group)
        
        scroll.setWidget(scroll_widget)
        layout.addWidget(scroll)
        
        self.tab_widget.addTab(tab, "🏆 Comparison")
    
    def create_final_predictions_tab(self):
        """Create final predictions tab"""
        tab = QWidget()
        main_layout = QVBoxLayout(tab)
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_widget = QWidget()
        layout = QVBoxLayout(scroll_widget)
        
        # Predictions table
        table_group = QGroupBox("🎯 Final Predictions Summary")
        table_layout = QVBoxLayout()
        
        table = QTableWidget()
        table.setColumnCount(4)
        table.setHorizontalHeaderLabels(['Timeframe', 'Period', 'Predicted Price', 'Change'])
        
        row = 0
        for timeframe in ['4h', '1d', '1w']:
            if timeframe not in self.all_predictions:
                continue
            
            best_model = self.predictor.best_models.get(timeframe)
            if not best_model or best_model not in self.all_predictions[timeframe]:
                continue
            
            predictions = self.all_predictions[timeframe][best_model]
            
            for i in range(min(len(predictions), 7)):
                table.insertRow(row)
                
                if timeframe == '4h':
                    period = f"{(i+1)*4} hours"
                elif timeframe == '1d':
                    period = f"Day {i+1}"
                else:
                    period = f"Week {i+1}"
                
                price = predictions[i]
                change = ((price / self.predictor.reference_price - 1) * 100)
                
                tf_item = QTableWidgetItem(timeframe.upper())
                tf_item.setFont(QFont("Arial", 9, QFont.Weight.Bold))
                table.setItem(row, 0, tf_item)
                
                table.setItem(row, 1, QTableWidgetItem(period))
                
                price_item = QTableWidgetItem(f"${price:.2f}")
                price_item.setFont(QFont("Arial", 9, QFont.Weight.Bold))
                table.setItem(row, 2, price_item)
                
                trend = "📈" if change > 0 else "📉" if change < 0 else "➡️"
                change_item = QTableWidgetItem(f"{change:+.2f}% {trend}")
                change_item.setFont(QFont("Arial", 9, QFont.Weight.Bold))
                
                if change > 5:
                    change_item.setBackground(QColor(0, 255, 0, 100))
                elif change > 0:
                    change_item.setBackground(QColor(144, 238, 144, 100))
                elif change < -5:
                    change_item.setBackground(QColor(255, 0, 0, 100))
                elif change < 0:
                    change_item.setBackground(QColor(255, 182, 193, 100))
                
                table.setItem(row, 3, change_item)
                row += 1
        
        table.resizeColumnsToContents()
        table.setColumnWidth(0, 120)
        table.setColumnWidth(1, 150)
        table.setColumnWidth(2, 150)
        table.setColumnWidth(3, 150)
        table_layout.addWidget(table)
        table_group.setLayout(table_layout)
        layout.addWidget(table_group)
        
        # Best models info
        info_group = QGroupBox("🏆 Best Models Used")
        info_layout = QVBoxLayout()
        
        for timeframe in ['4h', '1d', '1w']:
            if timeframe in self.predictor.best_models:
                best_model = self.predictor.best_models[timeframe]
                result = self.predictor.all_model_results[timeframe][best_model]
                
                label = QLabel(f"• <b>{timeframe.upper()}</b>: {best_model} "
                             f"(R²={result['r2']:.4f}, MAE=${result['mae']:.2f}, "
                             f"Direction Acc={result['direction_accuracy']:.2%})")
                label.setFont(QFont("Arial", 10))
                info_layout.addWidget(label)
        
        info_group.setLayout(info_layout)
        layout.addWidget(info_group)
        
        # Collapsible chart
        chart_container = QWidget()
        chart_container_layout = QVBoxLayout(chart_container)
        chart_container_layout.setContentsMargins(0, 0, 0, 0)
        
        self.chart_toggle_button = QPushButton("▼ Show Visualization Chart")
        self.chart_toggle_button.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                color: white;
                border: none;
                padding: 10px;
                font-size: 13px;
                border-radius: 4px;
                font-weight: bold;
                text-align: left;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
        """)
        self.chart_toggle_button.clicked.connect(self.toggle_final_chart)
        chart_container_layout.addWidget(self.chart_toggle_button)
        
        self.final_chart_group = QGroupBox("📈 Predictions Visualization")
        chart_layout = QVBoxLayout()
        
        self.final_chart_canvas = ChartCanvas(self, width=14, height=6)
        self.plot_final_predictions_chart(self.final_chart_canvas)
        chart_layout.addWidget(self.final_chart_canvas)
        
        self.final_chart_group.setLayout(chart_layout)
        self.final_chart_group.setVisible(False)
        chart_container_layout.addWidget(self.final_chart_group)
        
        layout.addWidget(chart_container)
        
        scroll.setWidget(scroll_widget)
        main_layout.addWidget(scroll)
        
        self.tab_widget.addTab(tab, "🎯 Final Predictions")
    
    def toggle_final_chart(self):
        """Toggle final chart visibility"""
        is_visible = self.final_chart_group.isVisible()
        self.final_chart_group.setVisible(not is_visible)
        
        if is_visible:
            self.chart_toggle_button.setText("▼ Show Visualization Chart")
        else:
            self.chart_toggle_button.setText("▲ Hide Visualization Chart")
    
    def plot_final_predictions_chart(self, canvas):
        """Plot final predictions"""
        canvas.fig.clear()
        
        ax = canvas.fig.add_subplot(111)
        
        colors = {'4h': '#e74c3c', '1d': '#3498db', '1w': '#2ecc71'}
        markers = {'4h': 'o', '1d': 's', '1w': '^'}
        
        for timeframe in ['4h', '1d', '1w']:
            if timeframe not in self.all_predictions:
                continue
            
            best_model = self.predictor.best_models.get(timeframe)
            if not best_model or best_model not in self.all_predictions[timeframe]:
                continue
            
            predictions = self.all_predictions[timeframe][best_model][:7]
            x = list(range(1, len(predictions) + 1))
            
            ax.plot(x, predictions, marker=markers[timeframe], 
                   color=colors[timeframe], linewidth=2, markersize=8,
                   label=f'{timeframe.upper()} ({best_model})', alpha=0.8)
        
        ax.axhline(y=self.predictor.reference_price, color='purple', 
                  linestyle='--', linewidth=2, 
                  label=f'Current Price: ${self.predictor.reference_price:.2f}')
        
        ax.set_xlabel('Period', fontweight='bold', fontsize=11)
        ax.set_ylabel('Price ($)', fontweight='bold', fontsize=11)
        ax.set_title('Final Price Predictions - Best Models', fontweight='bold', fontsize=13)
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        canvas.fig.tight_layout()
        canvas.draw()
    
    def closeEvent(self, event):
        """Handle window close"""
        if self.ws_thread and self.ws_thread.isRunning():
            self.ws_thread.stop()
            self.ws_thread.wait()
        
        event.accept()


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    gui = ETHPredictorGUI()
    gui.show()
    
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
