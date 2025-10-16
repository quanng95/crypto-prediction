import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QTableWidget, QTableWidgetItem, QPushButton,
                             QLabel, QTabWidget, QTextEdit, QComboBox, QGroupBox,
                             QProgressBar, QMessageBox, QSplitter, QScrollArea)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QFont, QColor, QTextCursor
import pandas as pd
import numpy as np
from eth import AdvancedETHPredictor
import traceback
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

class PriceUpdateThread(QThread):
    """Thread for updating real-time price"""
    price_updated = pyqtSignal(dict)
    
    def __init__(self, symbol):
        super().__init__()
        self.symbol = symbol
        self.running = True
        self.predictor = AdvancedETHPredictor()
    
    def run(self):
        while self.running:
            try:
                ticker_data = self.predictor.get_24h_ticker(self.symbol)
                if ticker_data:
                    self.price_updated.emit(ticker_data)
            except Exception as e:
                print(f"Price update error: {e}")
            
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

class ChartCanvas(FigureCanvas):
    """Canvas for matplotlib charts"""
    def __init__(self, parent=None, width=12, height=6, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        super().__init__(self.fig)
        self.setParent(parent)

class ETHPredictorGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.predictor = None
        self.all_predictions = None
        self.price_thread = None
        self.all_data = None
        self.init_ui()
        
    def init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle("🔮 Crypto Prediction")
        self.setGeometry(50, 50, 1800, 1000)
        
        # Main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        main_layout.setSpacing(5)
        main_layout.setContentsMargins(10, 10, 10, 10)
        
        # Real-time price display (TOP)
        self.price_widget = self.create_realtime_price_widget()
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
        
        # Console log (collapsible)
        self.create_console_log()
        main_layout.addWidget(self.console_container)
        
        # Tab widget for results
        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)
        
        # Set style
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f5f5f5;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #cccccc;
                border-radius: 5px;
                margin-top: 5px;
                padding-top: 5px;
                background-color: white;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 8px 16px;
                font-size: 13px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
            QTableWidget {
                background-color: white;
                alternate-background-color: #f9f9f9;
                gridline-color: #e0e0e0;
            }
            QHeaderView::section {
                background-color: #4CAF50;
                color: white;
                padding: 5px;
                font-weight: bold;
                border: none;
            }
            QTabWidget::pane {
                border: 1px solid #cccccc;
                background-color: white;
            }
            QTabBar::tab {
                background-color: #e0e0e0;
                padding: 8px 16px;
                margin-right: 2px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }
            QTabBar::tab:selected {
                background-color: #4CAF50;
                color: white;
            }
        """)
    
    def create_console_log(self):
        """Create console log widget"""
        self.console_container = QWidget()
        console_layout = QVBoxLayout(self.console_container)
        console_layout.setContentsMargins(0, 0, 0, 0)
        
        # Toggle button
        self.console_toggle_button = QPushButton("▼ Show Console Log")
        self.console_toggle_button.setStyleSheet("""
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
        self.console_toggle_button.clicked.connect(self.toggle_console)
        console_layout.addWidget(self.console_toggle_button)
        
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
        console_layout.addWidget(self.console_log)
        
        self.console_container.setMaximumHeight(250)
    
    def toggle_console(self):
        """Toggle console visibility"""
        is_visible = self.console_log.isVisible()
        self.console_log.setVisible(not is_visible)
        
        if is_visible:
            self.console_toggle_button.setText("▼ Show Console Log")
        else:
            self.console_toggle_button.setText("▲ Hide Console Log")
    
    def append_log(self, message):
        """Append message to console log"""
        self.console_log.append(message)
        # Auto scroll to bottom
        self.console_log.moveCursor(QTextCursor.MoveOperation.End)
    
    def clear_console(self):
        """Clear console log"""
        self.console_log.clear()
    
    def create_realtime_price_widget(self):
        """Create real-time price display widget"""
        widget = QGroupBox()
        layout = QHBoxLayout()
        layout.setSpacing(20)
        
        # Symbol label
        self.symbol_display = QLabel("ETHUSDT")
        self.symbol_display.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        self.symbol_display.setStyleSheet("color: #2c3e50;")
        
        # Current price
        self.current_price_label = QLabel("$0.00")
        self.current_price_label.setFont(QFont("Arial", 28, QFont.Weight.Bold))
        self.current_price_label.setStyleSheet("color: #27ae60;")
        
        # 24h change
        self.change_label = QLabel("--")
        self.change_label.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        
        # 24h high
        self.high_label = QLabel("High: $0.00")
        self.high_label.setFont(QFont("Arial", 11))
        self.high_label.setStyleSheet("color: #555;")
        
        # 24h low
        self.low_label = QLabel("Low: $0.00")
        self.low_label.setFont(QFont("Arial", 11))
        self.low_label.setStyleSheet("color: #555;")
        
        # 24h volume
        self.volume_label = QLabel("Volume: 0")
        self.volume_label.setFont(QFont("Arial", 11))
        self.volume_label.setStyleSheet("color: #555;")
        
        # Last update time
        self.update_time_label = QLabel("Last update: --")
        self.update_time_label.setFont(QFont("Arial", 9))
        self.update_time_label.setStyleSheet("color: #888;")
        
        # Layout
        layout.addWidget(self.symbol_display)
        layout.addWidget(self.current_price_label)
        layout.addWidget(self.change_label)
        layout.addSpacing(20)
        layout.addWidget(self.high_label)
        layout.addWidget(self.low_label)
        layout.addWidget(self.volume_label)
        layout.addStretch()
        layout.addWidget(self.update_time_label)
        
        widget.setLayout(layout)
        widget.setMaximumHeight(100)
        
        return widget
    
    def update_realtime_price(self, ticker_data):
        """Update real-time price display"""
        price = ticker_data['price']
        change = ticker_data['change']
        change_percent = ticker_data['change_percent']
        high = ticker_data['high']
        low = ticker_data['low']
        volume = ticker_data['volume']
        
        # Update price
        self.current_price_label.setText(f"${price:,.2f}")
        
        # Update change with color
        if change >= 0:
            self.current_price_label.setStyleSheet("color: #27ae60;")  # Green
            self.change_label.setText(f"📈 +${change:.2f} (+{change_percent:.2f}%)")
            self.change_label.setStyleSheet("color: #27ae60; font-weight: bold;")
        else:
            self.current_price_label.setStyleSheet("color: #e74c3c;")  # Red
            self.change_label.setText(f"📉 ${change:.2f} ({change_percent:.2f}%)")
            self.change_label.setStyleSheet("color: #e74c3c; font-weight: bold;")
        
        # Update other info
        self.high_label.setText(f"High: ${high:,.2f}")
        self.low_label.setText(f"Low: ${low:,.2f}")
        self.volume_label.setText(f"Volume: {volume:,.0f}")
        
        # Update time
        from datetime import datetime
        self.update_time_label.setText(f"Last update: {datetime.now().strftime('%H:%M:%S')}")
    
    def create_control_panel(self):
        """Create control panel"""
        panel = QGroupBox("Control Panel")
        layout = QHBoxLayout()
        
        # Symbol input
        symbol_label = QLabel("Symbol:")
        symbol_label.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        self.symbol_input = QComboBox()
        self.symbol_input.addItems(["ETHUSDT", "BTCUSDT", "PAXGUSDT", "BNBUSDT", "SOLUSDT", "ADAUSDT", "DOGEUSDT", "ARBUSDT"])
        self.symbol_input.currentTextChanged.connect(self.on_symbol_changed)
        
        # Timezone input
        timezone_label = QLabel("Timezone:")
        timezone_label.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        self.timezone_input = QComboBox()
        self.timezone_input.addItems([
            "Asia/Ho_Chi_Minh",
            "America/New_York",
            "Europe/London",
            "Asia/Tokyo",
            "Asia/Shanghai"
        ])
        
        # Run button
        self.run_button = QPushButton("🚀 Run Analysis")
        self.run_button.clicked.connect(self.run_analysis)
        
        # Start price monitoring automatically
        self.start_price_monitoring()
        
        layout.addWidget(symbol_label)
        layout.addWidget(self.symbol_input)
        layout.addSpacing(20)
        layout.addWidget(timezone_label)
        layout.addWidget(self.timezone_input)
        layout.addSpacing(20)
        layout.addWidget(self.run_button)
        layout.addStretch()
        
        panel.setLayout(layout)
        panel.setMaximumHeight(80)
        return panel
    
    def on_symbol_changed(self, symbol):
        """Handle symbol change"""
        self.symbol_display.setText(symbol)
        # Restart price monitoring with new symbol
        self.start_price_monitoring()
    
    def start_price_monitoring(self):
        """Start real-time price monitoring"""
        # Stop existing thread if any
        if self.price_thread and self.price_thread.isRunning():
            self.price_thread.stop()
            self.price_thread.wait()
        
        # Start new thread
        symbol = self.symbol_input.currentText()
        self.price_thread = PriceUpdateThread(symbol)
        self.price_thread.price_updated.connect(self.update_realtime_price)
        self.price_thread.start()
    
    def run_analysis(self):
        """Start analysis in background thread"""
        self.run_button.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate
        self.status_label.setText("Running analysis...")
        self.status_label.setStyleSheet("color: orange; font-size: 11px;")
        
        # Clear previous results and console
        self.tab_widget.clear()
        self.clear_console()
        
        # Show console automatically
        self.console_log.setVisible(True)
        self.console_toggle_button.setText("▲ Hide Console Log")
        
        # Start analysis thread
        symbol = self.symbol_input.currentText()
        timezone = self.timezone_input.currentText()
        
        self.thread = AnalysisThread(symbol, timezone)
        self.thread.finished.connect(self.on_analysis_finished)
        self.thread.progress.connect(self.on_progress_update)
        self.thread.error.connect(self.on_analysis_error)
        self.thread.log_message.connect(self.append_log)
        self.thread.start()
    
    def on_progress_update(self, message):
        """Update progress message"""
        self.status_label.setText(message)
    
    def on_analysis_finished(self, predictor, all_predictions):
        """Handle analysis completion"""
        self.predictor = predictor
        self.all_predictions = all_predictions
        
        self.progress_bar.setVisible(False)
        self.run_button.setEnabled(True)
        self.status_label.setText("✅ Analysis complete!")
        self.status_label.setStyleSheet("color: green; font-size: 12px;")
        
        # Hide console after completion
        QTimer.singleShot(2000, self.hide_console_after_completion)
        
        # Display results
        self.display_results()
    
    def hide_console_after_completion(self):
        """Hide console after analysis completion"""
        self.console_log.setVisible(False)
        self.console_toggle_button.setText("▼ Show Console Log")
    
    def on_analysis_error(self, error_msg):
        """Handle analysis error"""
        self.progress_bar.setVisible(False)
        self.run_button.setEnabled(True)
        self.status_label.setText("❌ Analysis failed")
        self.status_label.setStyleSheet("color: red; font-size: 11px;")
        
        self.append_log(f"\n❌ ERROR: {error_msg}")
        QMessageBox.critical(self, "Error", error_msg)
    
    def display_results(self):
        """Display all results in tabs"""
        if not self.predictor or not self.all_predictions:
            return
        
        # Tab 1: Summary with charts
        self.create_summary_tab()
        
        # Tab 2-4: Individual timeframe results with charts
        for timeframe in ['4h', '1d', '1w']:
            if timeframe in self.all_predictions:
                self.create_timeframe_tab(timeframe)
        
        # Tab 5: Model Comparison
        self.create_comparison_tab()
        
        # Tab 6: Final Predictions (Best Models)
        self.create_final_predictions_tab()
    
    def create_summary_tab(self):
        """Create summary tab with charts"""
        tab = QWidget()
        main_layout = QVBoxLayout(tab)
        
        # Create scroll area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_widget = QWidget()
        layout = QVBoxLayout(scroll_widget)
        
        # # Reference info
        # info_group = QGroupBox("📊 Analysis Information")
        # info_layout = QVBoxLayout()
        
        # ref_price = QLabel(f"💰 Reference Price: ${self.predictor.reference_price:.2f}")
        # ref_price.setFont(QFont("Arial", 13, QFont.Weight.Bold))
        
        # ref_time = QLabel(f"⏰ Reference Time: {self.predictor.reference_time_local}")
        # ref_time.setFont(QFont("Arial", 11))
        
        # info_layout.addWidget(ref_price)
        # info_layout.addWidget(ref_time)
        # info_group.setLayout(info_layout)
        # layout.addWidget(info_group)
        
        # Model performance summary table
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
                    
                    # Color code R² score
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
        
        # Create chart canvas
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
        
        # Create subplots
        gs = canvas.fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        
        # Chart 1: R² Score Comparison
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
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom', fontweight='bold')
        
        # Chart 2: MAE Comparison
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
                    f'${height:.2f}',
                    ha='center', va='bottom', fontweight='bold')
        
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
                    f'{height:.1f}%',
                    ha='center', va='bottom', fontweight='bold')
        
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
        """Create tab for specific timeframe with charts"""
        tab = QWidget()
        main_layout = QVBoxLayout(tab)
        
        # Create scroll area
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
        pred_group = QGroupBox(f"📈 {timeframe.upper()} - Predictions by Model (First 7 Periods)")
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
        """Plot charts for specific timeframe"""
        canvas.fig.clear()
        
        if timeframe not in self.all_predictions:
            return
        
        # Create subplots
        ax1 = canvas.fig.add_subplot(121)
        ax2 = canvas.fig.add_subplot(122)
        
        # Chart 1: Predictions comparison (first 7 periods)
        models = list(self.all_predictions[timeframe].keys())
        x = np.arange(7)
        width = 0.8 / len(models)
        
        for i, model_name in enumerate(models):
            predictions = self.all_predictions[timeframe][model_name][:7]
            offset = (i - len(models)/2) * width + width/2
            ax1.bar(x + offset, predictions, width, label=model_name, alpha=0.8)
        
        # Add reference line
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
        
        # Chart 2: Model metrics comparison
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
            
            # Add value labels
            for i, (bar, score) in enumerate(zip(bars, r2_scores)):
                ax2.text(score, i, f' {score:.4f}', va='center', fontweight='bold')
        
        canvas.fig.tight_layout()
        canvas.draw()
    
    def create_comparison_tab(self):
        """Create model comparison tab"""
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
            content += "<tr style='background-color: #4CAF50; color: white;'>"
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
        """Create final predictions tab with collapsible chart"""
        tab = QWidget()
        main_layout = QVBoxLayout(tab)
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_widget = QWidget()
        layout = QVBoxLayout(scroll_widget)
        
        # Predictions table (no header - moved to top)
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
        
        # Collapsible chart section
        chart_container = QWidget()
        chart_container_layout = QVBoxLayout(chart_container)
        chart_container_layout.setContentsMargins(0, 0, 0, 0)
        
        # Toggle button for chart
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
        
        # Chart group (initially hidden)
        self.final_chart_group = QGroupBox("📈 Predictions Visualization")
        chart_layout = QVBoxLayout()
        
        self.final_chart_canvas = ChartCanvas(self, width=14, height=6)
        self.plot_final_predictions_chart(self.final_chart_canvas)
        chart_layout.addWidget(self.final_chart_canvas)
        
        self.final_chart_group.setLayout(chart_layout)
        self.final_chart_group.setVisible(False)  # Hidden by default
        chart_container_layout.addWidget(self.final_chart_group)
        
        layout.addWidget(chart_container)
        
        scroll.setWidget(scroll_widget)
        main_layout.addWidget(scroll)
        
        self.tab_widget.addTab(tab, "🎯 Final Predictions")
    
    def toggle_final_chart(self):
        """Toggle visibility of final predictions chart"""
        is_visible = self.final_chart_group.isVisible()
        self.final_chart_group.setVisible(not is_visible)
        
        if is_visible:
            self.chart_toggle_button.setText("▼ Show Visualization Chart")
        else:
            self.chart_toggle_button.setText("▲ Hide Visualization Chart")
    
    def plot_final_predictions_chart(self, canvas):
        """Plot final predictions chart"""
        canvas.fig.clear()
        
        ax = canvas.fig.add_subplot(111)
        
        # Plot predictions for each timeframe
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
        
        # Add reference line
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
        """Handle window close event"""
        # Stop price monitoring thread
        if self.price_thread and self.price_thread.isRunning():
            self.price_thread.stop()
            self.price_thread.wait()
        event.accept()

def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    gui = ETHPredictorGUI()
    gui.show()
    
    sys.exit(app.exec())

if __name__ == '__main__':
    main()
