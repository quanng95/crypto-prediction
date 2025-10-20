import streamlit as st
import time

def format_price(price):
    """Format price based on its value"""
    if price >= 1000:
        return f"{price:,.2f}"
    elif price >= 100:
        return f"{price:,.2f}"
    elif price >= 1:
        return f"{price:,.4f}"
    elif price >= 0.01:
        return f"{price:,.6f}"
    elif price >= 0.0001:
        return f"{price:,.8f}"
    else:
        return f"{price:.10f}".rstrip('0').rstrip('.')

def format_change(change):
    """Format change percentage"""
    return f"{change:+.2f}%"

def render_sidebar():
    """Render expandable sidebar with real-time ticker"""
    
    # Initialize sidebar state
    if 'sidebar_expanded' not in st.session_state:
        st.session_state.sidebar_expanded = False
    
    # Sidebar CSS
    st.markdown("""
    <style>
        /* Sidebar Container */
        .sidebar-container {
            position: fixed;
            left: 0;
            top: 0;
            height: 100vh;
            width: 280px;
            background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
            transform: translateX(-280px);
            transition: transform 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            z-index: 9999;
            box-shadow: 4px 0 20px rgba(0, 0, 0, 0.5);
            overflow-y: auto;
            overflow-x: hidden;
        }
        
        .sidebar-container.expanded {
            transform: translateX(0);
        }
        
        /* Scrollbar Styling */
        .sidebar-container::-webkit-scrollbar {
            width: 6px;
        }
        
        .sidebar-container::-webkit-scrollbar-track {
            background: #1a1a2e;
        }
        
        .sidebar-container::-webkit-scrollbar-thumb {
            background: #667eea;
            border-radius: 3px;
        }
        
        .sidebar-container::-webkit-scrollbar-thumb:hover {
            background: #764ba2;
        }
        
        /* Toggle Button at Edge */
        .sidebar-toggle-edge {
            position: fixed;
            left: 0;
            top: 50%;
            transform: translateY(-50%);
            width: 30px;
            height: 60px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 0 8px 8px 0;
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            z-index: 9998;
            transition: all 0.3s ease;
            box-shadow: 2px 0 10px rgba(102, 126, 234, 0.4);
        }
        
        .sidebar-toggle-edge:hover {
            width: 35px;
            box-shadow: 2px 0 15px rgba(102, 126, 234, 0.6);
        }
        
        .sidebar-toggle-edge.hidden {
            transform: translateX(-50px);
            opacity: 0;
        }
        
        .sidebar-toggle-edge span {
            color: white;
            font-size: 18px;
            font-weight: bold;
        }
        
        /* Toggle Button Inside Sidebar */
        .sidebar-toggle-inside {
            position: absolute;
            right: 10px;
            top: 10px;
            width: 35px;
            height: 35px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 8px;
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            transition: all 0.3s ease;
            border: 1px solid rgba(255, 255, 255, 0.2);
        }
        
        .sidebar-toggle-inside:hover {
            background: rgba(255, 255, 255, 0.2);
            transform: scale(1.1);
        }
        
        .sidebar-toggle-inside span {
            color: white;
            font-size: 18px;
            font-weight: bold;
        }
        
        /* Sidebar Header */
        .sidebar-header {
            padding: 20px 15px 15px 15px;
            border-bottom: 2px solid rgba(102, 126, 234, 0.3);
            margin-bottom: 10px;
        }
        
        .sidebar-title {
            color: #ffffff;
            font-size: 20px;
            font-weight: bold;
            margin: 0;
            margin-top: 25px;
            text-align: center;
        }
        
        /* Ticker Item */
        .sidebar-ticker {
            padding: 12px 15px;
            margin: 5px 10px;
            background: rgba(255, 255, 255, 0.05);
            border-radius: 8px;
            border-left: 3px solid transparent;
            transition: all 0.2s ease;
            cursor: pointer;
        }
        
        .sidebar-ticker:hover {
            background: rgba(255, 255, 255, 0.1);
            transform: translateX(5px);
        }
        
        .sidebar-ticker.selected {
            background: rgba(102, 126, 234, 0.2);
            border-left-color: #667eea;
        }
        
        .ticker-symbol {
            color: #ffffff;
            font-size: 14px;
            font-weight: bold;
            margin-bottom: 4px;
            display: flex;
            align-items: center;
            gap: 5px;
        }
        
        .ticker-price {
            font-size: 16px;
            font-weight: bold;
            margin-bottom: 2px;
        }
        
        .ticker-price.up {
            color: #27ae60;
        }
        
        .ticker-price.down {
            color: #e74c3c;
        }
        
        .ticker-change {
            font-size: 12px;
        }
        
        .ticker-change.up {
            color: #27ae60;
        }
        
        .ticker-change.down {
            color: #e74c3c;
        }
        
        /* Backdrop */
        .sidebar-backdrop {
            position: fixed;
            top: 0;
            left: 0;
            width: 100vw;
            height: 100vh;
            background: rgba(0, 0, 0, 0.5);
            backdrop-filter: blur(2px);
            z-index: 9997;
            opacity: 0;
            pointer-events: none;
            transition: opacity 0.3s ease;
        }
        
        .sidebar-backdrop.visible {
            opacity: 1;
            pointer-events: auto;
        }
        
        /* Status Indicator */
        .status-indicator {
            display: inline-block;
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: #27ae60;
            animation: pulse 2s infinite;
        }
        
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Toggle button at edge (when collapsed)
    edge_button_class = "hidden" if st.session_state.sidebar_expanded else ""
    st.markdown(f"""
    <div class="sidebar-toggle-edge {edge_button_class}" onclick="window.parent.postMessage({{type: 'streamlit:setComponentValue', key: 'toggle_sidebar', value: true}}, '*')">
        <span>◀</span>
    </div>
    """, unsafe_allow_html=True)
    
    # Backdrop (when expanded)
    backdrop_class = "visible" if st.session_state.sidebar_expanded else ""
    st.markdown(f"""
    <div class="sidebar-backdrop {backdrop_class}" onclick="window.parent.postMessage({{type: 'streamlit:setComponentValue', key: 'toggle_sidebar', value: true}}, '*')"></div>
    """, unsafe_allow_html=True)
    
    # Sidebar container
    sidebar_class = "expanded" if st.session_state.sidebar_expanded else ""
    
    # Create sidebar content
    sidebar_content = f"""
    <div class="sidebar-container {sidebar_class}">
        <div class="sidebar-toggle-inside" onclick="window.parent.postMessage({{type: 'streamlit:setComponentValue', key: 'toggle_sidebar', value: true}}, '*')">
            <span>▶</span>
        </div>
        
        <div class="sidebar-header">
            <h2 class="sidebar-title">
                <span class="status-indicator"></span>
                Market Watch
            </h2>
        </div>
        
        <div class="sidebar-tickers">
    """
    
    # Get symbols from session state
    if 'ws_handler' in st.session_state:
        symbols = ["ETHUSDT", "BTCUSDT", "PAXGUSDT", "BNBUSDT", "SOLUSDT", 
                   "DOGEUSDT", "KAITOUSDT", "ADAUSDT"]
        
        for symbol in symbols:
            ticker_data = st.session_state.ws_handler.get_price(symbol)
            
            if ticker_data:
                price = ticker_data['price']
                change_pct = ticker_data['change_percent']
                is_up = change_pct >= 0
                
                formatted_price = format_price(price)
                formatted_change = format_change(change_pct)
                
                price_class = "up" if is_up else "down"
                arrow = "▲" if is_up else "▼"
                
                selected_class = "selected" if st.session_state.get('selected_symbol') == symbol else ""
                
                sidebar_content += f"""
                <div class="sidebar-ticker {selected_class}" onclick="window.parent.postMessage({{type: 'streamlit:setComponentValue', key: 'select_symbol', value: '{symbol}'}}, '*')">
                    <div class="ticker-symbol">{symbol}</div>
                    <div class="ticker-price {price_class}">${formatted_price}</div>
                    <div class="ticker-change {price_class}">{arrow} {formatted_change}</div>
                </div>
                """
            else:
                sidebar_content += f"""
                <div class="sidebar-ticker">
                    <div class="ticker-symbol">{symbol}</div>
                    <div class="ticker-price">Loading...</div>
                </div>
                """
    
    sidebar_content += """
        </div>
    </div>
    """
    
    st.markdown(sidebar_content, unsafe_allow_html=True)
    
    # Handle toggle button click
    if st.button("", key="sidebar_toggle_handler", type="primary"):
        st.session_state.sidebar_expanded = not st.session_state.sidebar_expanded
        st.rerun()
    
    # Auto-refresh fragment for real-time updates
    @st.fragment(run_every="0.5s")
    def update_sidebar_prices():
        """Update sidebar prices in real-time"""
        if st.session_state.sidebar_expanded and 'ws_handler' in st.session_state:
            # Trigger re-render by updating a dummy state
            if 'sidebar_update_count' not in st.session_state:
                st.session_state.sidebar_update_count = 0
            st.session_state.sidebar_update_count += 1
    
    if st.session_state.sidebar_expanded:
        update_sidebar_prices()