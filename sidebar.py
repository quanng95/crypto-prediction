import streamlit as st
import time

def format_price_sidebar(price):
    """Format price based on its value"""
    if price >= 1000:
        return f"${price:,.2f}"
    elif price >= 100:
        return f"${price:,.2f}"
    elif price >= 1:
        return f"${price:,.4f}"
    elif price >= 0.01:
        return f"${price:,.6f}"
    elif price >= 0.0001:
        return f"${price:,.8f}"
    else:
        return f"${price:.11f}".rstrip('0').rstrip('.')

def get_ticker_data(ws_handler, symbol):
    """Get ticker data from WebSocket"""
    try:
        data = ws_handler.get_price(symbol)
        if data and (time.time() - data['timestamp']) < 5:
            return data
        return None
    except:
        return None

@st.fragment(run_every="0.5s")
def render_sidebar_content(ws_handler, symbols, selected_symbol):
    """Render sidebar content with realtime updates"""
    
    st.markdown("""
    <div style="text-align: center; margin-bottom: 20px;">
        <h2 style="color: #ffffff; margin: 0;">📊 Markets</h2>
        <p style="color: #7f8c8d; font-size: 14px; margin: 5px 0;">Live Prices</p>
    </div>
    """, unsafe_allow_html=True)
    
    for symbol in symbols:
        ticker_data = get_ticker_data(ws_handler, symbol)
        
        if ticker_data:
            price = ticker_data['price']
            change_pct = ticker_data['change_percent']
            is_up = change_pct >= 0
            
            formatted_price = format_price_sidebar(price)
            
            # Highlight selected symbol
            is_selected = (symbol == selected_symbol)
            border_color = "#667eea" if is_selected else "#3d3d3d"
            bg_color = "#3d3d3d" if is_selected else "#2d2d2d"
            
            st.markdown(f"""
            <div style="
                background-color: {bg_color};
                padding: 12px;
                border-radius: 8px;
                margin-bottom: 10px;
                border: 2px solid {border_color};
                cursor: pointer;
                transition: all 0.3s ease;
            ">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <div style="color: #ffffff; font-weight: bold; font-size: 14px;">{symbol.replace('USDT', '')}</div>
                        <div style="color: {'#27ae60' if is_up else '#e74c3c'}; font-size: 12px; margin-top: 2px;">
                            {'▲' if is_up else '▼'} {abs(change_pct):.2f}%
                        </div>
                    </div>
                    <div style="text-align: right;">
                        <div style="color: {'#27ae60' if is_up else '#e74c3c'}; font-weight: bold; font-size: 16px;">
                            {formatted_price}
                        </div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Button to select symbol
            if st.button(
                f"📈 Analyze {symbol.replace('USDT', '')}",
                key=f"sidebar_analyze_{symbol}",
                use_container_width=True,
                type="secondary" if not is_selected else "primary"
            ):
                st.session_state.selected_symbol = symbol
                st.session_state.trigger_analysis = True
                st.rerun()
        else:
            # Fallback UI when no data
            st.markdown(f"""
            <div style="
                background-color: #2d2d2d;
                padding: 12px;
                border-radius: 8px;
                margin-bottom: 10px;
                border: 1px solid #3d3d3d;
            ">
                <div style="color: #7f8c8d; font-size: 14px; text-align: center;">
                    {symbol} - Loading...
                </div>
            </div>
            """, unsafe_allow_html=True)

def render_custom_sidebar(ws_handler, symbols, selected_symbol):
    """Main sidebar render function"""
    
    # Custom CSS for sidebar
    st.markdown("""
    <style>
        /* Hide default Streamlit sidebar */
        [data-testid="stSidebar"] {
            display: none;
        }
        
        /* Custom sidebar container */
        .custom-sidebar {
            position: fixed;
            left: -300px;
            top: 0;
            width: 300px;
            height: 100vh;
            background: linear-gradient(180deg, #1e1e1e 0%, #2d2d2d 100%);
            box-shadow: 2px 0 15px rgba(0, 0, 0, 0.5);
            transition: left 0.3s ease;
            z-index: 999;
            overflow-y: auto;
            padding: 20px;
        }
        
        .custom-sidebar.open {
            left: 0;
        }
        
        /* Toggle button */
        .sidebar-toggle {
            position: fixed;
            left: 0;
            top: 50%;
            transform: translateY(-50%);
            width: 40px;
            height: 80px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 0 10px 10px 0;
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            z-index: 1000;
            transition: all 0.3s ease;
            box-shadow: 2px 0 10px rgba(102, 126, 234, 0.4);
        }
        
        .sidebar-toggle:hover {
            width: 45px;
            box-shadow: 2px 0 15px rgba(102, 126, 234, 0.6);
        }
        
        .sidebar-toggle.open {
            left: 300px;
        }
        
        .toggle-icon {
            color: white;
            font-size: 24px;
            font-weight: bold;
        }
        
        /* Scrollbar styling */
        .custom-sidebar::-webkit-scrollbar {
            width: 8px;
        }
        
        .custom-sidebar::-webkit-scrollbar-track {
            background: #1e1e1e;
        }
        
        .custom-sidebar::-webkit-scrollbar-thumb {
            background: #667eea;
            border-radius: 4px;
        }
        
        .custom-sidebar::-webkit-scrollbar-thumb:hover {
            background: #764ba2;
        }
        
        /* Adjust main content when sidebar is open */
        .main.sidebar-open {
            margin-left: 300px;
            transition: margin-left 0.3s ease;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Initialize sidebar state
    if 'sidebar_open' not in st.session_state:
        st.session_state.sidebar_open = False
    
    # Toggle button HTML
    toggle_class = "open" if st.session_state.sidebar_open else ""
    toggle_icon = "›" if st.session_state.sidebar_open else "‹"
    
    st.markdown(f"""
    <div class="sidebar-toggle {toggle_class}" onclick="toggleSidebar()">
        <span class="toggle-icon">{toggle_icon}</span>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar container
    sidebar_class = "open" if st.session_state.sidebar_open else ""
    
    # JavaScript for toggle functionality
    st.markdown("""
    <script>
        function toggleSidebar() {
            const sidebar = document.querySelector('.custom-sidebar');
            const toggle = document.querySelector('.sidebar-toggle');
            const main = document.querySelector('.main');
            
            if (sidebar && toggle && main) {
                sidebar.classList.toggle('open');
                toggle.classList.toggle('open');
                main.classList.toggle('sidebar-open');
                
                // Update icon
                const icon = toggle.querySelector('.toggle-icon');
                if (sidebar.classList.contains('open')) {
                    icon.textContent = '›';
                } else {
                    icon.textContent = '‹';
                }
            }
        }
        
        // Close sidebar when clicking outside
        document.addEventListener('click', function(event) {
            const sidebar = document.querySelector('.custom-sidebar');
            const toggle = document.querySelector('.sidebar-toggle');
            
            if (sidebar && toggle && sidebar.classList.contains('open')) {
                if (!sidebar.contains(event.target) && !toggle.contains(event.target)) {
                    toggleSidebar();
                }
            }
        });
    </script>
    """, unsafe_allow_html=True)
    
    # Render sidebar content in a container
    sidebar_container = st.container()
    
    with sidebar_container:
        st.markdown(f'<div class="custom-sidebar {sidebar_class}" id="customSidebar">', unsafe_allow_html=True)
        
        # Render content with auto-refresh
        render_sidebar_content(ws_handler, symbols, selected_symbol)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Button to toggle sidebar (fallback for mobile/touch devices)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        if st.button("☰ Markets" if not st.session_state.sidebar_open else "✕ Close", 
                     key="toggle_sidebar_btn",
                     type="secondary"):
            st.session_state.sidebar_open = not st.session_state.sidebar_open
            st.rerun()
