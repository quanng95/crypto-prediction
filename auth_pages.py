import streamlit as st
from database_postgres import Database

def render_login_page():
    """Render login page with remember me"""
    st.markdown("""
    <div class="header-container">
        <h1 class="header-title">🔐 Sign In</h1>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("### Welcome Back!")
        
        with st.form("login_form"):
            username = st.text_input("Username", placeholder="Enter your username")
            password = st.text_input("Password", type="password", placeholder="Enter your password")
            
            # Remember me checkbox
            remember_me = st.checkbox("🔒 Remember me for 30 days", value=True)
            
            col_btn1, col_btn2 = st.columns(2)
            
            with col_btn1:
                submit = st.form_submit_button("🚀 Sign In", use_container_width=True, type="primary")
            
            with col_btn2:
                signup_redirect = st.form_submit_button("📝 Sign Up", use_container_width=True)
            
            if submit:
                if not username or not password:
                    st.error("❌ Please fill in all fields")
                else:
                    db = Database()
                    user = db.authenticate_user(username, password)
                    
                    if user:
                        # Create session token
                        session_token = db.create_session(user['id'], remember_me)
                        
                        if session_token:
                            # Save to session state
                            st.session_state.authenticated = True
                            st.session_state.user = user
                            st.session_state.session_token = session_token
                            
                            # Load user's symbols
                            symbols = db.get_user_symbols(user['id'])
                            if symbols:
                                st.session_state.SYMBOLS = symbols
                            
                            st.success(f"✅ Welcome back, {user['username']}!")
                            st.balloons()
                            
                            st.session_state.page = "home"
                            st.rerun()
                        else:
                            st.error("❌ Error creating session")
                    else:
                        st.error("❌ Invalid username or password")
            
            if signup_redirect:
                st.session_state.page = "signup"
                st.rerun()

def render_user_menu():
    """Render user menu with auto-login check"""
    
    # Check for existing session on page load
    if not st.session_state.get('authenticated', False):
        if 'session_token' in st.session_state:
            db = Database()
            user = db.validate_session(st.session_state.session_token)
            
            if user:
                st.session_state.authenticated = True
                st.session_state.user = user
                
                # Load user's symbols
                symbols = db.get_user_symbols(user['id'])
                if symbols:
                    st.session_state.SYMBOLS = symbols
    
    # Rest of the user menu code...
    if st.session_state.get('authenticated', False):
        user = st.session_state.user
        
        col1, col2, col3 = st.columns([6, 1, 1])
        
        with col2:
            st.markdown(f"""
            <div style="text-align: right; padding: 10px;">
                <span style="color: #3498db; font-weight: bold;">👤 {user['username']}</span>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            if st.button("🚪 Logout", type="secondary", use_container_width=True):
                db = Database()
                
                # Save symbols before logout
                db.save_user_symbols(user['id'], st.session_state.SYMBOLS)
                
                # Delete session
                if 'session_token' in st.session_state:
                    db.delete_session(st.session_state.session_token)
                    del st.session_state.session_token
                
                # Clear session
                st.session_state.authenticated = False
                st.session_state.user = None
                
                # Reset to default symbols
                st.session_state.SYMBOLS = [
                    "ETHUSDT", "BTCUSDT", "PAXGUSDT", "BNBUSDT", "SOLUSDT",
                    "LINKUSDT", "PEPEUSDT", "XRPUSDT", "DOGEUSDT", "KAITOUSDT", "ADAUSDT"
                ]
                
                st.session_state.page = "home"
                st.success("✅ Logged out successfully!")
                st.rerun()
    else:
        col1, col2 = st.columns([7, 1])
        
        with col2:
            if st.button("🔐 Sign In", type="primary", use_container_width=True):
                st.session_state.page = "login"
                st.rerun()
