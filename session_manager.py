import streamlit as st
from database_postgres import Database
import streamlit.components.v1 as components

class SessionManager:
    
    @staticmethod
    def save_session_to_browser(session_token: str):
        """Save session to browser localStorage"""
        components.html(
            f"""
            <script>
                localStorage.setItem('session_token', '{session_token}');
                console.log('Session saved to localStorage');
            </script>
            """,
            height=0,
        )
    
    @staticmethod
    def get_session_from_browser():
        """Get session from browser localStorage"""
        session_token = components.html(
            """
            <script>
                const token = localStorage.getItem('session_token');
                if (token) {
                    window.parent.postMessage({type: 'streamlit:setComponentValue', value: token}, '*');
                } else {
                    window.parent.postMessage({type: 'streamlit:setComponentValue', value: null}, '*');
                }
            </script>
            """,
            height=0,
        )
        return session_token
    
    @staticmethod
    def clear_session_from_browser():
        """Clear session from browser localStorage"""
        components.html(
            """
            <script>
                localStorage.removeItem('session_token');
                console.log('Session cleared from localStorage');
            </script>
            """,
            height=0,
        )
    
    @staticmethod
    def auto_login():
        """Auto login from session"""
        # Skip if already authenticated
        if st.session_state.get('authenticated', False):
            return True
        
        # Check if session_token exists in session_state (from previous page load)
        if 'session_token' in st.session_state and st.session_state.session_token:
            token = st.session_state.session_token
            
            db = Database()
            user = db.validate_session(token)
            
            if user:
                st.session_state.authenticated = True
                st.session_state.user = user
                
                # Load user's symbols
                symbols = db.get_user_symbols(user['id'])
                if symbols:
                    st.session_state.SYMBOLS = symbols
                
                print(f"✅ Auto-login successful for {user['username']}")
                return True
            else:
                # Invalid token, clear it
                del st.session_state.session_token
        
        return False