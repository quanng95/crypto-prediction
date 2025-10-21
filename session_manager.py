import streamlit as st
import extra_streamlit_components as stx
from database_postgres import Database

class SessionManager:
    def __init__(self):
        self.cookie_manager = stx.CookieManager()
    
    def save_session(self, session_token: str, remember_days: int = 30):
        """Save session token to cookie"""
        try:
            self.cookie_manager.set(
                'session_token', 
                session_token,
                expires_at=None if remember_days == 0 else f"{remember_days}d"
            )
            print(f"✅ Session saved to cookie")
        except Exception as e:
            print(f"Error saving session: {e}")
    
    def get_session(self) -> str:
        """Get session token from cookie"""
        try:
            token = self.cookie_manager.get('session_token')
            if token:
                print(f"✅ Session found in cookie")
            return token
        except Exception as e:
            print(f"Error getting session: {e}")
            return None
    
    def clear_session(self):
        """Clear session from cookie"""
        try:
            self.cookie_manager.delete('session_token')
            print(f"✅ Session cleared from cookie")
        except Exception as e:
            print(f"Error clearing session: {e}")
    
    def auto_login(self):
        """Auto login from cookie"""
        if st.session_state.get('authenticated', False):
            return True
        
        token = self.get_session()
        
        if token:
            db = Database()
            user = db.validate_session(token)
            
            if user:
                st.session_state.authenticated = True
                st.session_state.user = user
                st.session_state.session_token = token
                
                # Load user's symbols
                symbols = db.get_user_symbols(user['id'])
                if symbols:
                    st.session_state.SYMBOLS = symbols
                
                print(f"✅ Auto-login successful for {user['username']}")
                return True
            else:
                # Invalid token, clear cookie
                self.clear_session()
        
        return False
