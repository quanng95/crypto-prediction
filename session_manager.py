import streamlit as st
from streamlit_cookies_manager import EncryptedCookieManager
from database_postgres import Database
import os

class SessionManager:
    
    @staticmethod
    def get_cookie_manager():
        """Get cookie manager instance"""
        # Secret key for encryption (nên lưu trong env variable)
        cookies = EncryptedCookieManager(
            prefix="crypto_app_",
            password=os.getenv("COOKIE_PASSWORD", "your-secret-key-change-this")
        )
        
        if not cookies.ready():
            st.stop()
        
        return cookies
    
    @staticmethod
    def save_session(session_token: str):
        """Save session token to cookies"""
        cookies = SessionManager.get_cookie_manager()
        cookies['session_token'] = session_token
        cookies.save()
    
    @staticmethod
    def get_session():
        """Get session token from cookies"""
        cookies = SessionManager.get_cookie_manager()
        return cookies.get('session_token')
    
    @staticmethod
    def clear_session():
        """Clear session from cookies"""
        cookies = SessionManager.get_cookie_manager()
        if 'session_token' in cookies:
            del cookies['session_token']
            cookies.save()
    
    @staticmethod
    def auto_login():
        """Auto login from cookies"""
        # Skip if already authenticated
        if st.session_state.get('authenticated', False):
            return True
        
        # Try to get token from cookies
        session_token = SessionManager.get_session()
        
        if session_token:
            db = Database()
            user = db.validate_session(session_token)
            
            if user:
                # Restore session
                st.session_state.authenticated = True
                st.session_state.user = user
                st.session_state.session_token = session_token
                
                # Load user's symbols
                symbols = db.get_user_symbols(user['id'])
                if symbols:
                    st.session_state.SYMBOLS = symbols
                else:
                    # Initialize default symbols
                    default_symbols = [
                        "ETHUSDT", "BTCUSDT", "PAXGUSDT", "BNBUSDT", "SOLUSDT",
                        "LINKUSDT", "PEPEUSDT", "XRPUSDT", "DOGEUSDT", "KAITOUSDT", "ADAUSDT"
                    ]
                    st.session_state.SYMBOLS = default_symbols
                    db.save_user_symbols(user['id'], default_symbols)
                
                print(f"✅ Auto-login successful for {user['username']}")
                return True
            else:
                # Invalid token, clear it
                SessionManager.clear_session()
        
        return False
