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
            password=os.getenv("COOKIE_PASSWORD", "your-secret-key-change-this-to-random-string")
        )
        
        # QUAN TRỌNG: Không gọi st.stop() ở đây
        # Chỉ return cookies, check ready() ở nơi gọi
        return cookies
    
    @staticmethod
    def save_session(session_token: str):
        """Save session token to cookies"""
        try:
            cookies = SessionManager.get_cookie_manager()
            
            if not cookies.ready():
                return False
            
            cookies['session_token'] = session_token
            cookies.save()
            return True
        except Exception as e:
            print(f"Error saving session: {e}")
            return False
    
    @staticmethod
    def get_session():
        """Get session token from cookies"""
        try:
            cookies = SessionManager.get_cookie_manager()
            
            if not cookies.ready():
                return None
            
            return cookies.get('session_token')
        except Exception as e:
            print(f"Error getting session: {e}")
            return None
    
    @staticmethod
    def clear_session():
        """Clear session from cookies"""
        try:
            cookies = SessionManager.get_cookie_manager()
            
            if not cookies.ready():
                return False
            
            if 'session_token' in cookies:
                del cookies['session_token']
                cookies.save()
            return True
        except Exception as e:
            print(f"Error clearing session: {e}")
            return False
    
    @staticmethod
    def auto_login():
        """Auto login from cookies - KHÔNG GỌI STREAMLIT COMMANDS"""
        # Skip if already authenticated
        if st.session_state.get('authenticated', False):
            return True
        
        # Try to get token from cookies
        session_token = SessionManager.get_session()
        
        if session_token:
            db = Database()
            user = db.validate_session(session_token)
            
            if user:
                # Restore session (chỉ set session_state, không gọi st commands)
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
