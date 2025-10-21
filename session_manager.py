import streamlit as st
from database_postgres import Database
import secrets

class SessionManager:
    
    @staticmethod
    def save_session(session_token: str):
        """Save session token to query params"""
        try:
            # Lưu vào session_state (persist trong session)
            st.session_state.session_token = session_token
            
            # Lưu vào query params (persist qua F5)
            st.query_params["token"] = session_token
            
            return True
        except Exception as e:
            print(f"Error saving session: {e}")
            return False
    
    @staticmethod
    def get_session():
        """Get session token from query params or session_state"""
        try:
            # Ưu tiên lấy từ session_state (nhanh hơn)
            if 'session_token' in st.session_state:
                return st.session_state.session_token
            
            # Nếu không có, lấy từ query params (sau khi F5)
            token = st.query_params.get("token")
            
            if token:
                # Restore vào session_state
                st.session_state.session_token = token
                return token
            
            return None
        except Exception as e:
            print(f"Error getting session: {e}")
            return None
    
    @staticmethod
    def clear_session():
        """Clear session from query params and session_state"""
        try:
            # Xóa khỏi session_state
            if 'session_token' in st.session_state:
                del st.session_state.session_token
            
            # Xóa khỏi query params
            if "token" in st.query_params:
                del st.query_params["token"]
            
            return True
        except Exception as e:
            print(f"Error clearing session: {e}")
            return False
    
    @staticmethod
    def auto_login():
        """Auto login from session token"""
        # Skip if already authenticated
        if st.session_state.get('authenticated', False):
            return True
        
        # Try to get token
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
