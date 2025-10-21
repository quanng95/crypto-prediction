import streamlit as st
from streamlit_cookies_manager import EncryptedCookieManager
import hashlib
import json
from datetime import datetime, timedelta

class SessionManager:
    def __init__(self):
        # Initialize cookie manager with a secret key
        self.cookies = EncryptedCookieManager(
            prefix="crypto_app_",
            password="your-secret-key-here-change-this-in-production"  # Thay đổi trong production
        )
        
        if not self.cookies.ready():
            st.stop()
    
    def create_session_token(self, user_id: int, username: str) -> str:
        """Create a session token"""
        # Create token from user_id, username and timestamp
        data = f"{user_id}:{username}:{datetime.now().isoformat()}"
        token = hashlib.sha256(data.encode()).hexdigest()
        return token
    
    def save_session(self, user: dict, remember_me: bool = True):
        """Save user session to cookies"""
        session_data = {
            'user_id': user['id'],
            'username': user['username'],
            'email': user['email'],
            'token': self.create_session_token(user['id'], user['username']),
            'timestamp': datetime.now().isoformat()
        }
        
        # Set cookie expiration (30 days if remember_me, else session only)
        if remember_me:
            expiry = datetime.now() + timedelta(days=30)
            self.cookies['session_data'] = json.dumps(session_data)
            self.cookies['expiry'] = expiry.isoformat()
        else:
            self.cookies['session_data'] = json.dumps(session_data)
        
        self.cookies.save()
    
    def load_session(self) -> dict:
        """Load user session from cookies"""
        try:
            if 'session_data' in self.cookies:
                session_data = json.loads(self.cookies['session_data'])
                
                # Check expiry if exists
                if 'expiry' in self.cookies:
                    expiry = datetime.fromisoformat(self.cookies['expiry'])
                    if datetime.now() > expiry:
                        self.clear_session()
                        return None
                
                return session_data
            
            return None
        
        except Exception as e:
            print(f"Error loading session: {e}")
            return None
    
    def clear_session(self):
        """Clear user session"""
        if 'session_data' in self.cookies:
            del self.cookies['session_data']
        if 'expiry' in self.cookies:
            del self.cookies['expiry']
        self.cookies.save()
    
    def is_valid_session(self) -> bool:
        """Check if session is valid"""
        session_data = self.load_session()
        return session_data is not None
