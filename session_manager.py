import json
import hashlib
import os
from datetime import datetime, timedelta
from pathlib import Path

class SessionManager:
    def __init__(self):
        self.session_dir = Path(".streamlit_sessions")
        self.session_dir.mkdir(exist_ok=True)
        self.session_file = self.session_dir / "active_session.json"
    
    def create_session_token(self, user_id: int, username: str) -> str:
        """Create a session token"""
        data = f"{user_id}:{username}:{datetime.now().isoformat()}"
        token = hashlib.sha256(data.encode()).hexdigest()
        return token
    
    def save_session(self, user: dict, remember_me: bool = True):
        """Save user session to file"""
        session_data = {
            'user_id': user['id'],
            'username': user['username'],
            'email': user['email'],
            'token': self.create_session_token(user['id'], user['username']),
            'timestamp': datetime.now().isoformat(),
            'remember_me': remember_me
        }
        
        # Set expiration
        if remember_me:
            expiry = datetime.now() + timedelta(days=30)
            session_data['expiry'] = expiry.isoformat()
        
        try:
            with open(self.session_file, 'w') as f:
                json.dump(session_data, f)
            print(f"✅ Session saved for {user['username']}")
        except Exception as e:
            print(f"Error saving session: {e}")
    
    def load_session(self) -> dict:
        """Load user session from file"""
        try:
            if not self.session_file.exists():
                return None
            
            with open(self.session_file, 'r') as f:
                session_data = json.load(f)
            
            # Check expiry if exists
            if 'expiry' in session_data:
                expiry = datetime.fromisoformat(session_data['expiry'])
                if datetime.now() > expiry:
                    self.clear_session()
                    return None
            
            return session_data
        
        except Exception as e:
            print(f"Error loading session: {e}")
            return None
    
    def clear_session(self):
        """Clear user session"""
        try:
            if self.session_file.exists():
                os.remove(self.session_file)
            print("✅ Session cleared")
        except Exception as e:
            print(f"Error clearing session: {e}")
    
    def is_valid_session(self) -> bool:
        """Check if session is valid"""
        session_data = self.load_session()
        return session_data is not None
