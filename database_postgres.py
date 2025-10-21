import psycopg2
from psycopg2.extras import RealDictCursor
import hashlib
import json
import os
from typing import Optional, List, Dict
import re

class Database:
    def __init__(self):
        # Lấy DATABASE_URL từ environment variable (Railway tự động set)
        self.database_url = os.getenv('DATABASE_URL')
        
        if not self.database_url:
            raise Exception("DATABASE_URL not found in environment variables")
        
        # Fix for Railway PostgreSQL URL
        if self.database_url.startswith('postgres://'):
            self.database_url = self.database_url.replace('postgres://', 'postgresql://', 1)
        
        self.init_database()
    
    def get_connection(self):
        """Get database connection"""
        return psycopg2.connect(self.database_url)
    
    def init_database(self):
        """Initialize database tables"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Users table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id SERIAL PRIMARY KEY,
                username VARCHAR(255) UNIQUE NOT NULL,
                email VARCHAR(255) UNIQUE NOT NULL,
                password_hash VARCHAR(255) NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # User preferences table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_preferences (
                user_id INTEGER PRIMARY KEY,
                symbols TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
            )
        ''')
        
        # Sessions table (NEW - for remember me)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_sessions (
                id SERIAL PRIMARY KEY,
                user_id INTEGER NOT NULL,
                session_token VARCHAR(255) UNIQUE NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
            )
        ''')
        
        # Create index for faster session lookup
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_session_token 
            ON user_sessions(session_token)
        ''')
        
        conn.commit()
        cursor.close()
        conn.close()
    
    def hash_password(self, password: str) -> str:
        """Hash password using SHA-256"""
        return hashlib.sha256(password.encode()).hexdigest()
    
    def validate_password(self, password: str) -> tuple[bool, str]:
        """Validate password requirements"""
        if len(password) < 8:
            return False, "Password must be at least 8 characters long"
        
        if not re.search(r"[A-Z]", password):
            return False, "Password must contain at least one uppercase letter"
        
        if not re.search(r"[a-z]", password):
            return False, "Password must contain at least one lowercase letter"
        
        if not re.search(r"[0-9]", password):
            return False, "Password must contain at least one number"
        
        if not re.search(r"[!@#$%^&*(),.?\":{}|<>]", password):
            return False, "Password must contain at least one special character"
        
        return True, "Password is valid"
    
    def validate_email(self, email: str) -> bool:
        """Validate email format"""
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return re.match(pattern, email) is not None
    
    def create_user(self, username: str, email: str, password: str) -> tuple[bool, str]:
        """Create new user"""
        if not self.validate_email(email):
            return False, "Invalid email format"
        
        is_valid, message = self.validate_password(password)
        if not is_valid:
            return False, message
        
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            password_hash = self.hash_password(password)
            
            cursor.execute(
                "INSERT INTO users (username, email, password_hash) VALUES (%s, %s, %s) RETURNING id",
                (username, email, password_hash)
            )
            
            user_id = cursor.fetchone()[0]
            
            # Initialize default symbols
            default_symbols = [
                "ETHUSDT", "BTCUSDT", "PAXGUSDT", "BNBUSDT", "SOLUSDT",
                "LINKUSDT", "PEPEUSDT", "XRPUSDT", "DOGEUSDT", "KAITOUSDT", "ADAUSDT"
            ]
            
            cursor.execute(
                "INSERT INTO user_preferences (user_id, symbols) VALUES (%s, %s)",
                (user_id, json.dumps(default_symbols))
            )
            
            conn.commit()
            cursor.close()
            conn.close()
            
            return True, "Account created successfully!"
        
        except psycopg2.IntegrityError as e:
            if "username" in str(e):
                return False, "Username already exists"
            elif "email" in str(e):
                return False, "Email already exists"
            return False, "Error creating account"
        except Exception as e:
            return False, f"Error: {str(e)}"
    
    def authenticate_user(self, username: str, password: str) -> Optional[Dict]:
        """Authenticate user"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            password_hash = self.hash_password(password)
            
            cursor.execute(
                "SELECT id, username, email FROM users WHERE username = %s AND password_hash = %s",
                (username, password_hash)
            )
            
            result = cursor.fetchone()
            cursor.close()
            conn.close()
            
            if result:
                return dict(result)
            
            return None
        
        except Exception as e:
            print(f"Authentication error: {e}")
            return None
    
    def get_user_symbols(self, user_id: int) -> List[str]:
        """Get user's saved symbols"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            cursor.execute(
                "SELECT symbols FROM user_preferences WHERE user_id = %s",
                (user_id,)
            )
            
            result = cursor.fetchone()
            cursor.close()
            conn.close()
            
            if result:
                return json.loads(result[0])
            
            return []
        
        except Exception as e:
            print(f"Error getting symbols: {e}")
            return []
    
    def save_user_symbols(self, user_id: int, symbols: List[str]) -> bool:
        """Save user's symbols"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            cursor.execute(
                """
                INSERT INTO user_preferences (user_id, symbols) 
                VALUES (%s, %s)
                ON CONFLICT (user_id) 
                DO UPDATE SET symbols = EXCLUDED.symbols
                """,
                (user_id, json.dumps(symbols))
            )
            
            conn.commit()
            cursor.close()
            conn.close()
            
            return True
        
        except Exception as e:
            print(f"Error saving symbols: {e}")
            return False
    
    # ============================================
    # SESSION MANAGEMENT (NEW)
    # ============================================
    
    def create_session(self, user_id: int, remember_me: bool = True) -> str:
        """Create session token for user"""
        import secrets
        from datetime import datetime, timedelta
        
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            # Generate secure token
            session_token = secrets.token_urlsafe(32)
            
            # Set expiration
            if remember_me:
                expires_at = datetime.now() + timedelta(days=30)
            else:
                expires_at = datetime.now() + timedelta(hours=24)
            
            # Clean old sessions for this user
            cursor.execute(
                "DELETE FROM user_sessions WHERE user_id = %s",
                (user_id,)
            )
            
            # Insert new session
            cursor.execute(
                """
                INSERT INTO user_sessions (user_id, session_token, expires_at)
                VALUES (%s, %s, %s)
                """,
                (user_id, session_token, expires_at)
            )
            
            conn.commit()
            cursor.close()
            conn.close()
            
            return session_token
        
        except Exception as e:
            print(f"Error creating session: {e}")
            return None
    
    def validate_session(self, session_token: str) -> Optional[Dict]:
        """Validate session token and return user info"""
        from datetime import datetime
        
        try:
            conn = self.get_connection()
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            cursor.execute(
                """
                SELECT u.id, u.username, u.email, s.expires_at
                FROM user_sessions s
                JOIN users u ON s.user_id = u.id
                WHERE s.session_token = %s
                """,
                (session_token,)
            )
            
            result = cursor.fetchone()
            cursor.close()
            conn.close()
            
            if result:
                # Check if session expired
                if datetime.now() > result['expires_at']:
                    self.delete_session(session_token)
                    return None
                
                return {
                    'id': result['id'],
                    'username': result['username'],
                    'email': result['email']
                }
            
            return None
        
        except Exception as e:
            print(f"Error validating session: {e}")
            return None
    
    def delete_session(self, session_token: str) -> bool:
        """Delete session"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            cursor.execute(
                "DELETE FROM user_sessions WHERE session_token = %s",
                (session_token,)
            )
            
            conn.commit()
            cursor.close()
            conn.close()
            
            return True
        
        except Exception as e:
            print(f"Error deleting session: {e}")
            return False
    
    def cleanup_expired_sessions(self):
        """Clean up expired sessions (run periodically)"""
        from datetime import datetime
        
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            cursor.execute(
                "DELETE FROM user_sessions WHERE expires_at < %s",
                (datetime.now(),)
            )
            
            conn.commit()
            cursor.close()
            conn.close()
            
        except Exception as e:
            print(f"Error cleaning sessions: {e}")
