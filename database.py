import sqlite3
import hashlib
import json
from typing import Optional, List, Dict
import re

class Database:
    def __init__(self, db_path: str = "crypto_app.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Users table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # User preferences table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_preferences (
                user_id INTEGER PRIMARY KEY,
                symbols TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        conn.commit()
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
        # Validate email
        if not self.validate_email(email):
            return False, "Invalid email format"
        
        # Validate password
        is_valid, message = self.validate_password(password)
        if not is_valid:
            return False, message
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            password_hash = self.hash_password(password)
            
            cursor.execute(
                "INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)",
                (username, email, password_hash)
            )
            
            user_id = cursor.lastrowid
            
            # Initialize default symbols for new user
            default_symbols = [
                "ETHUSDT", "BTCUSDT", "PAXGUSDT", "BNBUSDT", "SOLUSDT",
                "LINKUSDT", "PEPEUSDT", "XRPUSDT", "DOGEUSDT", "KAITOUSDT", "ADAUSDT"
            ]
            
            cursor.execute(
                "INSERT INTO user_preferences (user_id, symbols) VALUES (?, ?)",
                (user_id, json.dumps(default_symbols))
            )
            
            conn.commit()
            conn.close()
            
            return True, "Account created successfully!"
        
        except sqlite3.IntegrityError as e:
            if "username" in str(e):
                return False, "Username already exists"
            elif "email" in str(e):
                return False, "Email already exists"
            return False, "Error creating account"
        except Exception as e:
            return False, f"Error: {str(e)}"
    
    def authenticate_user(self, username: str, password: str) -> Optional[Dict]:
        """Authenticate user and return user info"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            password_hash = self.hash_password(password)
            
            cursor.execute(
                "SELECT id, username, email FROM users WHERE username = ? AND password_hash = ?",
                (username, password_hash)
            )
            
            result = cursor.fetchone()
            conn.close()
            
            if result:
                return {
                    'id': result[0],
                    'username': result[1],
                    'email': result[2]
                }
            
            return None
        
        except Exception as e:
            print(f"Authentication error: {e}")
            return None
    
    def get_user_symbols(self, user_id: int) -> List[str]:
        """Get user's saved symbols"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute(
                "SELECT symbols FROM user_preferences WHERE user_id = ?",
                (user_id,)
            )
            
            result = cursor.fetchone()
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
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute(
                "UPDATE user_preferences SET symbols = ? WHERE user_id = ?",
                (json.dumps(symbols), user_id)
            )
            
            conn.commit()
            conn.close()
            
            return True
        
        except Exception as e:
            print(f"Error saving symbols: {e}")
            return False
