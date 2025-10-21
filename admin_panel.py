import streamlit as st
import pandas as pd
from database import Database
import json
from datetime import datetime

def render_admin_login():
    """Admin login page"""
    st.markdown("""
    <div class="header-container">
        <h1 class="header-title">🔐 Admin Panel</h1>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("### Admin Authentication")
        
        with st.form("admin_login"):
            admin_password = st.text_input("Admin Password", type="password")
            submit = st.form_submit_button("🔓 Login", use_container_width=True, type="primary")
            
            if submit:
                # Hardcoded admin password (trong production nên lưu trong env variable)
                if admin_password == "admin123@":
                    st.session_state.admin_authenticated = True
                    st.success("✅ Admin access granted!")
                    st.rerun()
                else:
                    st.error("❌ Invalid admin password!")

def render_admin_panel():
    """Main admin panel"""
    st.markdown("""
    <div class="header-container">
        <h1 class="header-title">👨‍💼 Admin Dashboard</h1>
    </div>
    """, unsafe_allow_html=True)
    
    # Logout button
    col1, col2 = st.columns([6, 1])
    with col2:
        if st.button("🚪 Logout", type="secondary"):
            st.session_state.admin_authenticated = False
            st.session_state.page = "home"
            st.rerun()
    
    st.markdown("---")
    
    db = Database()
    
    # Tabs for different admin functions
    tab1, tab2, tab3, tab4 = st.tabs([
        "👥 Users Management",
        "📊 User Statistics",
        "🗑️ Delete Users",
        "🔍 Search Users"
    ])
    
    # TAB 1: Users Management
    with tab1:
        st.markdown("### 👥 All Users")
        
        users_data = get_all_users(db)
        
        if users_data:
            df = pd.DataFrame(users_data)
            
            # Display user count
            st.metric("Total Users", len(df))
            
            # Display users table
            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "id": st.column_config.NumberColumn("ID", width="small"),
                    "username": st.column_config.TextColumn("Username", width="medium"),
                    "email": st.column_config.TextColumn("Email", width="medium"),
                    "symbols_count": st.column_config.NumberColumn("Symbols", width="small"),
                    "created_at": st.column_config.DatetimeColumn("Created At", width="medium")
                }
            )
            
            # Export to CSV
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download Users CSV",
                data=csv,
                file_name=f"users_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        else:
            st.info("No users found in database")
    
    # TAB 2: User Statistics
    with tab2:
        st.markdown("### 📊 Statistics")
        
        users_data = get_all_users(db)
        
        if users_data:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Users", len(users_data))
            
            with col2:
                avg_symbols = sum(u['symbols_count'] for u in users_data) / len(users_data)
                st.metric("Avg Symbols per User", f"{avg_symbols:.1f}")
            
            with col3:
                total_symbols = sum(u['symbols_count'] for u in users_data)
                st.metric("Total Tracked Symbols", total_symbols)
            
            st.markdown("---")
            
            # Most popular symbols
            st.markdown("#### 🔥 Most Popular Symbols")
            symbol_counts = get_popular_symbols(db)
            
            if symbol_counts:
                df_symbols = pd.DataFrame(symbol_counts, columns=['Symbol', 'Users Count'])
                st.bar_chart(df_symbols.set_index('Symbol'))
    
    # TAB 3: Delete Users
    with tab3:
        st.markdown("### 🗑️ Delete User")
        
        st.warning("⚠️ Warning: This action cannot be undone!")
        
        users_data = get_all_users(db)
        
        if users_data:
            usernames = [u['username'] for u in users_data]
            
            selected_user = st.selectbox("Select user to delete", usernames)
            
            if st.button("🗑️ Delete User", type="primary"):
                if delete_user(db, selected_user):
                    st.success(f"✅ User '{selected_user}' deleted successfully!")
                    st.rerun()
                else:
                    st.error("❌ Error deleting user")
    
    # TAB 4: Search Users
    with tab4:
        st.markdown("### 🔍 Search Users")
        
        search_query = st.text_input("Search by username or email", placeholder="Enter username or email...")
        
        if search_query:
            users_data = search_users(db, search_query)
            
            if users_data:
                for user in users_data:
                    with st.expander(f"👤 {user['username']} ({user['email']})"):
                        st.write(f"**User ID:** {user['id']}")
                        st.write(f"**Created:** {user['created_at']}")
                        st.write(f"**Symbols Count:** {user['symbols_count']}")
                        
                        # Show user's symbols
                        symbols = get_user_symbols_detail(db, user['id'])
                        if symbols:
                            st.write("**Tracked Symbols:**")
                            st.write(", ".join(symbols))
            else:
                st.info("No users found")

def get_all_users(db: Database):
    """Get all users from database"""
    try:
        import sqlite3
        conn = sqlite3.connect(db.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT u.id, u.username, u.email, u.created_at, up.symbols
            FROM users u
            LEFT JOIN user_preferences up ON u.id = up.user_id
        """)
        
        results = cursor.fetchall()
        conn.close()
        
        users_data = []
        for row in results:
            symbols = json.loads(row[4]) if row[4] else []
            users_data.append({
                'id': row[0],
                'username': row[1],
                'email': row[2],
                'created_at': row[3],
                'symbols_count': len(symbols)
            })
        
        return users_data
    
    except Exception as e:
        print(f"Error: {e}")
        return []

def get_popular_symbols(db: Database):
    """Get most popular symbols across all users"""
    try:
        import sqlite3
        conn = sqlite3.connect(db.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT symbols FROM user_preferences")
        results = cursor.fetchall()
        conn.close()
        
        symbol_count = {}
        for row in results:
            symbols = json.loads(row[0])
            for symbol in symbols:
                symbol_count[symbol] = symbol_count.get(symbol, 0) + 1
        
        # Sort by count
        sorted_symbols = sorted(symbol_count.items(), key=lambda x: x[1], reverse=True)
        
        return sorted_symbols[:10]  # Top 10
    
    except Exception as e:
        print(f"Error: {e}")
        return []

def delete_user(db: Database, username: str):
    """Delete user and their preferences"""
    try:
        import sqlite3
        conn = sqlite3.connect(db.db_path)
        cursor = conn.cursor()
        
        # Get user ID
        cursor.execute("SELECT id FROM users WHERE username = ?", (username,))
        result = cursor.fetchone()
        
        if result:
            user_id = result[0]
            
            # Delete preferences
            cursor.execute("DELETE FROM user_preferences WHERE user_id = ?", (user_id,))
            
            # Delete user
            cursor.execute("DELETE FROM users WHERE id = ?", (user_id,))
            
            conn.commit()
            conn.close()
            return True
        
        conn.close()
        return False
    
    except Exception as e:
        print(f"Error: {e}")
        return False

def search_users(db: Database, query: str):
    """Search users by username or email"""
    try:
        import sqlite3
        conn = sqlite3.connect(db.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT u.id, u.username, u.email, u.created_at, up.symbols
            FROM users u
            LEFT JOIN user_preferences up ON u.id = up.user_id
            WHERE u.username LIKE ? OR u.email LIKE ?
        """, (f"%{query}%", f"%{query}%"))
        
        results = cursor.fetchall()
        conn.close()
        
        users_data = []
        for row in results:
            symbols = json.loads(row[4]) if row[4] else []
            users_data.append({
                'id': row[0],
                'username': row[1],
                'email': row[2],
                'created_at': row[3],
                'symbols_count': len(symbols)
            })
        
        return users_data
    
    except Exception as e:
        print(f"Error: {e}")
        return []

def get_user_symbols_detail(db: Database, user_id: int):
    """Get detailed symbols for a user"""
    try:
        import sqlite3
        conn = sqlite3.connect(db.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT symbols FROM user_preferences WHERE user_id = ?", (user_id,))
        result = cursor.fetchone()
        conn.close()
        
        if result:
            return json.loads(result[0])
        
        return []
    
    except Exception as e:
        print(f"Error: {e}")
        return []
