import streamlit as st
import pandas as pd
from database_postgres import Database
import json
import os
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
                # Get admin password from environment variable
                correct_password = os.getenv("ADMIN_PASSWORD", "admin123@")
                
                if admin_password == correct_password:
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
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "👥 Users Management",
        "📊 Statistics",
        "🔍 Search & Edit",
        "🗑️ Delete Data",
        "🔧 Database Tools"
    ])
    
    # ============================================
    # TAB 1: Users Management
    # ============================================
    with tab1:
        st.markdown("### 👥 All Users")
        
        users_data = get_all_users(db)
        
        if users_data:
            df = pd.DataFrame(users_data)
            
            # Display metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Users", len(df))
            with col2:
                avg_symbols = df['symbols_count'].mean()
                st.metric("Avg Symbols/User", f"{avg_symbols:.1f}")
            with col3:
                total_symbols = df['symbols_count'].sum()
                st.metric("Total Symbols Tracked", total_symbols)
            with col4:
                active_today = len(df[df['created_at'].str.contains(datetime.now().strftime('%Y-%m-%d'))])
                st.metric("New Users Today", active_today)
            
            st.markdown("---")
            
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
                    "created_at": st.column_config.DatetimeColumn("Created At", width="medium", format="DD/MM/YYYY HH:mm")
                }
            )
            
            # Export to CSV
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download Users CSV",
                data=csv,
                file_name=f"users_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        else:
            st.info("No users found in database")
    
    # ============================================
    # TAB 2: Statistics
    # ============================================
    with tab2:
        st.markdown("### 📊 Database Statistics")
        
        users_data = get_all_users(db)
        
        if users_data:
            df = pd.DataFrame(users_data)
            
            # Overview metrics
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
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.bar_chart(df_symbols.set_index('Symbol'))
                
                with col2:
                    st.dataframe(
                        df_symbols,
                        use_container_width=True,
                        hide_index=True
                    )
            
            st.markdown("---")
            
            # User registration trend
            st.markdown("#### 📈 User Registration Trend")
            df['date'] = pd.to_datetime(df['created_at']).dt.date
            user_trend = df.groupby('date').size().reset_index(name='New Users')
            
            st.line_chart(user_trend.set_index('date'))
            
            st.markdown("---")
            
            # Symbols distribution
            st.markdown("#### 📊 Symbols Distribution")
            symbols_dist = df['symbols_count'].value_counts().sort_index()
            st.bar_chart(symbols_dist)
    
    # ============================================
    # TAB 3: Search & Edit
    # ============================================
    with tab3:
        st.markdown("### 🔍 Search & Edit Users")
        
        search_query = st.text_input("🔎 Search by username or email", placeholder="Enter username or email...")
        
        if search_query:
            users_data = search_users(db, search_query)
            
            if users_data:
                for user in users_data:
                    with st.expander(f"👤 {user['username']} ({user['email']})", expanded=True):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write(f"**User ID:** {user['id']}")
                            st.write(f"**Username:** {user['username']}")
                            st.write(f"**Email:** {user['email']}")
                            st.write(f"**Created:** {user['created_at']}")
                            st.write(f"**Symbols Count:** {user['symbols_count']}")
                        
                        with col2:
                            # Show user's symbols
                            symbols = get_user_symbols_detail(db, user['id'])
                            if symbols:
                                st.write("**Tracked Symbols:**")
                                st.write(", ".join(symbols))
                            else:
                                st.write("**No symbols tracked**")
                        
                        st.markdown("---")
                        
                        # Edit symbols
                        st.markdown("**Edit Symbols:**")
                        
                        col_edit1, col_edit2 = st.columns([3, 1])
                        
                        with col_edit1:
                            new_symbols_text = st.text_area(
                                "Symbols (comma separated)",
                                value=", ".join(symbols) if symbols else "",
                                key=f"symbols_edit_{user['id']}",
                                height=100
                            )
                        
                        with col_edit2:
                            st.write("")
                            st.write("")
                            if st.button("💾 Save", key=f"save_{user['id']}", use_container_width=True):
                                new_symbols = [s.strip().upper() for s in new_symbols_text.split(',') if s.strip()]
                                
                                if update_user_symbols(db, user['id'], new_symbols):
                                    st.success("✅ Symbols updated!")
                                    st.rerun()
                                else:
                                    st.error("❌ Error updating symbols")
                            
                            if st.button("🔄 Reset Password", key=f"reset_{user['id']}", use_container_width=True, type="secondary"):
                                new_password = f"temp_{user['id']}_123"
                                if reset_user_password(db, user['id'], new_password):
                                    st.success(f"✅ Password reset to: `{new_password}`")
                                else:
                                    st.error("❌ Error resetting password")
            else:
                st.info("No users found")
        else:
            st.info("👆 Enter username or email to search")
    
    # ============================================
    # TAB 4: Delete Data
    # ============================================
    with tab4:
        st.markdown("### 🗑️ Delete Data")
        
        st.warning("⚠️ **Warning:** These actions cannot be undone!")
        
        st.markdown("---")
        
        # Delete user
        st.markdown("#### Delete User")
        
        users_data = get_all_users(db)
        
        if users_data:
            usernames = [u['username'] for u in users_data]
            
            col1, col2 = st.columns([3, 1])
            
            with col1:
                selected_user = st.selectbox("Select user to delete", usernames, key="delete_user_select")
            
            with col2:
                st.write("")
                st.write("")
                if st.button("🗑️ Delete User", type="primary", use_container_width=True):
                    if delete_user(db, selected_user):
                        st.success(f"✅ User '{selected_user}' deleted successfully!")
                        st.rerun()
                    else:
                        st.error("❌ Error deleting user")
        
        st.markdown("---")
        
        # Delete all expired sessions
        st.markdown("#### Clean Expired Sessions")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.write("Remove all expired session tokens from database")
        
        with col2:
            if st.button("🧹 Clean Sessions", use_container_width=True):
                count = cleanup_expired_sessions(db)
                st.success(f"✅ Cleaned {count} expired sessions!")
        
        st.markdown("---")
        
        # Danger zone - Delete all users
        st.markdown("#### ⚠️ Danger Zone")
        
        with st.expander("🚨 Delete ALL Users (Danger!)"):
            st.error("**THIS WILL DELETE ALL USERS AND THEIR DATA!**")
            
            confirm_text = st.text_input("Type 'DELETE ALL USERS' to confirm", key="confirm_delete_all")
            
            if st.button("💥 DELETE ALL USERS", type="primary", use_container_width=True):
                if confirm_text == "DELETE ALL USERS":
                    if delete_all_users(db):
                        st.success("✅ All users deleted!")
                        st.rerun()
                    else:
                        st.error("❌ Error deleting users")
                else:
                    st.error("❌ Confirmation text incorrect!")
    
    # ============================================
    # TAB 5: Database Tools
    # ============================================
    with tab5:
        st.markdown("### 🔧 Database Tools")
        
        # Database info
        st.markdown("#### 📊 Database Information")
        
        db_info = get_database_info(db)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Users Table", f"{db_info['users_count']} rows")
        
        with col2:
            st.metric("Preferences Table", f"{db_info['preferences_count']} rows")
        
        with col3:
            st.metric("Sessions Table", f"{db_info['sessions_count']} rows")
        
        st.markdown("---")
        
        # Execute custom SQL (dangerous!)
        st.markdown("#### 🔥 Execute Custom SQL")
        
        with st.expander("⚠️ Advanced - Execute SQL Query"):
            st.warning("**Warning:** This can modify or delete data!")
            
            sql_query = st.text_area(
                "SQL Query",
                placeholder="SELECT * FROM users LIMIT 10;",
                height=150
            )
            
            col1, col2 = st.columns([3, 1])
            
            with col2:
                if st.button("▶️ Execute", type="primary", use_container_width=True):
                    result = execute_custom_sql(db, sql_query)
                    
                    if result['success']:
                        st.success("✅ Query executed successfully!")
                        
                        if result['data']:
                            df = pd.DataFrame(result['data'])
                            st.dataframe(df, use_container_width=True)
                            
                            # Export results
                            csv = df.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Results",
                                data=csv,
                                file_name=f"query_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv"
                            )
                        else:
                            st.info("Query executed but returned no data")
                    else:
                        st.error(f"❌ Error: {result['error']}")
        
        st.markdown("---")
        
        # Backup database
        st.markdown("#### 💾 Backup Database")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.write("Export all database data to JSON")
        
        with col2:
            if st.button("📥 Backup", use_container_width=True):
                backup_data = backup_database(db)
                
                if backup_data:
                    json_data = json.dumps(backup_data, indent=2, default=str)
                    
                    st.download_button(
                        label="💾 Download Backup",
                        data=json_data,
                        file_name=f"database_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json",
                        use_container_width=True
                    )

# ============================================
# HELPER FUNCTIONS
# ============================================

def get_all_users(db: Database):
    """Get all users from database"""
    try:
        conn = db.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT u.id, u.username, u.email, u.created_at, up.symbols
            FROM users u
            LEFT JOIN user_preferences up ON u.id = up.user_id
            ORDER BY u.created_at DESC
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
                'created_at': str(row[3]),
                'symbols_count': len(symbols)
            })
        
        return users_data
    
    except Exception as e:
        print(f"Error: {e}")
        return []

def get_popular_symbols(db: Database):
    """Get most popular symbols across all users"""
    try:
        conn = db.get_connection()
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
        
        return sorted_symbols[:15]  # Top 15
    
    except Exception as e:
        print(f"Error: {e}")
        return []

def search_users(db: Database, query: str):
    """Search users by username or email"""
    try:
        conn = db.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT u.id, u.username, u.email, u.created_at, up.symbols
            FROM users u
            LEFT JOIN user_preferences up ON u.id = up.user_id
            WHERE u.username ILIKE %s OR u.email ILIKE %s
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
                'created_at': str(row[3]),
                'symbols_count': len(symbols)
            })
        
        return users_data
    
    except Exception as e:
        print(f"Error: {e}")
        return []

def get_user_symbols_detail(db: Database, user_id: int):
    """Get detailed symbols for a user"""
    try:
        conn = db.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT symbols FROM user_preferences WHERE user_id = %s", (user_id,))
        result = cursor.fetchone()
        conn.close()
        
        if result:
            return json.loads(result[0])
        
        return []
    
    except Exception as e:
        print(f"Error: {e}")
        return []

def update_user_symbols(db: Database, user_id: int, symbols: list):
    """Update user's symbols"""
    try:
        conn = db.get_connection()
        cursor = conn.cursor()
        
        cursor.execute(
            """
            UPDATE user_preferences 
            SET symbols = %s 
            WHERE user_id = %s
            """,
            (json.dumps(symbols), user_id)
        )
        
        conn.commit()
        conn.close()
        return True
    
    except Exception as e:
        print(f"Error: {e}")
        return False

def reset_user_password(db: Database, user_id: int, new_password: str):
    """Reset user password"""
    try:
        password_hash = db.hash_password(new_password)
        
        conn = db.get_connection()
        cursor = conn.cursor()
        
        cursor.execute(
            "UPDATE users SET password_hash = %s WHERE id = %s",
            (password_hash, user_id)
        )
        
        conn.commit()
        conn.close()
        return True
    
    except Exception as e:
        print(f"Error: {e}")
        return False

def delete_user(db: Database, username: str):
    """Delete user and their data"""
    try:
        conn = db.get_connection()
        cursor = conn.cursor()
        
        # Get user ID
        cursor.execute("SELECT id FROM users WHERE username = %s", (username,))
        result = cursor.fetchone()
        
        if result:
            user_id = result[0]
            
            # Delete preferences
            cursor.execute("DELETE FROM user_preferences WHERE user_id = %s", (user_id,))
            
            # Delete sessions
            cursor.execute("DELETE FROM user_sessions WHERE user_id = %s", (user_id,))
            
            # Delete user
            cursor.execute("DELETE FROM users WHERE id = %s", (user_id,))
            
            conn.commit()
            conn.close()
            return True
        
        conn.close()
        return False
    
    except Exception as e:
        print(f"Error: {e}")
        return False

def cleanup_expired_sessions(db: Database):
    """Clean up expired sessions"""
    try:
        conn = db.get_connection()
        cursor = conn.cursor()
        
        cursor.execute(
            "DELETE FROM user_sessions WHERE expires_at < NOW()"
        )
        
        count = cursor.rowcount
        conn.commit()
        conn.close()
        
        return count
    
    except Exception as e:
        print(f"Error: {e}")
        return 0

def delete_all_users(db: Database):
    """Delete all users (DANGER!)"""
    try:
        conn = db.get_connection()
        cursor = conn.cursor()
        
        # Delete all preferences
        cursor.execute("DELETE FROM user_preferences")
        
        # Delete all sessions
        cursor.execute("DELETE FROM user_sessions")
        
        # Delete all users
        cursor.execute("DELETE FROM users")
        
        conn.commit()
        conn.close()
        return True
    
    except Exception as e:
        print(f"Error: {e}")
        return False

def get_database_info(db: Database):
    """Get database table info"""
    try:
        conn = db.get_connection()
        cursor = conn.cursor()
        
        # Count users
        cursor.execute("SELECT COUNT(*) FROM users")
        users_count = cursor.fetchone()[0]
        
        # Count preferences
        cursor.execute("SELECT COUNT(*) FROM user_preferences")
        preferences_count = cursor.fetchone()[0]
        
        # Count sessions
        cursor.execute("SELECT COUNT(*) FROM user_sessions")
        sessions_count = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            'users_count': users_count,
            'preferences_count': preferences_count,
            'sessions_count': sessions_count
        }
    
    except Exception as e:
        print(f"Error: {e}")
        return {
            'users_count': 0,
            'preferences_count': 0,
            'sessions_count': 0
        }

def execute_custom_sql(db: Database, sql_query: str):
    """Execute custom SQL query"""
    try:
        conn = db.get_connection()
        cursor = conn.cursor()
        
        cursor.execute(sql_query)
        
        # Check if query returns data
        if cursor.description:
            columns = [desc[0] for desc in cursor.description]
            results = cursor.fetchall()
            
            data = []
            for row in results:
                data.append(dict(zip(columns, row)))
            
            conn.commit()
            conn.close()
            
            return {'success': True, 'data': data, 'error': None}
        else:
            conn.commit()
            conn.close()
            return {'success': True, 'data': None, 'error': None}
    
    except Exception as e:
        return {'success': False, 'data': None, 'error': str(e)}

def backup_database(db: Database):
    """Backup entire database to JSON"""
    try:
        conn = db.get_connection()
        cursor = conn.cursor()
        
        # Get all users
        cursor.execute("SELECT * FROM users")
        users_columns = [desc[0] for desc in cursor.description]
        users = [dict(zip(users_columns, row)) for row in cursor.fetchall()]
        
        # Get all preferences
        cursor.execute("SELECT * FROM user_preferences")
        prefs_columns = [desc[0] for desc in cursor.description]
        preferences = [dict(zip(prefs_columns, row)) for row in cursor.fetchall()]
        
        # Get all sessions
        cursor.execute("SELECT * FROM user_sessions")
        sessions_columns = [desc[0] for desc in cursor.description]
        sessions = [dict(zip(sessions_columns, row)) for row in cursor.fetchall()]
        
        conn.close()
        
        return {
            'backup_date': datetime.now().isoformat(),
            'users': users,
            'preferences': preferences,
            'sessions': sessions
        }
    
    except Exception as e:
        print(f"Error: {e}")
        return None
