import streamlit as st
import requests
from typing import List, Dict
from datetime import datetime

class SymbolManager:
    """Simple symbol manager with search"""
    
    @staticmethod
    @st.cache_data(ttl=3600)
    def fetch_all_binance_symbols():
        """Fetch all USDT pairs from Binance"""
        try:
            url = "https://api.binance.com/api/v3/exchangeInfo"
            response = requests.get(url, timeout=10)
            data = response.json()
            
            symbols = []
            for symbol_info in data['symbols']:
                if (symbol_info['status'] == 'TRADING' and 
                    symbol_info['quoteAsset'] == 'USDT'):
                    symbols.append({
                        'symbol': symbol_info['symbol'],
                        'baseAsset': symbol_info['baseAsset']
                    })
            
            return symbols
        except Exception as e:
            print(f"Error fetching Binance symbols: {e}")
            return []
    
    @staticmethod
    def search_symbols(query: str, all_symbols: List[Dict]) -> List[Dict]:
        """Search symbols by query"""
        if not query or len(query) < 2:
            return []
        
        query = query.upper()
        results = []
        
        for symbol_info in all_symbols:
            symbol = symbol_info['symbol']
            base = symbol_info['baseAsset']
            
            if query in base or query in symbol:
                results.append(symbol_info)
                
                if len(results) >= 10:
                    break
        
        return results

def render_simple_add_symbol(current_symbols: List[str]) -> List[str]:
    """
    Simple symbol add interface with FORCED database persistence
    Returns updated symbol list
    """
    
    # Initialize
    if 'all_binance_symbols' not in st.session_state:
        st.session_state.all_binance_symbols = SymbolManager.fetch_all_binance_symbols()
    
    if 'search_query' not in st.session_state:
        st.session_state.search_query = ""
    
    if 'show_suggestions' not in st.session_state:
        st.session_state.show_suggestions = True
    
    # Search box
    search_query = st.text_input(
        "🔍 Add Symbol",
        value=st.session_state.search_query,
        placeholder="Type symbol (e.g., BTC, ETH, KAI...)",
        key="search_input",
        label_visibility="collapsed"
    )
    
    # Update search query
    if search_query != st.session_state.search_query:
        st.session_state.search_query = search_query
        st.session_state.show_suggestions = True
    
    # Search and show suggestions
    if st.session_state.show_suggestions and search_query and len(search_query) >= 2:
        results = SymbolManager.search_symbols(
            search_query, 
            st.session_state.all_binance_symbols
        )
        
        if results:
            st.markdown("**💡 Suggestions:**")
            
            for result in results:
                symbol = result['symbol']
                base = result['baseAsset']
                
                col1, col2 = st.columns([4, 1])
                
                with col1:
                    st.markdown(f"**{symbol}** ({base}/USDT)")
                
                with col2:
                    if symbol not in current_symbols:
                        if st.button("➕", key=f"add_{symbol}", use_container_width=True):
                            # Add to current symbols
                            updated_symbols = current_symbols + [symbol]
                            
                            # ✅ FORCE SAVE TO DATABASE if authenticated
                            if st.session_state.get('authenticated', False):
                                from database import Database
                                db = Database()
                                
                                # Try to save 3 times
                                for attempt in range(3):
                                    success = db.save_user_symbols(
                                        st.session_state.user['id'], 
                                        updated_symbols
                                    )
                                    
                                    if success:
                                        # Verify by reading back
                                        saved_symbols, _ = db.get_symbols_with_timestamp(st.session_state.user['id'])
                                        
                                        if symbol in saved_symbols:
                                            print(f"✅ Attempt {attempt + 1}: {symbol} verified in database")
                                            
                                            # Update session state
                                            st.session_state.SYMBOLS = updated_symbols
                                            st.session_state.symbols_timestamp = datetime.now().isoformat()
                                            
                                            # Clear search
                                            st.session_state.search_query = ""
                                            st.session_state.show_suggestions = False
                                            
                                            st.success(f"✅ Added {symbol}!")
                                            return updated_symbols
                                        else:
                                            print(f"⚠️ Attempt {attempt + 1}: {symbol} not found in database, retrying...")
                                            time.sleep(0.1)
                                    else:
                                        print(f"❌ Attempt {attempt + 1}: Failed to save")
                                        time.sleep(0.1)
                                
                                st.error(f"❌ Failed to save {symbol} after 3 attempts!")
                                return current_symbols
                            else:
                                # Not authenticated
                                st.session_state.SYMBOLS = updated_symbols
                                st.session_state.search_query = ""
                                st.session_state.show_suggestions = False
                                st.success(f"✅ Added {symbol}!")
                                return updated_symbols
                    else:
                        st.markdown("✓")
        else:
            st.info("No symbols found")
    
    return current_symbols
