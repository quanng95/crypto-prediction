import streamlit as st
import requests
from typing import List, Dict
from database_postgres import Database

class SymbolManager:
    """Symbol manager with Spot and Future support"""
    
    @staticmethod
    @st.cache_data(ttl=3600)
    def fetch_all_binance_symbols():
        """Fetch both Spot (USDT) and Future (PERP) symbols"""
        symbols = []
        
        try:
            # Fetch Spot symbols
            spot_url = "https://api.binance.com/api/v3/exchangeInfo"
            spot_response = requests.get(spot_url, timeout=10)
            spot_data = spot_response.json()
            
            for symbol_info in spot_data['symbols']:
                if (symbol_info['status'] == 'TRADING' and 
                    symbol_info['quoteAsset'] == 'USDT'):
                    symbols.append({
                        'symbol': symbol_info['symbol'],
                        'baseAsset': symbol_info['baseAsset'],
                        'type': 'SPOT'
                    })
            
            # Fetch Future symbols (PERP)
            future_url = "https://fapi.binance.com/fapi/v1/exchangeInfo"
            future_response = requests.get(future_url, timeout=10)
            future_data = future_response.json()
            
            for symbol_info in future_data['symbols']:
                if (symbol_info['status'] == 'TRADING' and 
                    symbol_info['contractType'] == 'PERPETUAL'):
                    symbols.append({
                        'symbol': symbol_info['symbol'],
                        'baseAsset': symbol_info['baseAsset'],
                        'type': 'FUTURE'
                    })
            
            return symbols
            
        except Exception as e:
            print(f"Error fetching symbols: {e}")
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
                
                if len(results) >= 15:  # Increased limit
                    break
        
        return results

def render_simple_add_symbol(current_symbols: List[str]) -> List[str]:
    """
    Simple symbol add interface with Spot/Future support
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
        "🔍 Add Symbol (Spot/Future)",
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
                symbol_type = result['type']
                
                col1, col2, col3 = st.columns([3, 1, 1])
                
                with col1:
                    type_emoji = "📊" if symbol_type == "SPOT" else "📈"
                    st.markdown(f"{type_emoji} **{symbol}** ({base})")
                
                with col2:
                    st.markdown(f"*{symbol_type}*")
                
                with col3:
                    if symbol not in current_symbols:
                        if st.button("➕", key=f"add_{symbol}", use_container_width=True):
                            current_symbols.append(symbol)
                            
                            # Auto-save to database if authenticated
                            if st.session_state.get('authenticated', False):
                                db = Database()
                                db.save_user_symbols(
                                    st.session_state.user['id'], 
                                    current_symbols
                                )
                                print(f"✅ Symbols auto-saved")
                            
                            # Clear search and hide suggestions
                            st.session_state.search_query = ""
                            st.session_state.show_suggestions = False
                            
                            st.success(f"✅ Added {symbol}!")
                            st.rerun()
        else:
            st.info("No symbols found")
    
    return current_symbols
