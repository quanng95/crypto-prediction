def get_custom_css():
    """Return custom CSS styling"""
    return """
<style>
    .main { 
        background-color: #1e1e1e;
        font-size: 16px !important;
    }
    
    /* Beautiful Header */
    .header-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 30px;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 30px;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
    }
    
    .header-title {
        color: #ffffff;
        font-size: 48px;
        font-weight: bold;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    /* Control price box */
    .control-price-box {
        background-color: #2d2d2d;
        padding: 10px;
        border-radius: 5px;
        margin-top: 0px;
        border: 1px solid #3d3d3d;
    }
    
    .control-symbol {
        color: #ffffff !important;
        font-weight: bold !important;
        font-size: 14px !important;
    }
    
    .control-price-up {
        color: #27ae60 !important;
        font-weight: bold;
        font-size: 28px;
    }
    
    .control-price-down {
        color: #e74c3c !important;
        font-weight: bold;
        font-size: 28px;
    }
    
    /* Trading signal boxes */
    .signal-box {
        background-color: #2d2d2d;
        padding: 20px;
        border-radius: 8px;
        margin: 10px 0;
        border: 2px solid #3d3d3d;
    }
    
    .signal-long {
        border-left: 4px solid #27ae60;
    }
    
    .signal-short {
        border-left: 4px solid #e74c3c;
    }
    
    .signal-neutral {
        border-left: 4px solid #95a5a6;
    }
    
    p, span, div, label {
        color: #e0e0e0 !important;
        font-size: 16px;
    }
    
    h1, h2, h3 {
        color: #ffffff !important;
    }
    
    h1 {
        font-size: 42px !important;
    }
    
    h2 {
        font-size: 36px !important;
    }
    
    h3 {
        font-size: 28px !important;
    }
    
    /* ẨN RUNNING INDICATOR */
    [data-testid="stStatusWidget"] {
        visibility: hidden;
        height: 0px;
        position: fixed;
    }
    
    /* ẨN HEADER STREAMLIT */
    header[data-testid="stHeader"] {
        display: none;
    }
    
    /* ẨN TOOLBAR */
    [data-testid="stToolbar"] {
        display: none;
    }
    
    /* ẨN FOOTER */
    footer {
        visibility: hidden;
        height: 0px;
    }
    
    /* ẨN MENU */
    #MainMenu {
        visibility: hidden;
    }
    
    /* ẨN DEPLOY BUTTON */
    .stDeployButton {
        display: none;
    }
    
    [data-testid="stMetricValue"] {
        font-size: 22px;
        color: #e0e0e0 !important;
    }
    
    [data-testid="stMetricLabel"] {
        color: #ffffff !important;
    }
    
    .stButton button {
        font-size: 15px;
    }
    
    .stSelectbox > div > div {
        background-color: #2d2d2d !important;
        color: #e0e0e0 !important;
    }
    
    [data-testid="stDataFrame"] {
        background-color: #2d2d2d;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        background-color: #2d2d2d;
    }
    
    .stTabs [data-baseweb="tab"] {
        color: #e0e0e0 !important;
    }
    
    .stTabs [aria-selected="true"] {
        color: #ffffff !important;
    }
    /* Auth Pages Styles */
    .auth-container {
        max-width: 500px;
        margin: 0 auto;
        padding: 20px;
    }
    
    input[type="text"], input[type="password"], input[type="email"] {
        background-color: #2d2d2d !important;
        border: 1px solid #3d3d3d !important;
        color: #e0e0e0 !important;
        border-radius: 5px !important;
        padding: 10px !important;
    }
    
    input[type="text"]:focus, input[type="password"]:focus, input[type="email"]:focus {
        border-color: #3498db !important;
        box-shadow: 0 0 5px rgba(52, 152, 219, 0.5) !important;
    }
</style>
"""
