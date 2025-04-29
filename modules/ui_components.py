import streamlit as st
import time
import base64
from typing import Callable, Optional, Any, Dict

def apply_retro_style() -> None:
    """Apply retro gaming CSS styling to the Streamlit app."""
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=VT323&family=Space+Mono&display=swap');

    /* Main retro styling */
    * {
        font-family: 'VT323', monospace;
    }
    
    h1, h2, h3 {
        font-family: 'VT323', monospace !important;
        color: #FFD700 !important; /* Golden yellow */
        text-shadow: 3px 3px 0px #FF6B6B; /* Coral shadow */
    }
    
    /* Code blocks */
    code {
        font-family: 'Space Mono', monospace !important;
        background-color: #000080 !important; /* Navy blue */
        color: #00FF00 !important; /* Green text */
        border: 2px solid #FFFFFF !important;
        padding: 5px !important;
    }
    
    /* Sidebar */
    .css-1d391kg {
        background-color: #000080 !important; /* Navy blue */
    }
    
    .css-1d391kg .block-container {
        border-right: 4px solid #FFD700 !important; /* Golden yellow */
    }
    
    /* Buttons */
    .stButton>button {
        font-family: 'VT323', monospace !important;
        background-color: #FF6B6B !important; /* Coral */
        color: white !important;
        border: 3px solid #FFD700 !important; /* Golden yellow */
        border-radius: 0px !important; /* Square borders for pixel look */
        box-shadow: 4px 4px 0px #000000 !important; /* Black shadow */
        transition: transform 0.1s, box-shadow 0.1s !important;
    }
    
    .stButton>button:hover {
        transform: translate(2px, 2px) !important;
        box-shadow: 2px 2px 0px #000000 !important; /* Reduced shadow on hover */
    }
    
    /* Progress bar */
    .stProgress .st-bo {
        background-color: #FF6B6B !important; /* Coral */
        height: 20px !important;
        border-radius: 0px !important; /* Square for pixel look */
    }
    
    .stProgress .st-bp {
        background-color: #FFD700 !important; /* Golden yellow */
        border-radius: 0px !important; /* Square for pixel look */
    }
    
    /* Dataframe styling */
    .dataframe {
        border: 3px solid #FFD700 !important; /* Golden yellow */
        font-family: 'Space Mono', monospace !important;
    }
    
    .dataframe th {
        background-color: #FF6B6B !important; /* Coral */
        color: white !important;
        border: 2px solid black !important;
    }
    
    .dataframe td {
        border: 2px solid black !important;
    }

    /* Pixel border effect for images */
    .pixel-border {
        border: 4px solid #FFD700 !important; /* Golden yellow */
        image-rendering: pixelated !important;
        padding: 0 !important;
        box-shadow: 8px 8px 0px #FF6B6B !important; /* Coral shadow */
    }

    /* Game message box */
    .game-message {
        background-color: #000080 !important; /* Navy blue */
        border: 4px solid #FFD700 !important; /* Golden yellow */
        color: white !important;
        padding: 10px !important;
        margin: 10px 0 !important;
        box-shadow: 5px 5px 0px #FF6B6B !important; /* Coral shadow */
    }
    
    /* Animated pixel text effect */
    @keyframes pixel-text {
        0% { text-shadow: 2px 2px 0px #FF6B6B; }
        50% { text-shadow: 3px 3px 0px #FF6B6B; }
        100% { text-shadow: 2px 2px 0px #FF6B6B; }
    }
    
    .pixel-text {
        animation: pixel-text 2s infinite;
    }

    /* Pixel-style select box */
    .stSelectbox label {
        color: #FFD700 !important;
        text-shadow: 2px 2px 0px #FF6B6B !important;
    }

    .stSelectbox > div > div {
        background-color: #000080 !important;
        border: 3px solid #FFD700 !important;
        border-radius: 0px !important;
        color: white !important;
    }

    /* Checkbox styling */
    .stCheckbox label {
        color: #FFD700 !important;
        text-shadow: 1px 1px 0px #FF6B6B !important;
    }

    .stCheckbox > div > div {
        background-color: #000080 !important;
        border: 2px solid #FFD700 !important;
        border-radius: 0px !important;
    }

    /* Radio button styling */
    .stRadio label {
        color: #FFD700 !important;
        text-shadow: 1px 1px 0px #FF6B6B !important;
    }

    /* Stats container */
    .stats-container {
        background-color: #000080 !important;
        padding: 10px !important;
        border: 3px solid #FFD700 !important;
        text-align: center !important;
    }

    .stats-value {
        font-size: 2.5em !important;
        color: white !important;
        font-family: "VT323", monospace !important;
        margin: 0 !important;
    }

    .stats-label {
        color: #FFD700 !important;
        margin: 0 !important;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }

    .stTabs [data-baseweb="tab"] {
        background-color: #000080 !important;
        border: 2px solid #FFD700 !important;
        border-radius: 0px !important;
        color: white !important;
        padding: 5px 15px !important;
    }

    .stTabs [aria-selected="true"] {
        background-color: #FF6B6B !important;
        border-bottom: 2px solid #FFD700 !important;
    }
    </style>
    """, unsafe_allow_html=True)

def pixel_title(title: str) -> None:
    """
    Display a pixel art title with animation.
    
    Args:
        title: Title text
    """
    st.markdown(f'<h1 class="pixel-text">{title}</h1>', unsafe_allow_html=True)

def game_message(message: str) -> None:
    """
    Display a game-style message box.
    
    Args:
        message: Message text
    """
    st.markdown(f'<div class="game-message">{message}</div>', unsafe_allow_html=True)

def pixel_progress_bar(progress: float) -> None:
    """
    Display a pixel art progress bar.
    
    Args:
        progress: Progress value (0.0 to 1.0)
    """
    st.progress(progress)
    if progress == 1.0:
        st.balloons()

def loading_animation(text: str = "LOADING", duration: float = 2.0) -> None:
    """
    Display a loading animation with dots.
    
    Args:
        text: Text to display before the dots
        duration: Duration of the animation in seconds
    """
    progress_text = st.empty()
    progress_bar = st.empty()
    
    start_time = time.time()
    
    while time.time() - start_time < duration:
        progress = min(1.0, (time.time() - start_time) / duration)
        dots = "." * (int(time.time() * 4) % 4)
        progress_text.markdown(f"<h3>{text}{dots}</h3>", unsafe_allow_html=True)
        progress_bar.progress(progress)
        time.sleep(0.05)
    
    progress_text.empty()
    progress_bar.empty()

def create_stat_box(title: str, value: Any, color: str = "#FFD700") -> None:
    """
    Create a retro-styled statistics box.
    
    Args:
        title: Title of the statistic
        value: Value to display
        color: Color for the title
    """
    st.markdown(f"""
    <div style='background-color: #000080; padding: 10px; border: 3px solid {color}; text-align: center;'>
        <h3 style='color: {color}; margin: 0;'>{title}</h3>
        <p style='font-size: 2.5em; color: white; font-family: "VT323", monospace; margin: 0;'>{value}</p>
    </div>
    """, unsafe_allow_html=True)

def arcade_button(label: str, key: str, help: str = "") -> bool:
    """
    Create an arcade-style button.
    
    Args:
        label: Button text
        key: Unique key for the button
        help: Help text
        
    Returns:
        True if button was clicked, False otherwise
    """
    return st.button(f"🎮 {label} 🎮", key=key, help=help)

def create_sidebar_controls() -> Dict[str, Any]:
    """
    Create the sidebar controls for the application.
    
    Returns:
        Dictionary containing the selected control values
    """
    st.sidebar.markdown("<h2 style='text-align:center; color:#FFD700; text-shadow: 2px 2px #FF6B6B;'>GAME CONTROLS</h2>", unsafe_allow_html=True)
    
    # Input options
    input_type = st.sidebar.radio(
        "SELECT YOUR INPUT:",
        ["Upload Image", "Camera", "Generate Synthetic Data"],
        index=2  # Default to synthetic for the MVP
    )
    
    # Anomaly detection options
    st.sidebar.markdown("<h3 style='color:#FFD700; text-shadow: 2px 2px #FF6B6B;'>ANOMALY TYPES</h3>", unsafe_allow_html=True)
    
    detect_misplaced = st.sidebar.checkbox("Misplaced Items", value=True)
    detect_damaged = st.sidebar.checkbox("Damaged Items", value=True)
    detect_duplicates = st.sidebar.checkbox("Duplicate Items", value=True)
    
    # Generate synthetic data options
    if input_type == "Generate Synthetic Data":
        st.sidebar.markdown("<h3 style='color:#FFD700; text-shadow: 2px 2px #FF6B6B;'>SYNTHETIC DATA</h3>", unsafe_allow_html=True)
        
        add_anomaly = st.sidebar.checkbox("Add Anomaly", value=True)
        
        if add_anomaly:
            anomaly_type = st.sidebar.selectbox(
                "Anomaly Type",
                ["misplaced", "damaged", "duplicate"]
            )
        else:
            anomaly_type = None
    else:
        add_anomaly = False
        anomaly_type = None
    
    # Start button - styled as a game start button
    start_button = st.sidebar.button("🎮 START SCAN 🎮")
    
    return {
        "input_type": input_type,
        "detect_misplaced": detect_misplaced,
        "detect_damaged": detect_damaged,
        "detect_duplicates": detect_duplicates,
        "add_anomaly": add_anomaly,
        "anomaly_type": anomaly_type,
        "start_button": start_button
    }

def show_mission_briefing() -> None:
    """Display the mission briefing in retro game style."""
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("<h3 style='color:#FFD700; text-shadow: 2px 2px #FF6B6B;'>MISSION BRIEFING</h3>", unsafe_allow_html=True)
        st.markdown("""
        <div style='font-family: VT323, monospace; font-size: 1.2em;'>
        Your retail shelves need constant monitoring to ensure promotional materials
        are correctly placed and in good condition. This AI-powered system will:
        
        - 🔍 Detect and classify promotional items
        - ⚠️ Identify misplaced materials
        - 🔧 Spot damaged or incorrectly installed items
        - 🔄 Find duplicate materials too close together
        
        Configure your scan parameters using the game controls on the left,
        then press START to begin your shelf integrity analysis!
        </div>
        """, unsafe_allow_html=True)
    
    return col2

def add_sound_effect(sound_type: str = "success") -> None:
    """
    Add a sound effect to the UI.
    
    Args:
        sound_type: Type of sound ("success" or "alert")
    """
    if sound_type == "success":
        st.markdown("""
        <audio autoplay>
          <source src="data:audio/wav;base64,UklGRjQnAABXQVZFZm10IBAAAAABAAEARKwAAESsAAABAAgAZGF0YRAnAAAAAAADAwMEBQQGBwgICQgKCwsMDQ4PEBESFBUXGBkaHB0eICIjJCYnKSorLS4wMTM0Njc5Ojs9Pj9BQkNFRkdISUpLTElKS0xNTk9QUVJTVFVWV1haW1xdXl9gYWJjZGVmZ2hpamtsbW5vcHFyc3N0dXZ3eHl6ent8fH19fn+AgYKDhIWG"
          type="audio/wav">
        </audio>
        """, unsafe_allow_html=True)
    elif sound_type == "alert":
        st.markdown("""
        <audio autoplay>
          <source src="data:audio/wav;base64,UklGRjQnAABXQVZFZm10IBAAAAABAAEARKwAAESsAAABAAgAZGF0YRAnAAAAAAEBAQIDAwQFBgcICQsMDQ8REhQWGBkbHR8hIyUnKSssLjAzNTc5PD5AQUNGRURJTE5RU1ZYW11fYmRnaWxucXN2eHt9gIKFiImCBQUHCAoLDQ8QEhQWGBocHiAiJCYoKy0vMTM2ODtOgkxvdG9SSUZEN3xLNTR2biMjUVBCPzUhEnhOd04zU3cTNzd2TjciEkhOAg=="
          type="audio/wav">
        </audio>
        """, unsafe_allow_html=True)