import streamlit as st
import requests
from PIL import Image
from io import BytesIO
import json
import logging
import base64

# --- Configuration and Setup ---
st.set_page_config(page_title="Vision.ai - Image Captioning", page_icon="👁️", layout="wide")

# Setup console logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# URL for the FastAPI backend (update if necessary)
API_URL = "http://127.0.0.1:8000/caption"  # Ensure this is correct

AVAILABLE_LANGUAGES = {
    "English": "en",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Russian": "ru",
    "Arabic": "ar",
    # Add more based on server's 'Helsinki-NLP/opus-mt-en-<code>' model availability
}

# --- Initialize Session State ---
if "navigation_radio" not in st.session_state:
    st.session_state.navigation_radio = "Home"
if "caption_data" not in st.session_state:
    st.session_state.caption_data = None
if "uploaded_image_bytes" not in st.session_state:
    st.session_state.uploaded_image_bytes = None
if "uploaded_image_info" not in st.session_state:
    st.session_state.uploaded_image_info = None


# --- Custom CSS for Enhanced Styling ---
st.markdown("""
<style>
/* General body styling */
body {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    color: #2A4759; /* Default dark text for main content */
    background-color: #F0F2F5; /* Light Grey page background */
    line-height: 1.6;
}

/* Sidebar styling */
[data-testid="stSidebar"] {
    background-color: #D4C9BE; /* White sidebar background */
    border-right: 1px solid #D1D9E1; /* Subtle border */
    padding: 25px 20px !important;
}

/* === FIX FOR SIDEBAR TEXT VISIBILITY === */
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] h4,
[data-testid="stSidebar"] .stMarkdown strong {
    color: #1E293B !important; /* Dark Slate Gray */
    font-weight: 600;
}
[data-testid="stSidebar"] .stRadio > label > div {
    font-size: 1.2rem !important;
    font-weight: bold;
    color: #1E293B !important; /* Dark Slate Gray */
    margin-bottom: 12px !important;
}
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label[data-baseweb="radio"] > div:last-child span {
    color: #2A4759 !important; /* Dark Slate Gray */
    font-size: 1.05rem;
    font-weight: 500;
}
[data-testid="stSidebar"] .stMarkdown p {
    color: #333333 !important; /* Dark Gray */
}
/* === END OF FIX FOR SIDEBAR TEXT VISIBILITY === */

/* Main content area */
section.main .block-container {
    /* background-color: #2A4759; */ 
    background-color: #FFFFFF; /* Ensuring content visibility */
    padding: 2rem 2.5rem;
    border-radius: 10px; 
    box-shadow: 0 3px 10px rgba(0,0,0,0.05); 
}

/* Header styling for main content */
h1, h2, h3 {
    color: #2A4759;
    font-weight: 700;
    margin-bottom: 0.75rem;
}
h1 { font-size: 2.3rem; margin-top: 0.5rem; }
h2 { font-size: 1.8rem; margin-top: 1.5rem; }
h3 { font-size: 1.5rem; margin-top: 1.25rem; }

/* Card-like styling for sections using st.container() */
div.stContainer { /* Targeting custom containers */
    /* background-color: #2A4759; */ 
    background-color: #FFFFFF;  /* Ensuring content visibility */
    padding: 25px;
    border-radius: 10px;
    box-shadow: 0 5px 15px rgba(0, 0, 0, 0.08);
    margin-bottom: 25px;
    border: 1px solid #E7E9EC;
}
/* Styling specifically for the product page's main configuration and results containers */
.product-section-container {
    background-color: #F8F9FA; 
    padding: 20px;
    border-radius: 8px;
    margin-bottom: 20px;
    border: 1px solid #E0E0E0;
}

/* Improve spacing for general markdown and elements in main content */
.stMarkdown, .stVerticalBlock > div:first-child {
    margin-bottom: 1.2rem;
}

/* Style the sidebar navigation radio buttons (non-text parts) */
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label {
    padding: 10px 8px !important;
    border-radius: 8px;
    transition: background-color 0.2s ease-in-out;
    margin-bottom: 5px;
}
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label:hover {
    background-color: #F0F2F5 !important;
}
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label[data-baseweb="radio"] > div:first-child {
     border-color: #F79B72 !important;
}
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label[data-baseweb="radio"][aria-checked="true"] {
    background-color: #FFF0E6 !important;
}
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label[data-baseweb="radio"][aria-checked="true"] > div:first-child {
     background-color: #F79B72 !important;
}

/* Animated header in main content */
.animated-header h1 {
    animation: fadeInUp 1s ease-out;
    opacity: 0;
    animation-fill-mode: forwards;
}
@keyframes fadeInUp {
    0% { opacity: 0; transform: translateY(20px); }
    100% { opacity: 1; transform: translateY(0); }
}

/* Style for the file uploader in main content */
.stFileUploader > label {
    color: #2A4759 !important;
    font-weight: bold !important;
    font-size: 1.1rem !important; 
    margin-bottom: 0.5rem;
}
.stFileUploader > div > div { /* The drag and drop area */
    border-radius: 8px;
    border: 2px dashed #B0B8C0;
    background-color: #F8F9FA;
    padding: 25px;
}
.stFileUploader > div > div:hover {
    border-color: #F79B72;
    background-color: #FEFBF9;
}

/* Style for buttons in main content */
.stButton button {
    background-color: #F79B72;
    color: white;
    border-radius: 8px;
    padding: 10px 24px; 
    font-size: 1rem;   
    font-weight: 600;
    border: none;
    transition: background-color 0.3s ease, transform 0.1s ease, box-shadow 0.2s ease;
    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
}
.stButton button:hover {
    background-color: #E37A50;
    color: white;
    transform: translateY(-2px);
    box-shadow: 0 4px 10px rgba(0,0,0,0.15);
}
.stButton button:active {
    transform: translateY(0px);
    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
}
.stButton button:disabled {
    background-color: #D1D9E1;
    color: #6c757d;
    box-shadow: none;
    transform: none;
}

/* Placeholder styling for empty areas if needed */
.custom-placeholder {
    background-color: #F8F9FA; padding: 30px 20px; border-radius: 8px;
    border: 1px dashed #D1D9E1; min-height: 100px; display: flex;
    align-items: center; justify-content: center; text-align: center; color: #6c757d;
    font-size: 1.1rem;
}

/* Image styling in main content */
.stImage img {
    border-radius: 8px; box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
    border: 1px solid #E0E0E0;
    max-height: 400px; 
    object-fit: contain; 
}

/* Footer in sidebar */
.sidebar-footer {
    font-size: 0.9rem; color: #6c757d !important;
    padding-top: 20px; border-top: 1px solid #E0E0E0; margin-top: 25px; text-align: center;
}

/* Ensure main content text is visible */
.main .block-container .stMarkdown p,
.main .block-container h1,
.main .block-container h2,
.main .block-container h3,
.main .block-container h4,
.main .block-container h5,
.main .block-container h6,
.main .block-container .stText, /* For st.text if used */
.main .block-container label, /* For widget labels */
.main .block-container .stSelectbox > div, /* For selectbox text */
.main .block-container .stSubheader { /* For st.subheader */
    color: #2A4759 !important; 
}

/* Styling for caption text within the caption boxes */
.caption-box {
    background-color:#2A4759; 
    color: #F0F2F5; 
    padding:10px;
    border-radius:5px;
    margin-bottom:10px;
}

</style>
""", unsafe_allow_html=True)


# --- Page Functions ---

def go_to_product_page_callback():
    st.session_state.navigation_radio = "Product"
    # st.rerun() # Removed: Changing navigation_radio should trigger a rerun automatically

def home_page():
    """Renders the Home Page content."""
    st.markdown('<div class="animated-header"><h1>Welcome to Vision.ai</h1></div>', unsafe_allow_html=True)
    st.markdown("##### Your gateway to cutting-edge AI image analysis and understanding.")

    with st.container():
        st.header("🚀 About Vision.ai")
        st.write("""
            Vision.ai leverages state-of-the-art deep learning models to understand and describe the content of images.
            Our powerful image captioning tool can automatically generate human-like descriptions, making images
            more accessible and searchable. Explore the future of visual AI with us.
        """)
        st.write("Whether you need automated alt text for accessibility, content analysis, or creative descriptions, Vision.ai provides accurate and relevant captions.")

    with st.container():
        st.header("✨ Key Features")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("🎯 Accurate Captioning")
            st.write("Generate precise and contextually relevant descriptions for your images.")
            st.subheader("🌍 Multilingual Support")
            st.write(f"Translate captions into various languages (server model dependent, {len(AVAILABLE_LANGUAGES)-1}+ available).")
        with col2:
            st.subheader("⚡ Fast Processing")
            st.write("Get results quickly thanks to optimized models and infrastructure.")
            st.subheader("🔧 Easy Integration")
            st.write("Simple API endpoint for seamless integration into your workflows.")

    with st.container():
        st.header("🛠️ How It Works")
        st.write("""
            Our system employs a sophisticated pipeline:
            1.  **Image Preprocessing:** Input images are standardized for optimal model performance.
            2.  **Vision Transformer (ViT):** A powerful model extracts rich visual features from the image.
            3.  **Language Model (e.g., GPT-2 variant):** These features are then fed into a language model that generates descriptive captions.
            4.  **Translation (Optional):** Captions can be translated using advanced neural machine translation models (e.g., Helsinki-NLP).
            5.  **Text-to-Speech (Server-Side, Optional):** Generated captions can be converted to speech by the server.
            """)

    with st.container():
        st.header("💡 Ready to Try?")
        st.write("Navigate to the **Product** section in the sidebar or click the button below to use the Vision.ai caption generator!")
        st.button("Go to Product Page", type="primary", on_click=go_to_product_page_callback, use_container_width=True)


def vision_ai_page():
    """Renders the Vision.ai Product Page content."""
    st.title("Vision.ai Caption Generator 👁️‍🗨️") 
    st.markdown("Upload an image to generate an English caption, optionally translate it, and hear it spoken if available.")

    target_lang_code = AVAILABLE_LANGUAGES["English"] # Default

    with st.container(): 
        st.markdown('<div class="product-section-container">', unsafe_allow_html=True) 
        st.subheader("⚙️ Configuration & Upload")

        cols = st.columns([0.6, 0.4]) # Adjusted column ratio for better layout if needed
        with cols[0]:
            st.markdown("**Target Language for Translation:**")
            target_lang_name = st.selectbox(
                "Select language:",
                list(AVAILABLE_LANGUAGES.keys()),
                index=list(AVAILABLE_LANGUAGES.keys()).index("English"), 
                label_visibility="collapsed",
                key="target_language_select"
            )
            target_lang_code = AVAILABLE_LANGUAGES[target_lang_name]

        # Display info messages outside the columns if they apply generally
        if target_lang_code != 'en':
            st.info( 
                f"Translation to **{target_lang_name} ({target_lang_code})** uses Helsinki-NLP/OPUS-MT models. Availability depends on the server.",
                icon="ℹ️"
            )
        
        st.info( 
            "Server-side Text-to-Speech (TTS) for captions depends on the server's TTS setup and language support.",
            icon="🔊"
        )

        uploaded_file = st.file_uploader(
            "Click to browse or drag and drop an image here.",
            type=["jpg", "png", "jpeg"],
            label_visibility="visible", 
            key="file_uploader_key" 
        )
        st.markdown('</div>', unsafe_allow_html=True) 


    if uploaded_file is not None:
        image_bytes = uploaded_file.getvalue()
        if st.session_state.get("uploaded_image_bytes") != image_bytes: 
            st.session_state.uploaded_image_bytes = image_bytes
            st.session_state.uploaded_image_info = {
                "name": uploaded_file.name,
                "type": uploaded_file.type,
            }
            st.session_state.caption_data = None # Reset caption data if new image is uploaded

    action_cols = st.columns([0.7, 0.3])
    with action_cols[0]:
        generate_disabled = st.session_state.uploaded_image_bytes is None
        if st.button("✨ Generate Caption & Translate", type="primary", use_container_width=True, disabled=generate_disabled):
            if st.session_state.uploaded_image_bytes:
                with st.spinner("Vision.ai is thinking... Please wait. 🤔"):
                    files = {"file": (
                        st.session_state.uploaded_image_info["name"],
                        st.session_state.uploaded_image_bytes,
                        st.session_state.uploaded_image_info["type"]
                    )}
                    # Ensure target_lang_code is correctly scoped from the selectbox
                    # This will use the latest value from the selectbox when the button is clicked
                    current_target_lang_code = AVAILABLE_LANGUAGES[st.session_state.target_language_select]
                    data_payload = {"target_language": current_target_lang_code}
                    
                    generic_ui_error_message = "An error occurred. Please check the application logs or contact support if the issue persists."

                    try:
                        response = requests.post(API_URL, files=files, data=data_payload, timeout=30) 
                        if response.status_code == 200:
                            st.session_state.caption_data = response.json()
                            st.success("Caption generated successfully!", icon="✅") 
                        else:
                            st.session_state.caption_data = None 
                            logger.error(f"Server Error: {response.status_code} - Response: {response.text}")
                            st.error(generic_ui_error_message, icon="❌") 
                            try:
                                error_details = response.json()
                                logger.error(f"Server Error Details: {error_details.get('error', 'No specific error provided by server.')}")
                            except json.JSONDecodeError:
                                logger.error("Server returned an error but the response was not valid JSON.")
                    except requests.exceptions.ConnectionError as e:
                        st.session_state.caption_data = None
                        logger.error(f"Connection Error: Could not connect to Vision.ai server at {API_URL}. Details: {e}")
                        st.error(generic_ui_error_message, icon="🌐") 
                    except requests.exceptions.Timeout as e:
                        st.session_state.caption_data = None
                        logger.error(f"Connection Timeout: The server took too long to respond. Details: {e}")
                        st.error(generic_ui_error_message, icon="⏳") 
                    except Exception as e:
                        st.session_state.caption_data = None
                        logger.error(f"An unexpected error occurred: {str(e)}", exc_info=True)
                        st.error(generic_ui_error_message, icon="🔥")
                # st.rerun() # Removed: Modifying session_state should trigger rerun
                # Instead of st.rerun(), if an immediate update is needed post-button press and
                # session state isn't enough, consider restructuring or using st.experimental_rerun()
                # cautiously. But usually, direct state modifications are preferred.
                # For now, removing it to see if default behavior is sufficient.

    with action_cols[1]:
        clear_disabled = st.session_state.uploaded_image_bytes is None and st.session_state.caption_data is None
        if st.button("🗑️ Clear All", use_container_width=True, disabled=clear_disabled):
            st.session_state.uploaded_image_bytes = None
            st.session_state.uploaded_image_info = None
            st.session_state.caption_data = None
            # Clear the uploader state by resetting its key if necessary
            if 'file_uploader_key' in st.session_state: # Check if the key exists
                st.session_state.file_uploader_key = None # Or use a unique value like str(uuid.uuid4())
            # st.rerun() # Removed: Modifying session_state should trigger rerun

    if st.session_state.uploaded_image_bytes:
        display_cols = st.columns([0.5, 0.5])
        with display_cols[0]:
            st.subheader("🖼️ Uploaded Image")
            try:
                image = Image.open(BytesIO(st.session_state.uploaded_image_bytes))
                # MODIFIED HERE: use_column_width changed to use_container_width
                st.image(image, use_container_width=True)
            except Exception as e:
                logger.error(f"Could not display image: {e}", exc_info=True)
                st.markdown("<div class='custom-placeholder'><p>Could not load image preview.</p></div>", unsafe_allow_html=True)
        
        with display_cols[1]:
            if st.session_state.caption_data:
                st.subheader("📜 Generated Captions & Audio") 
                with st.container(): 
                    data = st.session_state.caption_data
                    original_caption = data.get("original_english_caption", "N/A")
                    translated_caption = data.get("translated_caption", "N/A")
                    requested_lang_code = data.get("target_language_requested", "N/A")
                    translation_status = data.get("translation_status", "N/A")
                    translation_error = data.get("translation_error", "")

                    if original_caption and original_caption != "N/A":
                        st.markdown(f"**Original English Caption:**")
                        st.markdown(f"<div class='caption-box'>{original_caption}</div>", unsafe_allow_html=True)
                        
                        if data.get("original_english_audio_url"):
                            st.audio(data["original_english_audio_url"])
                        elif data.get("original_english_audio_b64"):
                            try:
                                audio_bytes = base64.b64decode(data["original_english_audio_b64"])
                                st.audio(audio_bytes) 
                            except Exception as e:
                                logger.error(f"Error decoding/playing original_english_audio_b64: {e}")
                        
                    if requested_lang_code != 'en' and requested_lang_code != "N/A":
                        lang_name_display = next(
                            (name for name, code in AVAILABLE_LANGUAGES.items() if code == requested_lang_code),
                            requested_lang_code.upper()
                        )
                        if translation_status == "success" and translated_caption and translated_caption != "N/A":
                            st.markdown(f"**Translated Caption ({lang_name_display}):**")
                            st.markdown(f"<div class='caption-box'>{translated_caption}</div>", unsafe_allow_html=True)

                            if data.get("translated_audio_url"):
                                st.audio(data["translated_audio_url"])
                            elif data.get("translated_audio_b64"):
                                try:
                                    audio_bytes = base64.b64decode(data["translated_audio_b64"])
                                    st.audio(audio_bytes)
                                except Exception as e:
                                    logger.error(f"Error decoding/playing translated_audio_b64 for {lang_name_display}: {e}")
                                    
                        elif translation_status == "failure":
                            logger.warning(f"Translation to {lang_name_display} failed. Server Details: {translation_error}. Original caption: {original_caption}")
                            # Display a message to the user about translation failure
                            st.warning(f"Translation to {lang_name_display} was not successful. {translation_error}", icon="⚠️")
            
            # This part is tricky with auto-reruns. If caption_data is None *after* a generation attempt
            # (due to error), an error message (st.error) would have already been displayed.
            # This placeholder might appear briefly or be redundant if an error message from the button click is present.
            elif st.session_state.uploaded_image_bytes and not st.session_state.caption_data:
                # Check if an error message from the button click might already be displayed.
                # This requires a more complex state to avoid double messaging (e.g. success/error already shown)
                # For now, let's keep it simple.
                if not st.session_state.get("_st_error_messages"): # Heuristic to check for existing error messages
                     st.markdown("<div class='custom-placeholder'><p>Click 'Generate Caption & Translate' to see the results!</p></div>", unsafe_allow_html=True)

    elif not st.session_state.uploaded_image_bytes: 
        st.markdown("<div class='custom-placeholder'><p>Upload an image and click 'Generate' to see the magic! ✨</p></div>", unsafe_allow_html=True)


# --- Main App Logic (Navigation) ---
st.sidebar.markdown("## 🧭 Vision.ai Navigation")
page_options = ["Home", "Product"]

# The st.sidebar.radio widget will use st.session_state.navigation_radio
# Changing its value (either by user interaction or programmatically like in go_to_product_page_callback)
# will cause Streamlit to rerun the script.
st.sidebar.radio(
    "Go to:",
    page_options,
    key="navigation_radio", 
)

# This was a bit redundant as the page display logic is below.
# st.sidebar.markdown("### <nobr>🛍️ Vision.ai Captioner</nobr>", unsafe_allow_html=True)
# if page == "Product": # 'page' variable was from st.sidebar.radio but its value is already in st.session_state.navigation_radio
#     pass


# --- Display Selected Page ---
if st.session_state.navigation_radio == "Home":
    home_page()
elif st.session_state.navigation_radio == "Product":
    vision_ai_page()

# --- Footer ---
st.sidebar.markdown("---")
st.sidebar.markdown(
    '<div class="sidebar-footer">© 2025 Vision.ai<br>Advancing Visual Intelligence</div>', 
    unsafe_allow_html=True
)