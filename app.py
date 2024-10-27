import streamlit as st
from typing import List, Dict, Any, Optional
import asyncio
from PIL import Image
import io
from core import (
    ChatConfig, 
    ChatLogger, 
    ChatMemory, 
    QuestionGenerator, 
    GeminiRAG, 
    ProductDatabase, 
    Language,
    UserManager,
    UserInfo
)

# UI Text translations
UI_TEXT = {
    Language.ENGLISH: {
        "title": "🌱 Product Assistant",
        "welcome_message": """
        👋 Welcome! I'm your product expert. I can help you learn about:
        - Product benefits and features
        - Application methods and timing
        - Dosage recommendations
        - Crop compatibility
        - Technical specifications
        - Analysis of crop images and issues
        
        Choose a question below or ask your own!
        """,
        "input_placeholder": "Type your question here...",
        "input_label": "Ask me anything about product:",
        "clear_chat": "Clear Chat",
        "language_selector": "Select Language",
        "sidebar_title": "User Information",
        "form_name": "Your Name",
        "form_mobile": "Mobile Number",
        "form_location": "Location",
        "form_purchase": "Have you purchased product?",
        "form_crop": "What crop are you growing?",
        "form_submit": "Save Information",
        "form_success": "✅ Information saved successfully!",
        "form_error": "❌ Error saving information. Please try again.",
        "form_required": "Please fill in all required fields.",
        "image_upload": "Upload an image (optional)",
        "image_helper": "Upload an image of your crop if you have any concerns or questions about it.",
        "image_processing": "Processing your image...",
        "initial_questions": [
            "What are the main benefits of product?",
            "How do I apply product correctly?",
            "Which crops is product suitable for?",
            "What is the recommended dosage?"
        ]
    },
    Language.HINDI: {
        "title": "🌱 GAPL स्टार्टर उत्पाद सहायक",
        "welcome_message": """
        👋 नमस्ते! मैं आपका GAPL स्टार्टर उत्पाद विशेषज्ञ हूं। मैं आपको इन विषयों में मदद कर सकता हूं:
        - उत्पाद के लाभ और विशेषताएं
        - प्रयोग विधि और समय
        - खुराक की सिफारिशें
        - फसल अनुकूलता
        - तकनीकी विवरण
        - फसल की छवियों और समस्याओं का विश्लेषण
        
        नीचे दिए गए प्रश्नों में से चुनें या अपना प्रश्न पूछें!
        """,
        "input_placeholder": "अपना प्रश्न यहां टाइप करें...",
        "input_label": "GAPL स्टार्टर के बारे में कुछ भी पूछें:",
        "clear_chat": "चैट साफ़ करें",
        "language_selector": "भाषा चुनें",
        "sidebar_title": "उपयोगकर्ता जानकारी",
        "form_name": "आपका नाम",
        "form_mobile": "मोबाइल नंबर",
        "form_location": "स्थान",
        "form_purchase": "क्या आपने GAPL स्टार्टर खरीदा है?",
        "form_crop": "आप कौन सी फसल उगा रहे हैं?",
        "form_submit": "जानकारी सहेजें",
        "form_success": "✅ जानकारी सफलतापूर्वक सहेजी गई!",
        "form_error": "❌ जानकारी सहेजने में त्रुटि। कृपया पुनः प्रयास करें।",
        "form_required": "कृपया सभी आवश्यक फ़ील्ड भरें।",
        "image_upload": "छवि अपलोड करें (वैकल्पिक)",
        "image_helper": "यदि आपके पास अपनी फसल के बारे में कोई चिंता या प्रश्न है, तो उसकी छवि अपलोड करें।",
        "image_processing": "आपकी छवि प्र्रोसेस की जा रही है...",
        "initial_questions": [
            "GAPL स्टार्टर के मुख्य लाभ क्या हैं?",
            "GAPL स्टार्टर का प्रयोग कैसे करें?",
            "GAPL स्टार्टर किन फसलों के लिए उपयुक्त है?",
            "अनुशंसित मात्रा क्या है?"
        ]
    }
}

# Initialize session state
if 'chat_memory' not in st.session_state:
    st.session_state.chat_memory = ChatMemory()
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'message_counter' not in st.session_state:
    st.session_state.message_counter = 0
if 'submitted_question' not in st.session_state:
    st.session_state.submitted_question = None
if 'language' not in st.session_state:
    st.session_state.language = Language.ENGLISH
if 'user_info' not in st.session_state:
    st.session_state.user_info = None

# Configure the page
st.set_page_config(
    page_title="Product Assistant",
    page_icon="🌱",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
.user-message {
    background-color: #72BF6A;
    color: black;
    padding: 15px;
    border-radius: 15px;
    margin: 10px 0;
}
.assistant-message {
    background-color: #000000;
    color: #98FB98;
    padding: 15px;
    border-radius: 15px;
    margin: 10px 0;
}
.stButton > button {
    background-color: #212B2A;
    color: white;
    border: none;
    padding: 10px 15px;
    border-radius: 5px;
    transition: background-color 0.3s;
}
.stButton > button:hover {
    background-color: #45a049;
}
.language-selector {
    position: fixed;
    top: 1rem;
    right: 1rem;
    z-index: 1000;
}
.image-upload {
    margin: 20px 0;
    padding: 15px;
    border-radius: 10px;
    border: 2px dashed #72BF6A;
}
</style>
""", unsafe_allow_html=True)

# Initialize components
@st.cache_resource
def initialize_components():
    config = ChatConfig()
    logger = ChatLogger(config.log_file)
    question_gen = QuestionGenerator(config.gemini_api_key)
    rag = GeminiRAG(config.gemini_api_key)
    db = ProductDatabase(config)
    user_manager = UserManager(config.user_data_file)
    return config, logger, question_gen, rag, db, user_manager

config, logger, question_gen, rag, db, user_manager = initialize_components()

# Load product database
@st.cache_resource
def load_database():
    with open("STARTER.md", "r", encoding="utf-8") as f:
        markdown_content = f.read()
    db.process_markdown(markdown_content)

try:
    load_database()
except Exception as e:
    st.error(f"Error loading database: {str(e)}")

async def process_question(question: str, image: Optional[bytes] = None):
    try:
        relevant_docs = db.search(question)
        context = rag.create_context(relevant_docs)
        answer = await rag.get_answer(
            question, 
            context, 
            st.session_state.language,
            st.session_state.user_info,
            image
        )
        follow_up_questions = await question_gen.generate_questions(
            question, 
            answer, 
            st.session_state.language,
            st.session_state.user_info
        )
        
        st.session_state.chat_memory.add_interaction(question, answer)
        logger.log_interaction(
            question, 
            answer, 
            st.session_state.language,
            st.session_state.user_info
        )
        
        st.session_state.message_counter += 1
        
        message_content = {
            "text": question,
            "has_image": image is not None
        }
        
        st.session_state.messages.append({
            "role": "user",
            "content": message_content,
            "message_id": st.session_state.message_counter
        })
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "questions": follow_up_questions,
            "message_id": st.session_state.message_counter
        })
    except Exception as e:
        st.error(f"Error processing question: {str(e)}")

def handle_submit():
    if st.session_state.user_input:
        st.session_state.submitted_question = st.session_state.user_input
        st.session_state.user_input = ""

def handle_language_change():
    # Clear chat when language changes
    st.session_state.messages = []
    st.session_state.chat_memory.clear_history()
    st.session_state.message_counter = 0
    st.rerun()

def render_user_form():
    """Render the user information form in the sidebar"""
    current_text = UI_TEXT[st.session_state.language]
    
    st.sidebar.title(current_text["sidebar_title"])
    
    with st.sidebar.form("user_info_form"):
        name = st.text_input(current_text["form_name"])
        mobile = st.text_input(current_text["form_mobile"])
        location = st.text_input(current_text["form_location"])
        has_purchased = st.checkbox(current_text["form_purchase"])
        crop_type = st.text_input(current_text["form_crop"])
        
        submitted = st.form_submit_button(current_text["form_submit"])
        
        if submitted:
            if name and mobile and location and crop_type:
                user_info = UserInfo(
                    name=name,
                    mobile=mobile,
                    location=location,
                    has_purchased=has_purchased,
                    crop_type=crop_type
                )
                
                if user_manager.save_user_info(user_info):
                    st.session_state.user_info = user_info
                    st.sidebar.success(current_text["form_success"])
                else:
                    st.sidebar.error(current_text["form_error"])
            else:
                st.sidebar.warning(current_text["form_required"])

def display_chat_message(message: Dict[str, Any]):
    """Display a chat message with proper formatting"""
    if message["role"] == "user":
        if isinstance(message["content"], dict):
            # Message with possible image
            st.markdown(
                f'<div class="user-message">👤 {message["content"]["text"]}</div>',
                unsafe_allow_html=True
            )
            if message["content"]["has_image"]:
                st.markdown(
                    '<div class="user-message">📷 Image uploaded</div>',
                    unsafe_allow_html=True
                )
        else:
            # Legacy message format
            st.markdown(
                f'<div class="user-message">👤 {message["content"]}</div>',
                unsafe_allow_html=True
            )
    else:
        st.markdown(
            f'<div class="assistant-message">🌱 {message["content"]}</div>',
            unsafe_allow_html=True
        )
        
        if message.get("questions"):
            cols = st.columns(2)
            for i, question in enumerate(message["questions"]):
                if cols[i % 2].button(
                    question,
                    key=f"followup_{message['message_id']}_{i}",
                    use_container_width=True
                ):
                    asyncio.run(process_question(question))

def main():
    # Language selector
    with st.container():
        cols = st.columns([3, 1])
        with cols[1]:
            selected_language = st.selectbox(
                UI_TEXT[st.session_state.language]["language_selector"],
                options=[Language.ENGLISH, Language.HINDI],
                format_func=lambda x: "English" if x == Language.ENGLISH else "हिंदी",
                key="language_selector",
                index=0 if st.session_state.language == Language.ENGLISH else 1,
                on_change=handle_language_change
            )
            st.session_state.language = selected_language

    # Render user form in sidebar
    render_user_form()

    current_text = UI_TEXT[st.session_state.language]
    
    st.title(current_text["title"])
    
    # Welcome message
    if not st.session_state.messages:
        st.markdown(current_text["welcome_message"])
        
        # Display initial questions as buttons
        cols = st.columns(2)
        for i, question in enumerate(current_text["initial_questions"]):
            if cols[i % 2].button(question, key=f"initial_{i}", use_container_width=True):
                asyncio.run(process_question(question))
    
    # Display chat history
    for message in st.session_state.messages:
        display_chat_message(message)
    
    # Input area with image upload
    with st.container():
        # Add image upload
        with st.expander(current_text["image_upload"], expanded=False):
            uploaded_file = st.file_uploader(
                "Drop your image here",
                type=['png', 'jpg', 'jpeg'],
                help=current_text["image_helper"],
                key="image_upload"
            )
            
            if uploaded_file:
                st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
        
        # Text input
        st.text_input(
            current_text["input_label"],
            key="user_input",
            placeholder=current_text["input_placeholder"],
            on_change=handle_submit
        )
        
        # Process submitted question
        if st.session_state.submitted_question:
            image_bytes = None
            if uploaded_file is not None:
                image_bytes = uploaded_file.getvalue()
                # Process submitted question continued...
            if st.session_state.submitted_question:
                with st.spinner(current_text["image_processing"] if image_bytes else ""):
                    asyncio.run(process_question(
                        st.session_state.submitted_question,
                        image_bytes
                    ))
                
                st.session_state.submitted_question = None
                st.rerun()
        
        # Chat controls
        cols = st.columns([4, 1])
        
        # Clear chat button
        if cols[1].button(current_text["clear_chat"], use_container_width=True):
            st.session_state.messages = []
            st.session_state.chat_memory.clear_history()
            st.session_state.message_counter = 0
            st.rerun()

def handle_error(error: Exception):
    """Handle errors gracefully"""
    error_messages = {
        Language.ENGLISH: {
            "generic": "An error occurred. Please try again.",
            "image": "Error processing image. Please try a different image or ask without an image.",
            "network": "Network error. Please check your connection and try again.",
        },
        Language.HINDI: {
            "generic": "एक त्रुटि हुई। कृपया पुनः प्रयास करें।",
            "image": "छवि प्रोसेसिंग में त्रुटि। कृपया कोई अन्य छवि आज़माएं या बिना छवि के पूछें।",
            "network": "नेटवर्क त्रुटि। कृपया अपना कनेक्शन जांचें और पुनः प्रयास करें।",
        }
    }
    
    language = st.session_state.language
    
    if "image" in str(error).lower():
        error_message = error_messages[language]["image"]
    elif "network" in str(error).lower():
        error_message = error_messages[language]["network"]
    else:
        error_message = error_messages[language]["generic"]
    
    st.error(error_message)
    logging.error(f"Error in app: {str(error)}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        handle_error(e)
