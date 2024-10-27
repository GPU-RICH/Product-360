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
        Welcome to the Product 360 Experience! 🌾
        Hi there! It’s great to connect with you. You’re now part of a growing family of over 10,000 farmers who trust and use this product. We’re proud to support you on this journey.
        You scanned the QR code for Entokill 250ml, you’ll find everything you need to know, along with stories from farmers just like you who have seen results with this product.
        Take a moment to explore some testimonials from your fellow farmers https://www.youtube.com/watch?v=EY489XtDYEo and get ready to make the most out of Entokill 250ml!        
        
        Choose amongst the questions below or ask your own!
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
        "title": "🌱 Entokill 250ml उत्पाद सहायक",
        "welcome_message": """
        Product 360 अनुभव में आपका स्वागत है! 🌾
        नमस्ते! अब आप उन 10,000+ किसानों के परिवार का हिस्सा बन गए हैं जो इस उत्पाद का उपयोग करते हैं।
        आपने Entokill 250ml का QR कोड स्कैन किया है—यहां आपको इस उत्पाद से जुड़ी सभी ज़रूरी जानकारी मिलेगी, साथ ही उन किसानों की कहानियाँ भी, जिन्होंने इस उत्पाद से बेहतरीन परिणाम हासिल किए हैं।
        थोड़ा समय निकालकर अपने साथी किसानों के (product name) से जुड़े अनुभव देखें -> https://www.youtube.com/watch?v=EY489XtDYEo और Entokill 250ml का अधिकतम लाभ उठाने के लिए तैयार हो जाएं!
        
        नीचे दिए गए सवालों में से कोई चुनें या अपना खुद का सवाल पूछें
        """,
        "input_placeholder": "अपना प्रश्न यहां टाइप करें...",
        "input_label": "कुछ भी पूछें:",
        "clear_chat": "चैट साफ़ करें",
        "language_selector": "भाषा चुनें",
        "sidebar_title": "उपयोगकर्ता जानकारी",
        "form_name": "आपका नाम",
        "form_mobile": "मोबाइल नंबर",
        "form_location": "स्थान",
        "form_purchase": "क्या आपने उत्पाद खरीदा है?",
        "form_crop": "आप कौन सी फसल उगा रहे हैं?",
        "form_submit": "जानकारी सहेजें",
        "form_success": "✅ जानकारी सफलतापूर्वक सहेजी गई!",
        "form_error": "❌ जानकारी सहेजने में त्रुटि। कृपया पुनः प्रयास करें।",
        "form_required": "कृपया सभी आवश्यक फ़ील्ड भरें।",
        "image_upload": "छवि अपलोड करें (वैकल्पिक)",
        "image_helper": "यदि आपके पास अपनी फसल के बारे में कोई चिंता या प्रश्न है, तो उसकी छवि अपलोड करें।",
        "image_processing": "आपकी छवि प्र्रोसेस की जा रही है...",
        "initial_questions": [
            "उत्पाद के मुख्य लाभ क्या हैं?",
            "उत्पाद का प्रयोग कैसे करें?",
            "उत्पाद किन फसलों के लिए उपयुक्त है?",
            "अनुशंसित मात्रा क्या है?"
        ]
    }
}

# Add to your UI_TEXT dictionary
UI_TEXT[Language.ENGLISH].update({
    "show_suggestions": "Show Related Products",
    "suggestions_title": "Related Products:",
})

UI_TEXT[Language.HINDI].update({
    "show_suggestions": "संबंधित उत्पाद दिखाएं",
    "suggestions_title": "संबंधित उत्पाद:",
})


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
    st.session_state.language = Language.HINDI
if 'user_info' not in st.session_state:
    st.session_state.user_info = None
if 'show_suggestions' not in st.session_state:
    st.session_state.show_suggestions = False
    
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
.product-card {
    border: 1px solid #ddd;
    border-radius: 10px;
    padding: 15px;
    margin: 10px;
    background-color: white;
    text-align: center;
}

.related-products {
    padding: 20px;
    margin: 20px 0;
    background-color: #f5f5f5;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# Add this function to display the fixed product suggestions
def display_product_suggestions():
    if st.session_state.show_suggestions:
        current_text = UI_TEXT[st.session_state.language]
        
        # Add a separator in sidebar
        st.sidebar.markdown("---")
        
        # Display products title
        st.sidebar.markdown(f"### {current_text['suggestions_title']}")
        
        # Product 1
        with st.sidebar.container():
            st.image("paras-npk.webp", caption="PARAS NPK 12:32:16 50 Kg", use_column_width=True)
            st.markdown("**PARAS NPK 12:32:16 50 Kg**")
            st.markdown("---")  # Separator between products
        
        # Product 2
        with st.sidebar.container():
            st.image("mosaic.webp", caption="MOSAIC MOP 50 Kg", use_column_width=True)
            st.markdown("**MOSAIC MOP 50 Kg**")
            st.markdown("---")  # Separator between products
        
        # Product 3
        with st.sidebar.container():
            st.image("paras_dap.webp", caption="PARAS DAP 50 Kg", use_column_width=True)
            st.markdown("**PARAS DAP 50 Kg**")
                    
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
    with open("ENTOKILL.md", "r", encoding="utf-8") as f:
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
                index=1
            )
            st.session_state.language = selected_language
    # Update session state language immediately when changed
    if st.session_state.language != selected_language:
        st.session_state.language = selected_language
        handle_language_change()
        
    # Add suggestions toggle at the top of sidebar
    current_text = UI_TEXT[st.session_state.language]
    st.sidebar.checkbox(
        current_text["show_suggestions"],
        key="show_suggestions",
        value=st.session_state.show_suggestions
    )

    # Render user form in sidebar
    render_user_form()
    
    # Display product suggestions in sidebar if enabled
    if st.session_state.show_suggestions:
        display_product_suggestions()

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
