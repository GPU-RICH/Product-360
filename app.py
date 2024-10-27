import streamlit as st
from typing import List, Dict, Any
import asyncio
from core import (
    ChatConfig, 
    ChatLogger, 
    ChatMemory, 
    QuestionGenerator, 
    GeminiRAG, 
    ProductDatabase, 
    Language,
    UserManager,
    UserInfo,
    ImageProcessor,
    init_image_database_from_csv
)
import os
from PIL import Image

# UI Text translations with added image-related content
UI_TEXT = {
    Language.ENGLISH: {
        "title": "🌱 Product Assistant",
        "welcome_message": """
        👋 Welcome! I'm your product expert. I can help you with:
        - Analyzing crop problems through images
        - Product benefits and features
        - Application methods and timing
        - Dosage recommendations
        - Crop compatibility
        - Technical specifications
        
        Upload an image of your crop or choose a question below!
        """,
        "input_placeholder": "Type your question here...",
        "input_label": "Ask me anything:",
        "clear_chat": "Clear Chat",
        "language_selector": "Select Language",
        "sidebar_title": "User Information",
        "form_name": "Your Name",
        "form_mobile": "Mobile Number",
        "form_location": "Location",
        "form_purchase": "Have you purchased this product?",
        "form_crop": "What crop are you growing?",
        "form_submit": "Save Information",
        "form_success": "✅ Information saved successfully!",
        "form_error": "❌ Error saving information. Please try again.",
        "form_required": "Please fill in all required fields.",
        "image_upload": "📸 Upload a photo of your crop (optional)",
        "processing_image": "Analyzing your image...",
        "similar_cases": "Similar Cases Found:",
        "case_confidence": "Match Confidence:",
        "initial_questions": [
            "What are the main benefits?",
            "How do I apply this correctly?",
            "Which crops is this suitable for?",
            "What is the recommended dosage?"
        ]
    },
    Language.HINDI: {
        "title": "🌱 उत्पाद सहायक",
        "welcome_message": """
        👋 नमस्ते! मैं आपका उत्पाद विशेषज्ञ हूं। मैं आपकी इन चीज़ों में मदद कर सकता हूं:
        - छवियों के माध्यम से फसल की समस्याओं का विश्लेषण
        - उत्पाद के लाभ और विशेषताएं
        - प्रयोग विधि और समय
        - खुराक की सिफारिशें
        - फसल अनुकूलता
        - तकनीकी विवरण
        
        अपनी फसल की तस्वीर अपलोड करें या नीचे से कोई प्रश्न चुनें!
        """,
        "input_placeholder": "अपना प्रश्न यहां टाइप करें...",
        "input_label": "कुछ भी पूछें:",
        "clear_chat": "चैट साफ़ करें",
        "language_selector": "भाषा चुनें",
        "sidebar_title": "उपयोगकर्ता जानकारी",
        "form_name": "आपका नाम",
        "form_mobile": "मोबाइल नंबर",
        "form_location": "स्थान",
        "form_purchase": "क्या आपने खरीदा है?",
        "form_crop": "आप कौन सी फसल उगा रहे हैं?",
        "form_submit": "जानकारी सहेजें",
        "form_success": "✅ जानकारी सफलतापूर्वक सहेजी गई!",
        "form_error": "❌ जानकारी सहेजने में त्रुटि। कृपया पुनः प्रयास करें।",
        "form_required": "कृपया सभी आवश्यक फ़ील्ड भरें।",
        "image_upload": "📸 अपनी फसल की तस्वीर अपलोड करें (वैकल्पिक)",
        "processing_image": "आपकी छवि का विश्लेषण किया जा रहा है...",
        "similar_cases": "समान मामले मिले:",
        "case_confidence": "मेल विश्वास:",
        "initial_questions": [
            "मुख्य लाभ क्या हैं?",
            "प्रयोग कैसे करें?",
            "किन फसलों के लिए उपयुक्त है?",
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
if 'current_image' not in st.session_state:
    st.session_state.current_image = None

# Configure the page
st.set_page_config(
    page_title="Product 360 Assistant",
    page_icon="🌱",
    layout="wide"
)

# Custom CSS with added image styling
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
.image-container {
    border: 2px solid #72BF6A;
    border-radius: 10px;
    padding: 10px;
    margin: 10px 0;
}
.similar-case {
    background-color: #1E1E1E;
    border: 1px solid #72BF6A;
    padding: 10px;
    border-radius: 10px;
    margin: 5px 0;
}
.confidence-meter {
    height: 10px;
    background-color: #72BF6A;
    border-radius: 5px;
    margin: 5px 0;
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
</style>
""", unsafe_allow_html=True)

# Initialize components
@st.cache_resource
def initialize_components():
    config = ChatConfig()
    logger = ChatLogger(config.log_file)
    question_gen = QuestionGenerator(config.gemini_api_key)
    image_processor = ImageProcessor(config)
    
    # Initialize image database from CSV
    if os.path.exists("crop_problems.csv"):
        init_image_database_from_csv("crop_problems.csv", image_processor)
    
    rag = GeminiRAG(config.gemini_api_key, image_processor)
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

async def process_question(question: str, image: Image.Image = None):
    """Process user question with optional image"""
    try:
        # Get text-based relevant docs
        relevant_docs = db.search(question)
        context = rag.create_context(relevant_docs)
        
        # Get answer with image analysis if image is provided
        answer, similar_images = await rag.get_answer(
            question=question,
            context=context,
            language=st.session_state.language,
            user_info=st.session_state.user_info,
            query_image=image
        )
        
        # Generate follow-up questions
        follow_up_questions = await question_gen.generate_questions(
            question=question,
            answer=answer,
            language=st.session_state.language,
            user_info=st.session_state.user_info,
            has_image=image is not None
        )
        
        # Log interaction
        st.session_state.chat_memory.add_interaction(
            question=question,
            answer=answer,
            has_image=image is not None
        )
        logger.log_interaction(
            question=question,
            answer=answer,
            language=st.session_state.language,
            user_info=st.session_state.user_info,
            has_image=image is not None
        )
        
        # Update message counter
        st.session_state.message_counter += 1
        
        # Add messages to chat history
        st.session_state.messages.append({
            "role": "user",
            "content": question,
            "image": image,
            "message_id": st.session_state.message_counter
        })
        
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "similar_images": similar_images,
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

def main():
    current_text = UI_TEXT[st.session_state.language]
    
    # Language selector
    with st.container():
        cols = st.columns([3, 1])
        with cols[1]:
            selected_language = st.selectbox(
                current_text["language_selector"],
                options=[Language.ENGLISH, Language.HINDI],
                format_func=lambda x: "English" if x == Language.ENGLISH else "हिंदी",
                key="language_selector",
                index=0 if st.session_state.language == Language.ENGLISH else 1,
                on_change=handle_language_change
            )
            st.session_state.language = selected_language
    
    # Render user form in sidebar
    render_user_form()
    
    # Main title
    st.title(current_text["title"])
    
    # Welcome message for new chat
    if not st.session_state.messages:
        st.markdown(current_text["welcome_message"])
        
        # Display initial questions as buttons
        cols = st.columns(2)
        for i, question in enumerate(current_text["initial_questions"]):
            if cols[i % 2].button(question, key=f"initial_{i}", use_container_width=True):
                asyncio.run(process_question(question))
    
    # Display chat history with images
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(
                f'<div class="user-message">👤 {message["content"]}</div>',
                unsafe_allow_html=True
            )
            if message.get("image"):
                with st.container():
                    st.image(message["image"], caption="Uploaded Image", use_column_width=True)
        else:
            st.markdown(
                f'<div class="assistant-message">🌱 {message["content"]}</div>',
                unsafe_allow_html=True
            )
            
            # Display similar images if any
            if message.get("similar_images"):
                st.subheader(current_text["similar_cases"])
                cols = st.columns(len(message["similar_images"]))
                for i, img_data in enumerate(message["similar_images"]):
                    with cols[i]:
                        with st.container():
                            st.image(img_data["path"], use_column_width=True)
                            st.markdown(f'<div class="similar-case">'
                                      f'<p><strong>Problem:</strong> {img_data["problem_type"]}</p>'
                                      f'<p>{img_data["description"]}</p>'
                                      f'<p><strong>{current_text["case_confidence"]}</strong></p>'
                                      f'</div>', unsafe_allow_html=True)
            
            # Display follow-up questions
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
        # Image upload
        uploaded_file = st.file_uploader(
            current_text["image_upload"],
            type=["jpg", "jpeg", "png"],
            key="image_upload"
        )
        
        if uploaded_file:
            try:
                image = Image.open(uploaded_file)
                st.session_state.current_image = image
                st.image(image, caption="Uploaded Image", use_column_width=True)
            except Exception as e:
                st.error(f"Error loading image: {str(e)}")
                st.session_state.current_image = None
        else:
            st.session_state.current_image = None
        
        # Text input
        st.text_input(
            current_text["input_label"],
            key="user_input",
            placeholder=current_text["input_placeholder"],
            on_change=handle_submit
        )
        
        # Process submitted question with image
        if st.session_state.submitted_question:
            with st.spinner(current_text["processing_image"] if st.session_state.current_image else ""):
                asyncio.run(process_question(
                    st.session_state.submitted_question,
                    st.session_state.current_image
                ))
            st.session_state.submitted_question = None
            st.session_state.current_image = None
            st.rerun()
        
        # Clear chat button
        cols = st.columns([4, 1])
        if cols[1].button(current_text["clear_chat"], use_container_width=True):
            st.session_state.messages = []
            st.session_state.chat_memory.clear_history()
            st.session_state.message_counter = 0
            st.rerun()

if __name__ == "__main__":
    main()
