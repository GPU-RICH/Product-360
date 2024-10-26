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
    UserDataManager,
    TRANSLATIONS
)
import uuid
import json
import time

# Enhanced session state initialization
if 'session_id' not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if 'language' not in st.session_state:
    st.session_state.language = 'en'
if 'chat_memory' not in st.session_state:
    st.session_state.chat_memory = ChatMemory()
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'user_data' not in st.session_state:
    st.session_state.user_data = {}
if 'show_data_collection' not in st.session_state:
    st.session_state.show_data_collection = False
if 'current_data_field' not in st.session_state:
    st.session_state.current_data_field = None
if 'submitted_question' not in st.session_state:
    st.session_state.submitted_question = None

# Initial questions in both languages
INITIAL_QUESTIONS = {
    'en': [
        "What are the main benefits of GAPL Starter?",
        "How do I apply GAPL Starter correctly?",
        "Which crops is GAPL Starter suitable for?",
        "What is the recommended dosage?"
    ],
    'hi': [
        "GAPL Starter के मुख्य लाभ क्या हैं?",
        "GAPL Starter को सही तरीके से कैसे लगाएं?",
        "GAPL Starter किन फसलों के लिए उपयुक्त है?",
        "अनुशंसित खुराक क्या है?"
    ]
}

# UI text translations
UI_TEXT = {
    'en': {
        'title': "🌱 GAPL Starter Product Assistant",
        'welcome': """
        👋 Welcome! I'm your GAPL Starter product expert. I can help you learn about:
        - Product benefits and features
        - Application methods and timing
        - Dosage recommendations
        - Crop compatibility
        - Technical specifications
        
        Choose a question below or ask your own!
        """,
        'input_placeholder': "Ask me anything about GAPL Starter...",
        'clear_chat': "Clear Chat",
        'session_info': "Session Information",
        'collected_info': "Collected Information:",
        'data_collection_title': "Please help us serve you better",
        'submit': "Submit",
        'language_selector': "Select Language / भाषा चुनें"
    },
    'hi': {
        'title': "🌱 GAPL Starter उत्पाद सहायक",
        'welcome': """
        👋 नमस्ते! मैं आपका GAPL Starter विशेषज्ञ हूं। मैं आपको इन विषयों में मदद कर सकता हूं:
        - उत्पाद के लाभ और विशेषताएं
        - उपयोग की विधियां और समय
        - खुराक की सिफारिशें
        - फसल अनुकूलता
        - तकनीकी विवरण
        
        नीचे दिए गए प्रश्न चुनें या अपना प्रश्न पूछें!
        """,
        'input_placeholder': "GAPL Starter के बारे में कुछ भी पूछें...",
        'clear_chat': "चैट साफ करें",
        'session_info': "सत्र जानकारी",
        'collected_info': "एकत्रित जानकारी:",
        'data_collection_title': "कृपया हमें आपकी बेहतर सेवा करने में मदद करें",
        'submit': "जमा करें",
        'language_selector': "Select Language / भाषा चुनें"
    }
}

# Configure the page
st.set_page_config(
    page_title="GAPL Starter Assistant",
    page_icon="🌱",
    layout="wide"
)

# Custom CSS with language-specific fonts
st.markdown("""
<style>
.user-message {
    background-color: #72BF6A;
    color: black;
    padding: 15px;
    border-radius: 15px;
    margin: 10px 0;
    font-family: 'Arial', 'Noto Sans Devanagari', sans-serif;
}
.assistant-message {
    background-color: #000000;
    color: #98FB98;
    padding: 15px;
    border-radius: 15px;
    margin: 10px 0;
    font-family: 'Arial', 'Noto Sans Devanagari', sans-serif;
}
.stButton > button {
    background-color: #212B2A;
    color: white;
    border: none;
    padding: 10px 15px;
    border-radius: 5px;
    transition: background-color 0.3s;
    font-family: 'Arial', 'Noto Sans Devanagari', sans-serif;
}
.stButton > button:hover {
    background-color: #45a049;
}
.data-collection-form {
    background-color: #f0f2f6;
    padding: 20px;
    border-radius: 10px;
    margin: 10px 0;
}
</style>
""", unsafe_allow_html=True)

# Load product database
@st.cache_resource
def initialize_database():
    try:
        with open("STARTER.md", "r", encoding="utf-8") as f:
            markdown_content = f.read()
        
        config = ChatConfig()
        db = ProductDatabase(config)
        db.process_markdown(markdown_content)
        return db
    except Exception as e:
        st.error(f"Error initializing database: {str(e)}")
        return None

# Modified initialization
@st.cache_resource
def initialize_components():
    config = ChatConfig()
    logger = ChatLogger(config.log_file)
    question_gen = QuestionGenerator(config.gemini_api_key)
    rag = GeminiRAG(config.gemini_api_key)
    user_manager = UserDataManager(config.user_data_file)
    db = initialize_database()  # Initialize database here
    return config, logger, question_gen, rag, db, user_manager

config, logger, question_gen, rag, db, user_manager = initialize_components()

# Add a check for database initialization
if db is None:
    st.error("Failed to initialize the product database. Please check if STARTER.md exists and is properly formatted.")
    st.stop()
    
def handle_data_collection():
    """Handles the data collection form"""
    if st.session_state.show_data_collection and st.session_state.current_data_field:
        with st.form(generate_unique_key("form", "data_collection")):
            st.subheader(UI_TEXT[st.session_state.language]['data_collection_title'])
            
            field = st.session_state.current_data_field
            value = None
            
            if field == 'mobile_number':
                value = st.text_input(
                    TRANSLATIONS[st.session_state.language]['mobile_prompt'],
                    key=generate_unique_key("input", "mobile"),
                    max_chars=10
                )
            elif field == 'location':
                value = st.text_input(
                    TRANSLATIONS[st.session_state.language]['location_prompt'],
                    key=generate_unique_key("input", "location")
                )
            elif field == 'crop':
                value = st.text_input(
                    TRANSLATIONS[st.session_state.language]['crop_prompt'],
                    key=generate_unique_key("input", "crop")
                )
            elif field == 'purchase_status':
                value = st.radio(
                    TRANSLATIONS[st.session_state.language]['purchase_prompt'],
                    ['Yes', 'No', 'Planning to purchase'] if st.session_state.language == 'en' else 
                    ['हां', 'नहीं', 'खरीदने की योजना है'],
                    key=generate_unique_key("radio", "purchase")
                )
            
            submit_key = generate_unique_key("submit", field)
            if st.form_submit_button(UI_TEXT[st.session_state.language]['submit'], key=submit_key):
                if value:
                    st.session_state.user_data[field] = value
                    user_manager.save_user_data(st.session_state.session_id, st.session_state.user_data)
                    st.session_state.show_data_collection = False
                    st.session_state.current_data_field = None
                    st.rerun()

async def process_question(question: str):
    """Process user questions and generate responses"""
    try:
        relevant_docs = db.search(question)
        context = rag.create_context(relevant_docs)
        
        # Get response with potential data collection prompt
        response_data = await rag.get_answer(
            question, 
            context, 
            st.session_state.user_data,
            st.session_state.language
        )
        
        answer = response_data["answer"]
        if response_data["collect_data"]:
            st.session_state.show_data_collection = True
            st.session_state.current_data_field = response_data["collect_data"]
        
        follow_up_questions = await question_gen.generate_questions(
            question, 
            answer,
            st.session_state.language
        )
        
        st.session_state.chat_memory.add_interaction(
            question, 
            answer, 
            st.session_state.user_data
        )
        
        logger.log_interaction(
            question, 
            answer, 
            st.session_state.session_id,
            st.session_state.language
        )
        
        st.session_state.messages.append({
            "role": "user",
            "content": question
        })
        
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "questions": follow_up_questions
        })
        
    except Exception as e:
        st.error(f"Error processing question: {str(e)}")

def handle_language_change():
    """Handle language selection change"""
    st.session_state.messages = []
    st.session_state.chat_memory.clear_history()
    st.rerun()

def generate_unique_key(prefix: str, identifier: Any) -> str:
    """Generate a unique key for Streamlit elements"""
    timestamp = int(time.time() * 1000)  # millisecond timestamp
    return f"{prefix}_{identifier}_{timestamp}"

def main():
    # Language selector in sidebar
    with st.sidebar:
        language = st.selectbox(
            "Select Language / भाषा चुनें",
            options=['English', 'हिंदी'],
            key='language_selector',
            on_change=handle_language_change
        )
        st.session_state.language = 'en' if language == 'English' else 'hi'
        
        # Display user information
        st.subheader(UI_TEXT[st.session_state.language]['session_info'])
        if st.session_state.user_data:
            st.write(UI_TEXT[st.session_state.language]['collected_info'])
            for key, value in st.session_state.user_data.items():
                st.write(f"- {key}: {value}")

    # Main chat interface
    st.title(UI_TEXT[st.session_state.language]['title'])
    
    # Data collection form
    handle_data_collection()
    
    # Welcome message
    if not st.session_state.messages:
        st.markdown(UI_TEXT[st.session_state.language]['welcome'])
        
        # Display initial questions as buttons with unique keys
        cols = st.columns(2)
        for i, question in enumerate(INITIAL_QUESTIONS[st.session_state.language]):
            unique_key = generate_unique_key("initial", i)
            if cols[i % 2].button(
                question, 
                key=unique_key, 
                use_container_width=True
            ):
                asyncio.run(process_question(question))
    
    # Display chat history
    for msg_idx, message in enumerate(st.session_state.messages):
        if message["role"] == "user":
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
                st.markdown(f"_{TRANSLATIONS[st.session_state.language]['follow_up_prefix']}_")
                cols = st.columns(2)
                for q_idx, question in enumerate(message["questions"]):
                    unique_key = generate_unique_key("followup", f"{msg_idx}_{q_idx}")
                    if cols[q_idx % 2].button(
                        question,
                        key=unique_key,
                        use_container_width=True
                    ):
                        asyncio.run(process_question(question))
    
   # Chat input with unique key
    input_key = generate_unique_key("input", len(st.session_state.messages))
    st.text_input(
        "",
        key=input_key,
        placeholder=UI_TEXT[st.session_state.language]['input_placeholder'],
        on_change=lambda: handle_submit() if st.session_state.user_input else None
    )
    
    # Clear chat button with unique key
    cols = st.columns([4, 1])
    clear_key = generate_unique_key("clear", "chat")
    if cols[1].button(
        UI_TEXT[st.session_state.language]['clear_chat'], 
        key=clear_key,
        use_container_width=True
    ):
        st.session_state.messages = []
        st.session_state.chat_memory.clear_history()
        st.rerun()

def handle_submit():
    """Handle user input submission"""
    if st.session_state.user_input:
        st.session_state.submitted_question = st.session_state.user_input
        st.session_state.user_input = ""
        asyncio.run(process_question(st.session_state.submitted_question))
        st.session_state.submitted_question = None
        st.rerun()

if __name__ == "__main__":
    main()
