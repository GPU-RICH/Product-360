# app.py

import streamlit as st
from typing import List, Dict, Any
import asyncio
from core import (
    ChatConfig, ChatLogger, ChatMemory, QuestionGenerator, 
    GeminiRAG, ProductDatabase, LanguageTranslator
)

# Enhanced session state initialization
def init_session_state():
    if 'chat_memory' not in st.session_state:
        st.session_state.chat_memory = ChatMemory()
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'language' not in st.session_state:
        st.session_state.language = "english"
    if 'initial_questions' not in st.session_state:
        st.session_state.initial_questions = {
            "english": [
                "What are the main benefits of GAPL Starter?",
                "How do I apply GAPL Starter correctly?",
                "Which crops is GAPL Starter suitable for?",
                "What is the recommended dosage?"
            ],
            "hindi": [
                "GAPL Starter के मुख्य फायदे क्या हैं?",
                "GAPL Starter को सही तरीके से कैसे लगाएं?",
                "GAPL Starter किन फसलों के लिए उपयुक्त है?",
                "अनुशंसित मात्रा क्या है?"
            ]
        }
    if 'metadata' not in st.session_state:
        st.session_state.metadata = {
            'mobile_number': None,
            'location': None,
            'crop_name': None,
            'purchase_status': None,
            'name': None
        }
    if 'metadata_collected' not in st.session_state:
        st.session_state.metadata_collected = False
    if 'message_counter' not in st.session_state:
        st.session_state.message_counter = 0

# Configure the page
st.set_page_config(
    page_title="GAPL Starter Assistant",
    page_icon="🌱",
    layout="wide"
)

# Custom CSS with Hindi font support
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans:wght@400;700&display=swap');

* {
    font-family: 'Noto Sans', sans-serif;
}

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

.metadata-form {
    background-color: #f0f2f6;
    padding: 20px;
    border-radius: 10px;
    margin: 15px 0;
}

.language-selector {
    position: fixed;
    top: 10px;
    right: 10px;
    z-index: 1000;
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
    rag = GeminiRAG(config.gemini_api_key)
    db = ProductDatabase(config)
    translator = LanguageTranslator()
    return config, logger, question_gen, rag, db, translator

def collect_metadata():
    """Displays metadata collection form"""
    with st.form(key="metadata_form", clear_on_submit=True):
        st.markdown("### " + get_translated_text("Please provide some information:", st.session_state.language))
        
        cols = st.columns(2)
        with cols[0]:
            mobile = st.text_input(
                get_translated_text("Mobile Number (required):", st.session_state.language),
                max_chars=10,
                help="10-digit mobile number"
            )
            
            location = st.text_input(
                get_translated_text("Location/District (required):", st.session_state.language)
            )
            
        with cols[1]:
            crop = st.text_input(
                get_translated_text("Main Crop (required):", st.session_state.language)
            )
            
            name = st.text_input(
                get_translated_text("Name (optional):", st.session_state.language)
            )
            
        purchase_status = st.radio(
            get_translated_text("Have you used GAPL Starter before?", st.session_state.language),
            options=get_translated_options(["Yes", "No", "Planning to purchase"], st.session_state.language)
        )
        
        submit_button = st.form_submit_button(
            get_translated_text("Submit", st.session_state.language)
        )
        
        if submit_button:
            if not mobile or not location or not crop:
                st.error(get_translated_text("Please fill in all required fields.", st.session_state.language))
                return False
                
            if not mobile.isdigit() or len(mobile) != 10:
                st.error(get_translated_text("Please enter a valid 10-digit mobile number.", st.session_state.language))
                return False
                
            st.session_state.metadata.update({
                'mobile_number': mobile,
                'location': location,
                'crop_name': crop,
                'name': name if name else None,
                'purchase_status': purchase_status
            })
            
            st.session_state.metadata_collected = True
            return True
            
    return False

def get_translated_text(text: str, language: str) -> str:
    """Helper function to translate text based on selected language"""
    if language.lower() == "hindi":
        return translator.to_hindi(text)
    return text

def get_translated_options(options: List[str], language: str) -> List[str]:
    """Helper function to translate a list of options"""
    if language.lower() == "hindi":
        return [translator.to_hindi(opt) for opt in options]
    return options

async def process_question(question: str):
    """Process user question and generate response"""
    try:
        # Translate to English if needed
        english_question = translator.to_english(question) if st.session_state.language == "hindi" else question
        
        # Get relevant documents
        relevant_docs = db.search(english_question)
        context = rag.create_context(relevant_docs)
        
        # Generate answer
        answer = await rag.get_answer(
            english_question, 
            context, 
            st.session_state.metadata,
            st.session_state.language
        )
        
        # Generate follow-up questions
        follow_up = await question_gen.generate_questions(
            english_question,
            answer,
            st.session_state.metadata,
            st.session_state.language
        )
        
        # Log interaction
        logger.log_interaction(question, answer, st.session_state.metadata)
        
        # Update chat memory
        st.session_state.chat_memory.add_interaction(question, answer)
        st.session_state.message_counter += 1
        
        # Add to messages
        st.session_state.messages.append({
            "role": "user",
            "content": question,
            "message_id": st.session_state.message_counter
        })
        
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "questions": follow_up,
            "message_id": st.session_state.message_counter
        })
        
    except Exception as e:
        st.error(f"Error: {str(e)}")

def main():
    # Initialize session state
    init_session_state()
    
    # Initialize components
    config, logger, question_gen, rag, db, translator = initialize_components()
    
    # Language selector
    with st.sidebar:
        language = st.radio(
            "Select Language / भाषा चुनें",
            ["English", "Hindi"],
            index=0 if st.session_state.language == "english" else 1
        )
        st.session_state.language = language.lower()
    
    # Title
    st.title("🌱 " + get_translated_text("GAPL Starter Product Assistant", st.session_state.language))
    
    # Collect metadata if not already done
    if not st.session_state.metadata_collected:
        if collect_metadata():
            st.rerun()
    
    # Display metadata summary
    if st.session_state.metadata_collected:
        with st.sidebar:
            st.markdown("### " + get_translated_text("User Information", st.session_state.language))
            for key, value in st.session_state.metadata.items():
                if value:
                    st.write(f"**{get_translated_text(key.replace('_', ' ').title(), st.session_state.language)}:** {value}")
    
    # Welcome message for new chat
    if not st.session_state.messages:
        welcome_msg = get_translated_text(
            """
            👋 Welcome! I'm your GAPL Starter product expert. I can help you with:
            - Product benefits and features
            - Application methods and timing
            - Dosage recommendations
            - Crop compatibility
            - Technical specifications
            
            Choose a question below or ask your own!
            """,
            st.session_state.language
        )
        st.markdown(welcome_msg)
        
        # Display initial questions as buttons
        cols = st.columns(2)
        questions = st.session_state.initial_questions[st.session_state.language]
        for i, question in enumerate(questions):
            if cols[i % 2].button(question, key=f"initial_{i}", use_container_width=True):
                asyncio.run(process_question(question))
    
    # Display chat history
    for message in st.session_state.messages:
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
            
            # Display follow-up questions
            if message.get("questions"):
                cols = st.columns(len(message["questions"]))
                for i, question in enumerate(message["questions"]):
                    if cols[i].button(
                        question,
                        key=f"followup_{message['message_id']}_{i}",
                        use_container_width=True
                    ):
                        asyncio.run(process_question(question))
    
    # Input area
    with st.container():
        user_input = st.text_input(
            get_translated_text("Ask me anything about GAPL Starter:", st.session_state.language),
            key="user_input",
            placeholder=get_translated_text("Type your question here...", st.session_state.language)
        )
        
        if user_input:
            asyncio.run(process_question(user_input))
            st.rerun()
        
        # Clear chat button
        cols = st.columns([4, 1])
        if cols[1].button(get_translated_text("Clear Chat", st.session_state.language)):
            if st.sidebar.button("⚠️ " + get_translated_text("Confirm Clear", st.session_state.language)):
                st.session_state.messages = []
                st.session_state.chat_memory.clear_history()
                st.session_state.message_counter = 0
                st.rerun()

if __name__ == "__main__":
    main()
