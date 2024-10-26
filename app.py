import streamlit as st
from typing import List, Dict, Any
import asyncio
import uuid
from core import (
    ChatConfig, ChatLogger, ChatMemory, GeminiRAG, 
    ProductDatabase, UserDataManager, MetadataExtractor
)

# Enhanced session state initialization
def init_session_state():
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
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
            'crop_name': None
        }
    if 'message_counter' not in st.session_state:
        st.session_state.message_counter = 0

# Configure the page
st.set_page_config(
    page_title="GAPL Starter Assistant",
    page_icon="🌱",
    layout="wide"
)

# Custom CSS
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
    rag = GeminiRAG(config.gemini_api_key)
    db = ProductDatabase(config)
    user_data_manager = UserDataManager(config.user_data_file)
    return config, logger, rag, db, user_data_manager

config, logger, rag, db, user_data_manager = initialize_components()

@st.cache_resource
def load_database():
    with open("STARTER.md", "r", encoding="utf-8") as f:
        markdown_content = f.read()
    db.process_markdown(markdown_content)

try:
    load_database()
except Exception as e:
    st.error(f"Error loading database: {str(e)}")


async def process_question(question: str):
    """Process user question and generate response"""
    try:
        # Get relevant documents
        relevant_docs = db.search(question)
        context = rag.create_context(relevant_docs)
        
        # Generate answer
        answer = await rag.get_answer(
            question, 
            context, 
            st.session_state.metadata,
            st.session_state.language
        )
        
        # Extract any metadata from the user's response
        new_metadata = MetadataExtractor.extract_metadata(question)
        if new_metadata:
            st.session_state.metadata.update(new_metadata)
            user_data_manager.save_user_data(st.session_state.session_id, st.session_state.metadata)
        
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
            "message_id": st.session_state.message_counter
        })
        
    except Exception as e:
        st.error(f"Error: {str(e)}")

def main():
    # Initialize session state
    init_session_state()
    
    # Initialize components
    config, logger, rag, db, user_data_manager = initialize_components()
    
    # Language selector
    with st.sidebar:
        language = st.radio(
            "Select Language / भाषा चुनें",
            ["English", "Hindi"],
            index=0 if st.session_state.language == "english" else 1
        )
        st.session_state.language = language.lower()
    
    # Title
    title = "GAPL Starter Product Assistant" if st.session_state.language == "english" else "GAPL Starter उत्पाद सहायक"
    st.title("🌱 " + title)
    
    # Display metadata if available
    if any(st.session_state.metadata.values()):
        with st.sidebar:
            st.markdown("### " + ("User Information" if st.session_state.language == "english" else "उपयोगकर्ता की जानकारी"))
            for key, value in st.session_state.metadata.items():
                if value:
                    key_display = {
                        'mobile_number': 'Mobile' if st.session_state.language == "english" else 'मोबाइल',
                        'location': 'Location' if st.session_state.language == "english" else 'स्थान',
                        'crop_name': 'Crop' if st.session_state.language == "english" else 'फसल'
                    }.get(key, key)
                    st.write(f"**{key_display}:** {value}")
    
    # Welcome message for new chat
    if not st.session_state.messages:
        welcome_msg = {
            "english": """
            👋 Welcome! I'm your GAPL Starter product expert. I can help you with:
            - Product benefits and features
            - Application methods and timing
            - Dosage recommendations
            - Crop compatibility
            - Technical specifications
            
            Choose a question below or ask your own!
            """,
            "hindi": """
            👋 नमस्ते! मैं आपका GAPL Starter उत्पाद विशेषज्ञ हूं। मैं आपकी इन विषयों में मदद कर सकता हूं:
            - उत्पाद के लाभ और विशेषताएं
            - उपयोग की विधि और समय
            - मात्रा की सिफारिशें
            - फसल अनुकूलता
            - तकनीकी विवरण
            
            नीचे दिए गए प्रश्नों में से चुनें या अपना प्रश्न पूछें!
            """
        }
        st.markdown(welcome_msg[st.session_state.language])
        
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
    
    # Input area
    with st.container():
        placeholder = {
            "english": "Type your question here...",
            "hindi": "अपना प्रश्न यहाँ लिखें..."
        }
        
        user_input = st.text_input(
            "Ask me anything about GAPL Starter:" if st.session_state.language == "english" else "GAPL Starter के बारे में कुछ भी पूछें:",
            key="user_input",
            placeholder=placeholder[st.session_state.language]
        )
        
        if user_input:
            asyncio.run(process_question(user_input))
            st.rerun()
        
        # Clear chat button
        cols = st.columns([4, 1])
        clear_text = "Clear Chat" if st.session_state.language == "english" else "चैट साफ़ करें"
        if cols[1].button(clear_text):
            confirm_text = "Confirm Clear" if st.session_state.language == "english" else "पुष्टि करें"
            if st.sidebar.button("⚠️ " + confirm_text):
                st.session_state.messages = []
                st.session_state.chat_memory.clear_history()
                st.session_state.message_counter = 0
                st.rerun()

if __name__ == "__main__":
    main()
