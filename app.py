import streamlit as st
from typing import List, Dict, Any
import asyncio
from core import ChatConfig, ChatLogger, ChatMemory, QuestionGenerator, GeminiRAG, ProductDatabase, Language

# UI Text translations
UI_TEXT = {
    Language.ENGLISH: {
        "title": "🌱 GAPL Starter Product Assistant",
        "welcome_message": """
        👋 Welcome! I'm your GAPL Starter product expert. I can help you learn about:
        - Product benefits and features
        - Application methods and timing
        - Dosage recommendations
        - Crop compatibility
        - Technical specifications
        
        Choose a question below or ask your own!
        """,
        "input_placeholder": "Type your question here...",
        "input_label": "Ask me anything about GAPL Starter:",
        "clear_chat": "Clear Chat",
        "language_selector": "Select Language",
        "initial_questions": [
            "What are the main benefits of GAPL Starter?",
            "How do I apply GAPL Starter correctly?",
            "Which crops is GAPL Starter suitable for?",
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
        
        नीचे दिए गए प्रश्नों में से चुनें या अपना प्रश्न पूछें!
        """,
        "input_placeholder": "अपना प्रश्न यहां टाइप करें...",
        "input_label": "GAPL स्टार्टर के बारे में कुछ भी पूछें:",
        "clear_chat": "चैट साफ़ करें",
        "language_selector": "भाषा चुनें",
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

# Configure the page
st.set_page_config(
    page_title="GAPL Starter Assistant",
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
    return config, logger, question_gen, rag, db

config, logger, question_gen, rag, db = initialize_components()

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

async def process_question(question: str):
    try:
        relevant_docs = db.search(question)
        context = rag.create_context(relevant_docs)
        answer = await rag.get_answer(question, context, st.session_state.language)
        follow_up_questions = await question_gen.generate_questions(
            question, 
            answer, 
            st.session_state.language
        )
        
        st.session_state.chat_memory.add_interaction(question, answer)
        logger.log_interaction(question, answer, st.session_state.language)
        
        st.session_state.message_counter += 1
        
        st.session_state.messages.append({
            "role": "user",
            "content": question,
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
    
    # Input area
    with st.container():
        st.text_input(
            current_text["input_label"],
            key="user_input",
            placeholder=current_text["input_placeholder"],
            on_change=handle_submit
        )
        
        # Process submitted question
        if st.session_state.submitted_question:
            asyncio.run(process_question(st.session_state.submitted_question))
            st.session_state.submitted_question = None
            st.rerun()
        
        cols = st.columns([4, 1])
        # Clear chat button
        if cols[1].button(current_text["clear_chat"], use_container_width=True):
            st.session_state.messages = []
            st.session_state.chat_memory.clear_history()
            st.session_state.message_counter = 0
            st.rerun()

if __name__ == "__main__":
    main()
