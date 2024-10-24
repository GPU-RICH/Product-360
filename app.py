import streamlit as st
from typing import List, Dict, Any
import asyncio
from core import ChatConfig, ChatLogger, ChatMemory, QuestionGenerator, GeminiRAG, ProductDatabase

# Initialize session state
if 'chat_memory' not in st.session_state:
    st.session_state.chat_memory = ChatMemory()
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'initial_questions' not in st.session_state:
    st.session_state.initial_questions = [
        "What are the main benefits of GAPL Starter?",
        "How do I apply GAPL Starter correctly?",
        "Which crops is GAPL Starter suitable for?",
        "What is the recommended dosage?"
    ]
if 'message_counter' not in st.session_state:
    st.session_state.message_counter = 0
if 'submitted_question' not in st.session_state:
    st.session_state.submitted_question = None
if 'user_metadata' not in st.session_state:
    st.session_state.user_metadata = {
        'product_name': 'GAPL Starter',
        'mobile_number': None,
        'crop_name': None,
        'location': None
    }

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
.metadata-section {
    background-color: #f0f2f6;
    padding: 10px;
    border-radius: 5px;
    margin-top: 10px;
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

def handle_submit():
    if st.session_state.user_input:
        st.session_state.submitted_question = st.session_state.user_input
        st.session_state.user_input = ""

async def process_question(question: str):
    try:
        relevant_docs = db.search(question)
        context = rag.create_context(relevant_docs)
        answer = await rag.get_answer(question, context)
        follow_up_questions = await question_gen.generate_questions(question, answer)
        
        st.session_state.chat_memory.add_interaction(question, answer)
        logger.log_interaction(question, answer)
        
        st.session_state.message_counter += 1
        
        # Log metadata updates
        if any(new_value != st.session_state.user_metadata[key] 
               for key, new_value in st.session_state.user_metadata.items()):
            logger.log_interaction(
                "Metadata Update",
                f"Updated user metadata: {str(st.session_state.user_metadata)}"
            )
        
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

def display_metadata_summary():
    """Display collected metadata in a clean format"""
    metadata = st.session_state.user_metadata
    collected = {k: v for k, v in metadata.items() if v and k != 'product_name'}
    
    if collected:
        with st.sidebar:
            st.markdown("### 📊 Session Information")
            for key, value in collected.items():
                st.markdown(f"**{key.replace('_', ' ').title()}**: {value}")

def main():
    st.title("🌱 GAPL Starter Product Assistant")
    
    # Show metadata summary
    display_metadata_summary()
    
    # Welcome message
    if not st.session_state.messages:
        st.markdown("""
        👋 Welcome! I'm your GAPL Starter product expert. I can help you learn about:
        - Product benefits and features
        - Application methods and timing
        - Dosage recommendations
        - Crop compatibility
        - Technical specifications
        
        Choose a question below or ask your own!
        """)
        
        # Display initial questions as buttons
        cols = st.columns(2)
        for i, question in enumerate(st.session_state.initial_questions):
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
        st.text_input(
            "Ask me anything about GAPL Starter:",
            key="user_input",
            placeholder="Type your question here...",
            on_change=handle_submit
        )
        
        if st.session_state.submitted_question:
            asyncio.run(process_question(st.session_state.submitted_question))
            st.session_state.submitted_question = None
            st.rerun()
        
        # Clear chat button with confirmation
        cols = st.columns([4, 1])
        if cols[1].button("Clear Chat", use_container_width=True):
            if st.sidebar.button("⚠️ Confirm Clear"):
                st.session_state.messages = []
                st.session_state.chat_memory.clear_history()
                st.session_state.message_counter = 0
                # Keep metadata intact
                st.rerun()

if __name__ == "__main__":
    main()
