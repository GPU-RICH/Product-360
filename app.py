import streamlit as st
from typing import List, Dict, Any
import asyncio
from core import ChatConfig, ChatLogger, ChatMemory, QuestionGenerator, GeminiRAG, ProductDatabase, UserInfoParser

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
# Add these to your existing session state initializations
if 'user_info_collected' not in st.session_state:
    st.session_state.user_info_collected = False
if 'user_info' not in st.session_state:
    st.session_state.user_info = None

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
        answer = await rag.get_answer(question, context)
        follow_up_questions = await question_gen.generate_questions(question, answer)
        
        st.session_state.chat_memory.add_interaction(question, answer)
        logger.log_interaction(question, answer)
        
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

def main():
    st.title("🌱 GAPL Starter Product Assistant")
    
    # Initialize UserInfoParser
    user_info_parser = UserInfoParser(config.gemini_api_key)
    
    # Handle user information collection
    if not st.session_state.user_info_collected:
        if not st.session_state.messages:
            welcome_message = """🌱 Hey Farmer!! Good to see you. I can help you with GAPL Starter, but before that could you please give me some info regarding yourself?

Please provide your:
- Name
- Phone number
- Have you purchased GAPL Starter?
- What crops do you grow?
- Location"""
            
            st.session_state.messages.append({
                "role": "assistant",
                "content": welcome_message,
                "message_id": st.session_state.message_counter
            })
        
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
        
        # Create a form for user information
        with st.form(key='user_info_form'):
            user_input = st.text_input(
                "Please provide your information:",
                key="user_info_input",
                placeholder="Example: My name is John and my phone number is 1234567890. I grow wheat and rice in Delhi"
            )
            submit_button = st.form_submit_button("Submit")
            
            if submit_button and user_input:
                # Parse user info
                user_info = asyncio.run(user_info_parser.parse_user_info(user_input))
                st.session_state.user_info = user_info
                st.session_state.user_info_collected = True
                
                # Add user message to chat history
                st.session_state.messages.append({
                    "role": "user",
                    "content": user_input,
                    "message_id": st.session_state.message_counter
                })
                
                # Add confirmation message
                confirmation_message = f"""Thank you for providing your information! I'll be happy to help you with any questions about GAPL Starter.

You can choose from the questions below or ask your own!"""
                
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": confirmation_message,
                    "message_id": st.session_state.message_counter + 1
                })
                
                st.session_state.message_counter += 2
                st.rerun()
    
    # Main chat interface after user info is collected
    else:
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
                
                # Display follow-up questions if available
                if message.get("questions"):
                    cols = st.columns(2)
                    for i, question in enumerate(message["questions"]):
                        if cols[i % 2].button(
                            question,
                            key=f"followup_{message['message_id']}_{i}",
                            use_container_width=True
                        ):
                            asyncio.run(process_question(question))
        
        # Display initial questions if this is the first interaction
        if len(st.session_state.messages) <= 2:  # Only user info and confirmation message
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
        
        # Chat input area
        with st.container():
            col1, col2 = st.columns([4, 1])
            
            with col1:
                st.text_input(
                    "Ask me anything about GAPL Starter:",
                    key="user_input",
                    placeholder="Type your question here...",
                    on_change=handle_submit
                )
            
            with col2:
                if st.button("Clear Chat", use_container_width=True):
                    st.session_state.messages = []
                    st.session_state.chat_memory.clear_history()
                    st.session_state.message_counter = 0
                    st.session_state.user_info_collected = False
                    st.rerun()
            
            # Process submitted question
            if st.session_state.submitted_question:
                asyncio.run(process_question(st.session_state.submitted_question))
                st.session_state.submitted_question = None
                st.rerun()

def handle_submit():
    if st.session_state.user_input:
        st.session_state.submitted_question = st.session_state.user_input
        st.session_state.user_input = ""

if __name__ == "__main__":
    main()
