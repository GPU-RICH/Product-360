#APP.PY
import streamlit as st
from typing import List, Dict, Any
import asyncio
import json
from core import ChatConfig, ChatLogger, ChatMemory, QuestionGenerator, GeminiRAG, ProductDatabase

# Enhanced session state initialization
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
if 'user_data' not in st.session_state:
    st.session_state.user_data = {}


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
    """Enhanced question processing with metadata collection triggers"""
    try:
        relevant_docs = db.search(question)
        context = rag.create_context(relevant_docs)
        answer = await rag.get_answer(question, context)
        follow_up_questions = await question_gen.generate_questions(question, answer)

        st.session_state.chat_memory.add_interaction(question, answer)
        logger.log_interaction(question, answer)

        # Collect user data after a few interactions
        message_count = len(st.session_state.messages)
        if message_count in [1,3,5]: #Collect data after 1st, 3rd and 5th message
            user_data = collect_user_data()
            st.session_state.user_data.update(user_data)
            if all(st.session_state.user_data.values()):
                save_user_data(st.session_state.user_data)

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

def collect_user_data():
    """Collects user data using a form"""
    user_info = {}
    if 'mobile' not in st.session_state.user_data:
        mobile = st.text_input("Your mobile number (optional):", key="mobile")
        if mobile:
            user_info['mobile'] = mobile
    if 'location' not in st.session_state.user_data:
        location = st.text_input("Your location (optional):", key="location")
        if location:
            user_info['location'] = location
    if 'purchase_status' not in st.session_state.user_data:
        purchase_status = st.radio(
            "Have you purchased GAPL Starter before?",
            ['Yes', 'No', 'Planning to purchase'], key="purchase_status"
        )
        user_info['purchase_status'] = purchase_status
    if 'crop' not in st.session_state.user_data:
        crop = st.text_input("Which crop are you growing/planning to use GAPL Starter for? (optional):", key="crop")
        if crop:
            user_info['crop'] = crop
    if 'name' not in st.session_state.user_data:
        name = st.text_input("Your name (optional):", key="name")
        if name:
            user_info['name'] = name
    return user_info

def save_user_data(user_data):
    """Saves user data to a JSON file"""
    try:
        with open("user_data.json", "w") as f:
            json.dump(user_data, f, indent=4)
        st.success("User data saved successfully!")
    except Exception as e:
        st.error(f"Error saving user data: {str(e)}")

def handle_submit():
    if st.session_state.user_input:
        st.session_state.submitted_question = st.session_state.user_input
        st.session_state.user_input = ""

def main():
    st.title("🌱 GAPL Starter Product Assistant")

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
                cols = st.columns(2)
                for i, question in enumerate(message["questions"]):
                    if cols[i % 2].button(
                        question,
                        key=f"followup_{message['message_id']}_{i}",
                        use_container_width=True
                    ):
                        asyncio.run(process_question(question))


    # Add metadata display in sidebar
    with st.sidebar:
        st.subheader("Session Information")
        if st.session_state.user_data:
            st.write("Collected Information:")
            for key, value in st.session_state.user_data.items():
                if value:
                    st.write(f"- {key.title()}: {value}")
    
    # Input area
    with st.container():
        st.text_input(
            "Ask me anything about GAPL Starter:",
            key="user_input",
            placeholder="Type your question here...",
            on_change=handle_submit
        )
        
        # Process submitted question
        if st.session_state.submitted_question:
            asyncio.run(process_question(st.session_state.submitted_question))
            st.session_state.submitted_question = None
            st.rerun()
        
        cols = st.columns([4, 1])
        # Clear chat button
        if cols[1].button("Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.chat_memory.clear_history()
            st.session_state.message_counter = 0
            st.session_state.user_data = {}
            st.rerun()

if __name__ == "__main__":
    main()
