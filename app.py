#APP.PY
import streamlit as st
from typing import List, Dict, Any
import asyncio
from core import ChatConfig, ChatLogger, ChatMemory, QuestionGenerator, GeminiRAG, ProductDatabase, ChatBot, load_user_data, save_user_data

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

# New user metadata session state
if 'user_metadata' not in st.session_state:
    st.session_state.user_metadata = {
        'product_name': 'GAPL Starter',
        'purchase_status': None,
        'mobile_number': None,
        'crop_name': None,
        'location': None,
        'name': None
    }
if 'metadata_collection_state' not in st.session_state:
    st.session_state.metadata_collection_state = {
        'mobile_collected': False,
        'crop_collected': False,
        'location_collected': False,
        'purchase_collected': False,
        'name_collected': False
    }
if 'show_metadata_form' not in st.session_state:
    st.session_state.show_metadata_form = False

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
    return config

config = initialize_components()

# Load product database
@st.cache_resource
def load_database():
    with open("STARTER.md", "r", encoding="utf-8") as f:
        markdown_content = f.read()
    return markdown_content

try:
    markdown_content = load_database()
except Exception as e:
    st.error(f"Error loading database: {str(e)}")

# Initialize chatbot
@st.cache_resource
def initialize_chatbot(markdown_content):
    chatbot = ChatBot(config)
    chatbot.load_database()
    return chatbot

chatbot = initialize_chatbot(markdown_content)

def collect_user_metadata():
    """Displays and handles the metadata collection form"""
    if st.session_state.show_metadata_form:
        with st.form("metadata_form"):
            st.write("Please help us serve you better by providing some information:")
            
            if not st.session_state.metadata_collection_state['purchase_collected']:
                purchase_status = st.radio(
                    "Have you purchased GAPL Starter before?",
                    ['Yes', 'No', 'Planning to purchase']
                )
            
            if not st.session_state.metadata_collection_state['mobile_collected']:
                mobile = st.text_input(
                    "Your mobile number:",
                    max_chars=10
                )
            
            if not st.session_state.metadata_collection_state['crop_collected']:
                crop = st.text_input(
                    "Which crop are you growing/planning to use GAPL Starter for?"
                )
            
            if not st.session_state.metadata_collection_state['location_collected']:
                location = st.text_input(
                    "Your pincode/location:"
                )
            
            if not st.session_state.metadata_collection_state['name_collected']:
                name = st.text_input(
                    "Your name (optional):"
                )
            
            submitted = st.form_submit_button("Submit")
            
            if submitted:
                if not st.session_state.metadata_collection_state['purchase_collected']:
                    st.session_state.user_metadata['purchase_status'] = purchase_status
                    st.session_state.metadata_collection_state['purchase_collected'] = True
                
                if not st.session_state.metadata_collection_state['mobile_collected'] and mobile:
                    if len(mobile) == 10 and mobile.isdigit():
                        st.session_state.user_metadata['mobile_number'] = mobile
                        st.session_state.metadata_collection_state['mobile_collected'] = True
                    else:
                        st.error("Please enter a valid 10-digit mobile number")
                
                if not st.session_state.metadata_collection_state['crop_collected'] and crop:
                    st.session_state.user_metadata['crop_name'] = crop
                    st.session_state.metadata_collection_state['crop_collected'] = True
                
                if not st.session_state.metadata_collection_state['location_collected'] and location:
                    st.session_state.user_metadata['location'] = location
                    st.session_state.metadata_collection_state['location_collected'] = True
                
                if not st.session_state.metadata_collection_state['name_collected'] and name:
                    st.session_state.user_metadata['name'] = name
                    st.session_state.metadata_collection_state['name_collected'] = True
                
                # Check if all metadata is collected
                if all(st.session_state.metadata_collection_state.values()):
                    st.session_state.show_metadata_form = False
                    st.rerun()

async def process_question(question: str):
    """Enhanced question processing with metadata collection triggers"""
    try:
        relevant_docs = db.search(question)
        context = rag.create_context(relevant_docs)
        answer = await rag.get_answer(question, context)
        follow_up_questions = await question_gen.generate_questions(question, answer)

        # Save interaction in memory and log
        st.session_state.chat_memory.add_interaction(question, answer)
        logger.log_interaction(question, answer)

        # Check if user data was provided in the response, save if so
        user_data = collect_user_data()
        st.session_state.user_data.update(user_data)
        if all(st.session_state.user_data.values()):
            save_user_data(st.session_state.user_data)

        # Append messages for display
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

def main():
    st.title("🌱 GAPL Starter Product Assistant")

    collect_user_metadata()
    
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
        if any(st.session_state.user_metadata.values()):
            st.write("Collected Information:")
            for key, value in st.session_state.user_metadata.items():
                if value:
                    st.write(f"- {key.replace('_', ' ').title()}: {value}")
    
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
            st.rerun()

if __name__ == "__main__":
    main()
