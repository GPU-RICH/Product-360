import streamlit as st
from typing import List, Dict, Any
import asyncio
from core import (
    ChatConfig, ChatLogger, ChatMemory, QuestionGenerator, 
    GeminiRAG, ProductDatabase, Language, CustomerInfo, 
    CustomerDatabase, ResponseParser
)

# Initialize session state
if 'chat_memory' not in st.session_state:
    st.session_state.chat_memory = ChatMemory()
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'language' not in st.session_state:
    st.session_state.language = Language.HINDI
if 'customer_verified' not in st.session_state:
    st.session_state.customer_verified = False
if 'message_counter' not in st.session_state:
    st.session_state.message_counter = 0
if 'submitted_message' not in st.session_state:
    st.session_state.submitted_message = None
if 'initial_greeting_sent' not in st.session_state:
    st.session_state.initial_greeting_sent = False
if 'data_collection_started' not in st.session_state:
    st.session_state.data_collection_started = False

# Configure the page
st.set_page_config(
    page_title="GAPL Starter सहायक",
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
    parser = ResponseParser(config.gemini_api_key)
    customer_db = CustomerDatabase()
    return config, logger, question_gen, rag, db, parser, customer_db

config, logger, question_gen, rag, db, parser, customer_db = initialize_components()

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

async def process_user_response(response: str):
    data_state = st.session_state.chat_memory.data_collection_state
    current_question = data_state.get_next_question(st.session_state.language)
    current_field = data_state.fields[data_state.current_question_index]
    
    is_valid, extracted_value = await parser.parse_user_response(
        current_question, 
        response, 
        current_field
    )
    
    if is_valid:
        data_state.store_answer(extracted_value)
        
        if data_state.is_complete():
            # Create and save customer info
            customer_info = CustomerInfo(
                mobile=data_state.collected_data['mobile'],
                location=data_state.collected_data['location'],
                purchase_status=data_state.collected_data['purchase_status'],
                crop_type=data_state.collected_data['crop_type'],
                name=data_state.collected_data.get('name'),
                data_collection_complete=True
            )
            
            customer_db.save_customer(customer_info)
            st.session_state.chat_memory.set_customer_info(customer_info)
            st.session_state.customer_verified = True
            
            # Add completion message
            completion_msg = {
                Language.HINDI: "धन्यवाद! अब आप GAPL Starter के बारे में कोई भी प्रश्न पूछ सकते हैं।",
                Language.ENGLISH: "Thank you! You can now ask any questions about GAPL Starter."
            }
            
            st.session_state.messages.append({
                "role": "assistant",
                "content": completion_msg[st.session_state.language],
                "message_id": st.session_state.message_counter
            })
            st.session_state.message_counter += 1
            
        else:
            # Ask next question
            next_question = data_state.get_next_question(st.session_state.language)
            st.session_state.messages.append({
                "role": "assistant",
                "content": next_question,
                "message_id": st.session_state.message_counter
            })
            st.session_state.message_counter += 1
    else:
        # Invalid response message
        error_msgs = {
            Language.HINDI: f"कृपया सही {current_field} प्रदान करें।",
            Language.ENGLISH: f"Please provide a valid {current_field}."
        }
        st.session_state.messages.append({
            "role": "assistant",
            "content": error_msgs[st.session_state.language],
            "message_id": st.session_state.message_counter
        })
        st.session_state.message_counter += 1

async def process_question(question: str):
    try:
        relevant_docs = db.search(question)
        context = rag.create_context(relevant_docs)
        answer = await rag.get_answer(
            question, 
            context, 
            st.session_state.language,
            st.session_state.chat_memory.customer_info
        )
        
        follow_up_questions = await question_gen.generate_questions(
            question, 
            answer,
            st.session_state.language
        )
        
        st.session_state.chat_memory.add_interaction(question, answer)
        logger.log_interaction(
            question, 
            answer, 
            st.session_state.chat_memory.customer_info
        )
        
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
        st.session_state.submitted_message = st.session_state.user_input
        st.session_state.user_input = ""

def main():
    # Language toggle
    col1, col2 = st.columns([4, 1])
    with col2:
        if st.button(
            "Switch to English" if st.session_state.language == Language.HINDI else "हिंदी में बदलें"
        ):
            st.session_state.language = (
                Language.ENGLISH if st.session_state.language == Language.HINDI 
                else Language.HINDI
            )
            st.rerun()
    
    with col1:
        st.title(
            "🌱 GAPL स्टार्टर उत्पाद सहायक" 
            if st.session_state.language == Language.HINDI 
            else "🌱 GAPL Starter Product Assistant"
        )

    # Initial greeting
    if not st.session_state.initial_greeting_sent:
        greeting = {
            Language.HINDI: "नमस्ते! मैं GAPL स्टार्टर प्रोडक्ट असिस्टेंट हूं। आपकी सहायता के लिए कुछ जानकारी चाहिए।",
            Language.ENGLISH: "Hello! I'm the GAPL Starter Product Assistant. I need some information to assist you better."
        }
        
        st.session_state.messages.append({
            "role": "assistant",
            "content": greeting[st.session_state.language],
            "message_id": st.session_state.message_counter
        })
        st.session_state.message_counter += 1
        st.session_state.initial_greeting_sent = True

        # Start data collection
        first_question = st.session_state.chat_memory.data_collection_state.get_next_question(
            st.session_state.language
        )
        st.session_state.messages.append({
            "role": "assistant",
            "content": first_question,
            "message_id": st.session_state.message_counter
        })
        st.session_state.message_counter += 1
        st.session_state.data_collection_started = True

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
            "आपका जवाब यहां टाइप करें:" if st.session_state.language == Language.HINDI 
            else "Type your response here:",
            key="user_input",
            on_change=handle_submit
        )
        
        if st.session_state.submitted_message:
            message = st.session_state.submitted_message
            st.session_state.submitted_message = None
            
            st.session_state.messages.append({
                "role": "user",
                "content": message,
                "message_id": st.session_state.message_counter
            })
            st.session_state.message_counter += 1
            
            if not st.session_state.customer_verified:
                asyncio.run(process_user_response(message))
            else:
                asyncio.run(process_question(message))
            
            st.rerun()

        # Clear chat button
        if st.button(
            "चैट साफ़ करें" if st.session_state.language == Language.HINDI else "Clear Chat"
        ):
            st.session_state.messages = []
            st.session_state.chat_memory.clear_history()
            st.session_state.message_counter = 0
            st.session_state.customer_verified = False
            st.session_state.initial_greeting_sent = False
            st.session_state.data_collection_started = False
            st.rerun()

if __name__ == "__main__":
    main()
