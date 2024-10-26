import streamlit as st
from typing import List, Dict, Any
import asyncio
from core import ChatConfig, ChatLogger, ChatMemory, QuestionGenerator, GeminiRAG, ProductDatabase, Language, CustomerInfo

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
if 'submitted_question' not in st.session_state:
    st.session_state.submitted_question = None
if 'initial_questions' not in st.session_state:
    st.session_state.initial_questions = {
        Language.HINDI: [
            "GAPL स्टार्टर के मुख्य लाभ क्या हैं?",
            "GAPL स्टार्टर का सही प्रयोग कैसे करें?",
            "GAPL स्टार्टर किन फसलों के लिए उपयुक्त है?",
            "अनुशंसित खुराक क्या है?"
        ],
        Language.ENGLISH: [
            "What are the main benefits of GAPL Starter?",
            "How do I apply GAPL Starter correctly?",
            "Which crops is GAPL Starter suitable for?",
            "What is the recommended dosage?"
        ]
    }

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
.customer-info {
    background-color: #f0f8ff;
    padding: 20px;
    border-radius: 10px;
    margin: 10px 0;
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

def validate_mobile(mobile: str) -> bool:
    return mobile.isdigit() and len(mobile) == 10

def collect_customer_info():
    st.markdown(f"### {config.greetings[st.session_state.language]}")
    
    with st.form("customer_info_form"):
        mobile = st.text_input(
            config.customer_info_prompts[st.session_state.language]["mobile"],
            key="mobile"
        )
        location = st.text_input(
            config.customer_info_prompts[st.session_state.language]["location"],
            key="location"
        )
        purchase_status = st.selectbox(
            config.customer_info_prompts[st.session_state.language]["purchase_status"],
            ["हाँ", "नहीं"] if st.session_state.language == Language.HINDI else ["Yes", "No"]
        )
        crop_type = st.text_input(
            config.customer_info_prompts[st.session_state.language]["crop_type"],
            key="crop_type"
        )
        name = st.text_input(
            config.customer_info_prompts[st.session_state.language]["name"],
            key="name"
        )
        
        submit_button = st.form_submit_button(
            "जमा करें" if st.session_state.language == Language.HINDI else "Submit"
        )
        
        if submit_button:
            if not validate_mobile(mobile):
                st.error(
                    "कृपया 10 अंकों का वैध मोबाइल नंबर दर्ज करें" 
                    if st.session_state.language == Language.HINDI 
                    else "Please enter a valid 10-digit mobile number"
                )
                return False
            
            if not all([location, purchase_status, crop_type]):
                st.error(
                    "कृपया सभी आवश्यक जानकारी भरें"
                    if st.session_state.language == Language.HINDI
                    else "Please fill all required information"
                )
                return False
            
            customer_info = CustomerInfo(
                mobile=mobile,
                location=location,
                purchase_status=purchase_status,
                crop_type=crop_type,
                name=name if name else None
            )
            st.session_state.chat_memory.set_customer_info(customer_info)
            return True
    
    return False

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
        st.session_state.submitted_question = st.session_state.user_input
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
    
    # Collect customer information if not already verified
    if not st.session_state.customer_verified:
        if collect_customer_info():
            st.session_state.customer_verified = True
            st.rerun()
        return
    
    # Display customer info
    customer_info = st.session_state.chat_memory.customer_info
    if customer_info:
        with st.expander(
            "ग्राहक जानकारी" if st.session_state.language == Language.HINDI 
            else "Customer Information"
        ):
            info_text = f"""
            {'मोबाइल' if st.session_state.language == Language.HINDI else 'Mobile'}: {customer_info.mobile}
            {'स्थान' if st.session_state.language == Language.HINDI else 'Location'}: {customer_info.location}
            {'फसल' if st.session_state.language == Language.HINDI else 'Crop'}: {customer_info.crop_type}
            """
            if customer_info.name:
                info_text += f"{'नाम' if st.session_state.language == Language.HINDI else 'Name'}: {customer_info.name}"
            st.markdown(info_text)
    
    # Welcome message
    if not st.session_state.messages:
        welcome_msg = {
            Language.HINDI: """
            👋 नमस्ते! मैं आपका GAPL स्टार्टर प्रोडक्ट विशेषज्ञ हूं। मैं इन विषयों में आपकी मदद कर सकता हूं:
            - उत्पाद के लाभ और विशेषताएं
            - उपयोग की विधियां और समय
            - खुराक की सिफारिशें
            - फसल अनुकूलता
            - तकनीकी विवरण
            
            नीचे दिए गए प्रश्न चुनें या अपना प्रश्न पूछें!
            """,
            Language.ENGLISH: """
            👋 Welcome! I'm your GAPL Starter product expert. I can help you learn about:
            - Product benefits and features
            - Application methods and timing
            - Dosage recommendations
            - Crop compatibility
            - Technical specifications
            
            Choose a question below or ask your own!
            """
        }
        st.markdown(welcome_msg[st.session_state.language])
        
        # Display initial questions as buttons
        cols = st.columns(2)
        current_questions = st.session_state.initial_questions[st.session_state.language]
        for i, question in enumerate(current_questions):
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
            "GAPL स्टार्टर के बारे में कुछ भी पूछें:" 
            if st.session_state.language == Language.HINDI 
            else "Ask me anything about GAPL Starter:",
            key="user_input",
            placeholder="यहां अपना प्रश्न लिखें..." 
            if st.session_state.language == Language.HINDI 
            else "Type your question here...",
            on_change=handle_submit
        )
        
        # Process submitted question
        if st.session_state.submitted_question:
            asyncio.run(process_question(st.session_state.submitted_question))
            st.session_state.submitted_question = None
            st.rerun()
        
        cols = st.columns([4, 1])
        # Clear chat button
        if cols[1].button(
            "चैट साफ़ करें" if st.session_state.language == Language.HINDI else "Clear Chat", 
            use_container_width=True
        ):
            st.session_state.messages = []
            st.session_state.chat_memory.clear_history()
            st.session_state.message_counter = 0
            st.session_state.customer_verified = False
            st.rerun()

if __name__ == "__main__":
    main()
