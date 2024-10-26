import streamlit as st
from typing import List, Dict, Any, Optional
import asyncio
from core import (
    ChatConfig, ChatLogger, ChatMemory, QuestionGenerator, 
    GeminiRAG, ProductDatabase, FarmerInfo
)

# Initialize session state
if 'chat_memory' not in st.session_state:
    st.session_state.chat_memory = ChatMemory()
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'message_counter' not in st.session_state:
    st.session_state.message_counter = 0
if 'submitted_question' not in st.session_state:
    st.session_state.submitted_question = None
if 'farmer_info' not in st.session_state:
    st.session_state.farmer_info = None
if 'language' not in st.session_state:
    st.session_state.language = "english"

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
.farmer-info {
    background-color: #f0f8ff;
    padding: 10px;
    border-radius: 5px;
    margin-bottom: 20px;
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

def farmer_info_to_dict(farmer_info: FarmerInfo) -> dict:
    """Convert FarmerInfo object to dictionary"""
    return {
        "mobile": farmer_info.mobile,
        "location": farmer_info.location,
        "crop_type": farmer_info.crop_type,
        "purchase_status": farmer_info.purchase_status,
        "name": farmer_info.name
    }

def dict_to_farmer_info(data: dict) -> FarmerInfo:
    """Convert dictionary to FarmerInfo object"""
    return FarmerInfo(
        mobile=data["mobile"],
        location=data["location"],
        crop_type=data["crop_type"],
        purchase_status=data["purchase_status"],
        name=data["name"]
    )
    
def initialize_farmer_form():
    """Create a form to collect farmer information"""
    st.markdown("### 👨‍🌾 Welcome to GAPL Starter Assistant!")
    
    with st.form("farmer_info"):
        st.markdown("""
        To provide you with the most relevant advice for your farm, we need some basic information. 
        This helps us customize our recommendations based on your specific needs and location.
        """)
        
        mobile = st.text_input("📱 Mobile Number*", 
                             help="We'll use this to send you important updates about GAPL Starter")
        
        location = st.text_input("📍 Your Location (District/State)*",
                               help="This helps us provide location-specific guidance")
        
        crop_type = st.selectbox("🌾 Main Crop*",
                                ["Rice", "Wheat", "Cotton", "Sugarcane", "Vegetables", "Other"],
                                help="Select your primary crop for targeted advice")
        
        purchase_status = st.radio("🛒 GAPL Starter Purchase Status*",
                                 ["Already using", "Planning to buy", "Just exploring"],
                                 help="This helps us tailor our guidance to your needs")
        
        name = st.text_input("👤 Your Name (Optional)",
                           help="We'd love to address you personally")
        
        submitted = st.form_submit_button("Start Chat")
        
        if submitted:
            if mobile and location and crop_type and purchase_status:
                farmer_info = FarmerInfo(
                    mobile=mobile,
                    location=location,
                    crop_type=crop_type,
                    purchase_status=purchase_status,
                    name=name if name else None
                )
                return farmer_info_to_dict(farmer_info)  # Return dictionary instead of FarmerInfo object
            else:
                st.error("Please fill in all required fields marked with *")
                return None
        return None

async def process_question(question: str):
    try:
        # Convert dictionary back to FarmerInfo object for processing
        farmer_info = dict_to_farmer_info(st.session_state.farmer_info) if st.session_state.farmer_info else None
        
        relevant_docs = db.search(question)
        context = rag.create_context(relevant_docs)
        answer = await rag.get_answer(
            question, 
            context, 
            farmer_info,
            st.session_state.language
        )
        follow_up_questions = await question_gen.generate_questions(
            question, 
            answer,
            farmer_info,
            st.session_state.language
        )
        
        st.session_state.chat_memory.add_interaction(
            question, 
            answer,
            farmer_info
        )
        logger.log_interaction(
            question, 
            answer,
            farmer_info
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

def display_farmer_info():
    """Display current farmer information"""
    farmer_dict = st.session_state.farmer_info
    if farmer_dict:
        with st.sidebar:
            st.markdown("### 👨‍🌾 Farmer Details")
            st.markdown(f"""
            🌾 **Crop**: {farmer_dict['crop_type']}  
            📍 **Location**: {farmer_dict['location']}  
            🛒 **Status**: {farmer_dict['purchase_status']}  
            👤 **Name**: {farmer_dict['name'] or 'Anonymous'}
            """)

def display_initial_questions():
    """Display initial contextual questions based on farmer info"""
    farmer = st.session_state.farmer_info
    if farmer.purchase_status == "Already using":
        questions = [
            "What are the best practices for applying GAPL Starter?",
            f"How can I maximize GAPL Starter's benefits for {farmer.crop_type}?",
            "When should I apply the next dose?",
            "Can you share success stories from other farmers?"
        ]
    elif farmer.purchase_status == "Planning to buy":
        questions = [
            "What are the main benefits of GAPL Starter?",
            f"Is GAPL Starter suitable for {farmer.crop_type}?",
            "What is the recommended dosage and cost?",
            "How quickly can I expect to see results?"
        ]
    else:  # Just exploring
        questions = [
            "What makes GAPL Starter different from other products?",
            f"How does GAPL Starter work with {farmer.crop_type}?",
            "What results have other farmers seen?",
            "What is the cost-benefit ratio?"
        ]
    
    cols = st.columns(2)
    for i, question in enumerate(questions):
        if cols[i % 2].button(question, key=f"initial_{i}", use_container_width=True):
            asyncio.run(process_question(question))

def display_farmer_info():
    """Display current farmer information"""
    farmer = st.session_state.farmer_info
    if farmer:
        with st.sidebar:
            st.markdown("### 👨‍🌾 Farmer Details")
            st.markdown(f"""
            🌾 **Crop**: {farmer.crop_type}  
            📍 **Location**: {farmer.location}  
            🛒 **Status**: {farmer.purchase_status}  
            👤 **Name**: {farmer.name or 'Anonymous'}
            """)

def handle_submit():
    if st.session_state.user_input:
        st.session_state.submitted_question = st.session_state.user_input
        st.session_state.user_input = ""

def main():
    # Language toggle in sidebar
    with st.sidebar:
        st.title("🌐 Language / भाषा")
        language = st.toggle("Switch to Hindi / हिंदी में बदलें", 
                           help="Toggle between English and Hindi")
        st.session_state.language = "hindi" if language else "english"
    
    # Main content
    if not st.session_state.farmer_info:
        farmer_info_dict = initialize_farmer_form()
        if farmer_info_dict:
            st.session_state.farmer_info = farmer_info_dict  # Store dictionary in session state
            st.rerun()
    else:
        st.title("🌱 GAPL Starter Product Assistant")
        display_farmer_info()
        
        # Welcome message
        if not st.session_state.messages:
            greeting = f"Namaste" if st.session_state.language == "hindi" else "Hello"
            name = f" {st.session_state.farmer_info.name}" if st.session_state.farmer_info.name else ""
            st.markdown(f"""
            👋 {greeting}{name}! I'm your GAPL Starter product expert. I can help you with:
            - Product benefits and features
            - Application methods and timing
            - Dosage recommendations
            - {st.session_state.farmer_info.crop_type} specific guidance
            - Technical specifications
            
            Choose a question below or ask your own!
            """)
            
            display_initial_questions()
        
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
            
            # Action buttons
            cols = st.columns([4, 1, 1])
            
            # Edit Info button
            if cols[1].button("Edit Info", use_container_width=True):
                st.session_state.farmer_info = None
                st.rerun()
            
            # Clear Chat button
            if cols[2].button("Clear Chat", use_container_width=True):
                st.session_state.messages = []
                st.session_state.chat_memory.clear_history()
                st.session_state.message_counter = 0
                st.rerun()

if __name__ == "__main__":
    main()
