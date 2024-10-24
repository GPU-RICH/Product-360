import streamlit as st
from typing import List, Dict, Any
import asyncio
from core import ChatConfig, ChatLogger, ChatMemory, ProductDatabase

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

# User metadata session state
if 'user_metadata' not in st.session_state:
    st.session_state.user_metadata = {
        'product_name': 'GAPL Starter',
        'purchase_status': None,
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
                
                # Check if all metadata is collected
                if all(st.session_state.metadata_collection_state.values()):
                    st.session_state.show_metadata_form = False
                    st.rerun()
class EnhancedQuestionGenerator:
    """Enhanced question generator that integrates metadata collection"""
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.generation_config = {
            "temperature": 0.2,
            "top_p": 0.95,
            "top_k": 64,
            "max_output_tokens": 2048,
        }
        self.model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            generation_config=self.generation_config
        )
        
    async def generate_questions(self, question: str, answer: str) -> List[str]:
        try:
            metadata = st.session_state.user_metadata
            chat = self.model.start_chat(history=[])
            
            # Determine which metadata question to ask based on what's missing
            metadata_question = None
            if not metadata['purchase_status'] and 'benefit' in question.lower():
                metadata_question = "Have you already purchased GAPL Starter for your farm?"
            elif not metadata['crop_name'] and ('crop' in question.lower() or 'apply' in question.lower()):
                metadata_question = "Which crops are you currently growing or planning to use GAPL Starter with?"
            elif not metadata['mobile_number'] and metadata['purchase_status']:
                metadata_question = "To provide you with specific guidance, could you share your mobile number?"
            elif not metadata['location'] and metadata['crop_name']:
                metadata_question = "What's your location/pincode? This helps me provide region-specific recommendations."

            prompt = f"""Based on this product information interaction:
            
            Question: {question}
            Answer: {answer}
            
            Generate 3 relevant follow-up questions about GAPL Starter.
            Focus on:
            - Application methods and timing
            - Benefits and effectiveness
            - Compatibility with specific crops
            - Scientific backing and results
            
            Return ONLY the numbered questions (1-3), one per line.
            """
            
            response = chat.send_message(prompt).text
            
            questions = []
            for line in response.split('\n'):
                line = line.strip()
                if line and any(line.startswith(f'{i}.') for i in range(1, 4)):
                    questions.append(line.split('.', 1)[1].strip())
            
            # Add metadata question if applicable
            if metadata_question and len(questions) > 0:
                questions[-1] = metadata_question
            
            return questions[:3]
            
        except Exception as e:
            logging.error(f"Error generating questions: {str(e)}")
            return [
                "How should I store GAPL Starter?",
                "What results can I expect to see?",
                "Could you share which crops you're growing?"
            ]
async def process_question(question: str):
    """Enhanced question processing with metadata collection"""
    try:
        # Check if the question is answering a metadata question
        metadata = st.session_state.user_metadata
        
        # Process metadata responses
        question_lower = question.lower()
        if "purchased" in question_lower or "bought" in question_lower:
            metadata['purchase_status'] = question
        elif any(crop in question_lower for crop in ['growing', 'farming', 'cultivating']):
            metadata['crop_name'] = question
        elif question.replace(" ", "").isdigit() and len(question.replace(" ", "")) == 10:
            metadata['mobile_number'] = question.replace(" ", "")
        elif any(loc in question_lower for loc in ['pincode', 'location', 'district', 'village']):
            metadata['location'] = question
        
        # Regular question processing
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
        
        # Add small metadata display
        if any(v for k, v in st.session_state.user_metadata.items() if k != 'product_name'):
            with st.expander("Session Info"):
                for key, value in st.session_state.user_metadata.items():
                    if value and key != 'product_name':
                        st.write(f"- {key.replace('_', ' ').title()}: {value}")

if __name__ == "__main__":
    main()
