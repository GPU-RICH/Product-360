import streamlit as st
from typing import List, Dict, Any, Optional
import asyncio
from PIL import Image
import io
from back import (ChatConfig, ChatLogger, ChatMemory, QuestionGenerator, 
                  GeminiRAG, ProductDatabase, UserManager, UserInfo)

# Product Configuration
# In APP.PY, update the PRODUCT_CONFIG dictionary:

PRODUCT_CONFIG = {
    "GAPL STARTER 1KG": {
        "markdown_file": "STARTER.md",
        "title": "🌱 GAPL STARTER 1KG उत्पाद सहायक",
        "video_url": "https://www.youtube.com/embed/9kHyeQ9B7TQ",
        "suggestions": [
            {"image": "INNO_AG.webp", "name": "INNO AG Stimgo MGR 1 Kg"},
            {"image": "IFFCO.webp", "name": "IFFCO Sagarika Bucket 10 Kg"},
            {"image": "ORGA.webp", "name": "ORGANIC PDM 50 Kg"}
        ],
        "initial_questions": [
            "उत्पाद के मुख्य लाभ क्या हैं?",
            "उत्पाद का प्रयोग कैसे करें?",
            "उत्पाद किन फसलों के लिए उपयुक्त है?",
            "अनुशंसित मात्रा क्या है?"
        ]
    },
    "ENTOKILL 250ML": {
        "markdown_file": "ENTOKILL.md",
        "title": "🌱 Entokill 250ml उत्पाद सहायक",
        "video_url": "https://www.youtube.com/embed/EY489XtDYEo",
        "suggestions": [
            {"image": "paras-npk.webp", "name": "PARAS NPK 12:32:16 50 Kg"},
            {"image": "mosaic.webp", "name": "MOSAIC MOP 50 Kg"},
            {"image": "paras_dap.webp", "name": "PARAS DAP 50 Kg"}
        ],
        "initial_questions": [
            "उत्पाद के मुख्य लाभ क्या हैं?",
            "उत्पाद का प्रयोग कैसे करें?",
            "उत्पाद किन फसलों के लिए उपयुक्त है?",
            "अनुशंसित मात्रा क्या है?"
        ]
    },
    "DEHAAT KHURAK 3000": {
        "markdown_file": "KHURAK.md",
        "title": "🌱 DeHaat Khurak 3000 उत्पाद सहायक",
        "video_url": "https://www.youtube.com/embed/Q55lYWMu40o",
        "suggestions": [
            {"image": "doodh_plus.webp", "name": "Doodh Plus 5 Kg"},
            {"image": "balance.webp", "name": "DEHAAT BALANCE DIET 25 KG"},
            {"image": "vetnoliv.webp", "name": "Vetnoliv 1 L"}
        ],
        "initial_questions": [
            "इस फीड का इस्तेमाल करने से दूध उत्पादन में कितनी वृद्धि होगी?",
            "क्या यह फीड गर्भवती पशुओं के लिए सुरक्षित है?",
            "पशु को प्रतिदिन कितनी मात्रा में फीड देना चाहिए?",
            "क्या इस फीड से पशु के स्वास्थ्य में कोई और सुधार होगा?"
        ]
    },
    "DOODH PLUS": {
        "markdown_file": "doodhplus.md",
        "title": "🌱 Doodh Plus उत्पाद सहायक",
        "video_url": "https://www.youtube.com/embed/3_Geihsy1KM",
        "suggestions": [
            {"image": "doodh_khurak.webp", "name": "DeHaat Khurak 5000 45 Kg"},
            {"image": "vetnocal.webp", "name": "Vetnocal Gold 5 L"},
            {"image": "kriya.webp", "name": "KriyaPro"}
        ],
        "initial_questions": [
            "दूध प्लस के नियमित इस्तेमाल से दूध की गुणवत्ता में क्या सुधार होगा?",
            "क्या यह उत्पाद पशु की प्रजनन क्षमता में सुधार करता है?",
            "इस उत्पाद को किस प्रकार खिलाना चाहिए?",
            "क्या इससे दूध में फैट की मात्रा बढ़ेगी?"
        ]
    }
}
# UI Text in Hindi
UI_TEXT = {
    "welcome_message": """
    Product 360 अनुभव में आपका स्वागत है! 🌾
    
    नमस्ते! अब आप उन 10,000+ किसानों के परिवार का हिस्सा बन गए हैं जो इस उत्पाद का उपयोग करते हैं।
    
    आपने {product} का QR कोड स्कैन किया है—यहां आपको इस उत्पाद से जुड़ी सभी ज़रूरी जानकारी मिलेगी, साथ ही उन किसानों की कहानियाँ भी, जिन्होंने इस उत्पाद से बेहतरीन परिणाम हासिल किए हैं।
    
    नीचे दिए गए सवालों में से कोई चुनें या अपना खुद का सवाल पूछें
    """,
    "product_select": "उत्पाद चुनें:",
    "input_placeholder": "अपना प्रश्न यहां टाइप करें...",
    "input_label": "कुछ भी पूछें:",
    "clear_chat": "चैट साफ़ करें",
    "sidebar_title": "अपनी जानकारी भरें और अगली खरीदारी पर 15% की छूट पाएं !!",
    "form_name": "आपका नाम",
    "form_mobile": "मोबाइल नंबर",
    "form_location": "स्थान",
    "form_purchase": "क्या आपने उत्पाद खरीदा है?",
    "form_crop": "यदि हाँ, तो इस उत्पाद से आपको क्या लाभ हुआ है?",
    "form_submit": "15% छूट कूपन सक्रिय करें",
    "form_success": "✅ 15% छूट कूपन सक्रिय !",
    "form_error": "❌ जानकारी सहेजने में त्रुटि। कृपया पुनः प्रयास करें।",
    "form_required": "कृपया सभी आवश्यक फ़ील्ड भरें।",
    "image_upload": "यदि आपके पास अपनी फसल के बारे में कोई चिंता या प्रश्न है, तो उसकी छवि अपलोड करें।",
    "image_helper": "यदि आपके पास अपनी फसल के बारे में कोई चिंता या प्रश्न है, तो उसकी छवि अपलोड करें।",
    "image_processing": "आपकी छवि प्र्रोसेस की जा रही है...",
    "show_suggestions": "संबंधित उत्पाद दिखाएं",
    "suggestions_title": "संबंधित उत्पाद:",
    "initial_questions": [
        "उत्पाद के मुख्य लाभ क्या हैं?",
        "उत्पाद का प्रयोग कैसे करें?",
        "उत्पाद किन फसलों के लिए उपयुक्त है?",
        "अनुशंसित मात्रा क्या है?"
    ]
}

UI_TEXT.update({
    "video_title": "उत्पाद डेमो वीडियो",
    "video_loading": "वीडियो लोड हो रहा है...",
    "video_error": "वीडियो लोड करने में समस्या हुई। कृपया पुनः प्रयास करें।"
})

def init_session_state():
    """Initialize all session state variables"""
    defaults = {
        'initialized': False,
        'chat_memory': ChatMemory(),
        'messages': [],
        'message_counter': 0,
        'processed_questions': set(),
        'trigger_rerun': False,
        'user_info': None,
        'show_suggestions': False,
        'selected_product': list(PRODUCT_CONFIG.keys())[0]
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
      
# Configure the page
st.set_page_config(
    page_title="Product Assistant",
    page_icon="🌱",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
.user-message {
    background-color: #98BF64;
    color: black;
    padding: 15px;
    border-radius: 15px;
    margin: 10px 0;
}
.assistant-message {
    background-color: #363E35;
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
.image-upload {
    margin: 20px 0;
    padding: 15px;
    border-radius: 10px;
    border: 2px dashed #72BF6A;
}
.product-card {
    border: 1px solid #ddd;
    border-radius: 10px;
    padding: 15px;
    margin: 10px;
    background-color: white;
    text-align: center;
}
.related-products {
    padding: 20px;
    margin: 20px 0;
    background-color: #f5f5f5;
    border-radius: 10px;
}
.video-container {
    position: relative;
    padding-bottom: 56.25%;
    margin: 20px 0;
    border-radius: 10px;
    overflow: hidden;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}
.video-container iframe {
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    border: none;
}
</style>
""", unsafe_allow_html=True)

def display_product_suggestions():
    if st.session_state.show_suggestions:
        st.sidebar.markdown("---")
        st.sidebar.markdown(f"### {UI_TEXT['suggestions_title']}")
        
        product_config = PRODUCT_CONFIG[st.session_state.selected_product]
        for suggestion in product_config['suggestions']:
            with st.sidebar.container():
                st.image(suggestion['image'], caption=suggestion['name'], use_column_width=True)
                st.markdown(f"**{suggestion['name']}**")
                st.markdown("---")
                    
# Initialize components
@st.cache_resource
def initialize_components():
    """Initialize and cache application components"""
    try:
        config = ChatConfig()
        logger = ChatLogger(config.log_file)
        question_gen = QuestionGenerator(config.gemini_api_key)
        rag = GeminiRAG(config.gemini_api_key)
        user_manager = UserManager(config.user_data_file)
        return config, logger, question_gen, rag, user_manager
    except Exception as e:
        st.error(f"Error initializing components: {str(e)}")
        raise e

def load_initial_database():
    """Load the default product database"""
    if not st.session_state.initialized:
        try:
            config, logger, question_gen, rag, user_manager = initialize_components()
            default_product = list(PRODUCT_CONFIG.keys())[0]
            markdown_file = PRODUCT_CONFIG[default_product]['markdown_file']
            
            db = ProductDatabase(config)
            with open(markdown_file, "r", encoding="utf-8") as f:
                markdown_content = f.read()
            db.process_markdown(markdown_content)
            
            st.session_state.db = db
            st.session_state.config = config
            st.session_state.logger = logger
            st.session_state.question_gen = question_gen
            st.session_state.rag = rag
            st.session_state.user_manager = user_manager
            st.session_state.initialized = True
            
        except Exception as e:
            st.error(f"Error loading initial database: {str(e)}")
            return None
def load_new_database(product_name: str):
    """Load a new product database when product selection changes"""
    try:
        markdown_file = PRODUCT_CONFIG[product_name]['markdown_file']
        db = ProductDatabase(st.session_state.config)
        with open(markdown_file, "r", encoding="utf-8") as f:
            markdown_content = f.read()
        db.process_markdown(markdown_content)
        st.session_state.db = db
    except Exception as e:
        st.error(f"डेटाबेस लोड करने में त्रुटि: {str(e)}")

def on_product_change():
    """Handle product selection change"""
    load_new_database(st.session_state.selected_product)
    st.session_state.messages = []
    st.session_state.chat_memory.clear_history()
    st.session_state.message_counter = 0

async def process_question(question: str, image: Optional[bytes] = None):
    """Process a question and update the chat state"""
    try:
        # If image is provided, use image processing
        if image:
            answer = await st.session_state.rag.get_answer(
                question=question,
                context="",  # No context needed for image analysis
                user_info=st.session_state.user_info,
                image=image
            )
        else:
            # Normal text-based processing
            relevant_docs = st.session_state.db.search(question)
            context = st.session_state.rag.create_context(relevant_docs)
            answer = await st.session_state.rag.get_answer(
                question=question,
                context=context,
                user_info=st.session_state.user_info,
                image=None
            )
        
        follow_up_questions = await st.session_state.question_gen.generate_questions(
            question, 
            answer,
            st.session_state.user_info
        )
        
        st.session_state.chat_memory.add_interaction(question, answer)
        st.session_state.logger.log_interaction(
            question, 
            answer,
            st.session_state.user_info
        )
        
        st.session_state.message_counter += 1
        
        # Create message content
        message_content = {
            "text": question,
            "has_image": image is not None
        }
        
        st.session_state.messages.append({
            "role": "user",
            "content": message_content,
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

def render_user_form():
    """Render the user information form in the sidebar"""
    st.sidebar.title(UI_TEXT["sidebar_title"])
    
    with st.sidebar.form("user_info_form"):
        name = st.text_input(UI_TEXT["form_name"])
        mobile = st.text_input(UI_TEXT["form_mobile"])
        location = st.text_input(UI_TEXT["form_location"])
        has_purchased = st.checkbox(UI_TEXT["form_purchase"])
        crop_type = st.text_input(UI_TEXT["form_crop"])
        
        submitted = st.form_submit_button(UI_TEXT["form_submit"])
        
        if submitted:
            if name and mobile and location and crop_type:
                user_info = UserInfo(
                    name=name,
                    mobile=mobile,
                    location=location,
                    has_purchased=has_purchased,
                    crop_type=crop_type
                )
                
                if st.session_state.user_manager.save_user_info(user_info):
                    st.session_state.user_info = user_info
                    st.sidebar.success(UI_TEXT["form_success"])
                else:
                    st.sidebar.error(UI_TEXT["form_error"])
            else:
                st.sidebar.warning(UI_TEXT["form_required"])

def main():
    # Initialize session state
    init_session_state()
    
    # Load initial database and components if not already initialized
    if not st.session_state.initialized:
        load_initial_database()
    
    # Product selection in sidebar
    st.sidebar.selectbox(
        UI_TEXT["product_select"],
        options=list(PRODUCT_CONFIG.keys()),
        key="selected_product",
        on_change=on_product_change
    )
    
    # Add suggestions toggle at the top of sidebar
    st.sidebar.checkbox(
        UI_TEXT["show_suggestions"],
        key="show_suggestions",
        value=st.session_state.show_suggestions
    )

    # Render user form in sidebar
    render_user_form()
    
    # Display product suggestions in sidebar if enabled
    if st.session_state.show_suggestions:
        display_product_suggestions()

    # Display product-specific title
    product_config = PRODUCT_CONFIG[st.session_state.selected_product]
    st.title(product_config['title'])

    # Add video section after title
    if "video_url" in product_config:
        st.subheader(UI_TEXT["video_title"])
        with st.container():
            video_html = f"""
                <div class='video-container'>
                    <iframe
                        src='{product_config['video_url']}'
                        frameborder='0'
                        allow='accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture'
                        allowfullscreen>
                    </iframe>
                </div>
                """
            st.markdown(video_html, unsafe_allow_html=True)
            
    # Welcome message
    if not st.session_state.messages:
        st.markdown(UI_TEXT["welcome_message"].format(product=st.session_state.selected_product))
        
        # Display product-specific initial questions as buttons
        product_config = PRODUCT_CONFIG[st.session_state.selected_product]
        initial_questions = product_config["initial_questions"]
        
        cols = st.columns(2)
        for i, question in enumerate(initial_questions):
            if cols[i % 2].button(question, key=f"initial_{i}", use_container_width=True):
                asyncio.run(process_question(question))
    
    # Display chat history
    for message in st.session_state.messages:
        if message["role"] == "user":
            if isinstance(message["content"], dict):
                st.markdown(
                    f'<div class="user-message">👤 {message["content"]["text"]}</div>',
                    unsafe_allow_html=True
                )
                if message["content"]["has_image"]:
                    st.markdown(
                        '<div class="user-message">📷 छवि अपलोड की गई</div>',
                        unsafe_allow_html=True
                    )
            else:
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
    
    # Input area with image upload and form
    with st.container():
        # Add image upload
        uploaded_file = st.file_uploader(
            UI_TEXT["image_upload"],
            type=['png', 'jpg', 'jpeg'],
            help=UI_TEXT["image_helper"],
            key="image_upload"
        )
        
        if uploaded_file:
            try:
                image = Image.open(uploaded_file)
                st.image(image, caption="अपलोड की गई छवि", use_column_width=True)
            except Exception as e:
                st.error("छवि लोड करने में समस्या हुई। कृपया दूसरी छवि आज़माएं।")
        
        # Create a form for input
        with st.form(key='input_form'):
            question = st.text_input(
                UI_TEXT["input_label"],
                key="user_input",
                placeholder=UI_TEXT["input_placeholder"]
            )
            submit = st.form_submit_button("भेजें")
        
        # Process input when submitted
        if submit and question:
            image_bytes = None
            if uploaded_file:
                try:
                    image_bytes = uploaded_file.getvalue()
                except Exception as e:
                    st.error("छवि प्रोसेस करने में समस्या हुई।")
            
            with st.spinner("🔄 आपका संदेश प्रोसेस किया जा रहा है..."):
                asyncio.run(process_question(question, image_bytes))
                if 'processed_questions' not in st.session_state:
                    st.session_state.processed_questions = set()
                st.session_state.processed_questions.add(question)
                st.rerun()
        
        # Clear chat controls
        cols = st.columns([4, 1])
        if cols[1].button(UI_TEXT["clear_chat"], use_container_width=True):
            st.session_state.messages = []
            st.session_state.chat_memory.clear_history()
            st.session_state.message_counter = 0
            if 'processed_questions' in st.session_state:
                st.session_state.processed_questions = set()
            st.rerun()


def handle_submit():
    """Handle the submission of user input"""
    if st.session_state.user_input:
        st.session_state.submitted_question = st.session_state.user_input
        st.session_state.user_input = ""

def handle_error(error: Exception):
    """Handle errors gracefully"""
    error_messages = {
        "generic": "एक त्रुटि हुई। कृपया पुनः प्रयास करें।",
        "image": "छवि प्रोसेसिंग में त्रुटि। कृपया कोई अन्य छवि आज़माएं या बिना छवि के पूछें।",
        "network": "नेटवर्क त्रुटि। कृपया अपना कनेक्शन जांचें और पुनः प्रयास करें।",
        "database": "डेटाबेस लोड करने में त्रुटि। कृपया पेज को रिफ्रेश करें।"
    }
    
    if "image" in str(error).lower():
        error_message = error_messages["image"]
    elif "network" in str(error).lower():
        error_message = error_messages["network"]
    elif "database" in str(error).lower():
        error_message = error_messages["database"]
    else:
        error_message = error_messages["generic"]
    
    st.error(error_message)
    logging.error(f"Error in app: {str(error)}")

if __name__ == "__main__":
    main()
