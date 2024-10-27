import os
import logging
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum
import torch
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
import google.generativeai as genai
from datetime import datetime
import json
from PIL import Image
import io

class Language(Enum):
    ENGLISH = "en-US"
    HINDI = "hi"

@dataclass
class UserInfo:
    """User information for context"""
    name: str
    mobile: str
    location: str
    has_purchased: bool
    crop_type: str

@dataclass
class ChatConfig:
    """Configuration for the chatbot"""
    embedding_model_name: str = 'all-MiniLM-L6-v2'
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    max_history: int = 3
    gemini_api_key: str = "AIzaSyBS_DFCJh82voYIKoglS-ow6ezGNg775pg"  # Replace with your API key
    log_file: str = "chat_history.txt"
    user_data_file: str = "user_data.json"
    default_language: Language = Language.HINDI

class UserManager:
    """Manages user information storage and retrieval"""
    def __init__(self, user_data_file: str):
        self.user_data_file = user_data_file
        self.ensure_file_exists()
    
    def ensure_file_exists(self):
        """Create user data file if it doesn't exist"""
        if not os.path.exists(self.user_data_file):
            with open(self.user_data_file, 'w', encoding='utf-8') as f:
                json.dump({}, f)
    
    def save_user_info(self, user_info: UserInfo):
        """Save user information to JSON file"""
        try:
            with open(self.user_data_file, 'r+', encoding='utf-8') as f:
                data = json.load(f)
                data[user_info.mobile] = {
                    "name": user_info.name,
                    "location": user_info.location,
                    "has_purchased": user_info.has_purchased,
                    "crop_type": user_info.crop_type,
                    "last_updated": datetime.now().isoformat()
                }
                f.seek(0)
                json.dump(data, f, indent=4)
                f.truncate()
            return True
        except Exception as e:
            logging.error(f"Error saving user info: {str(e)}")
            return False
    
    def get_user_info(self, mobile: str) -> Optional[UserInfo]:
        """Retrieve user information from JSON file"""
        try:
            with open(self.user_data_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if mobile in data:
                    user_data = data[mobile]
                    return UserInfo(
                        name=user_data["name"],
                        mobile=mobile,
                        location=user_data["location"],
                        has_purchased=user_data["has_purchased"],
                        crop_type=user_data["crop_type"]
                    )
                return None
        except Exception as e:
            logging.error(f"Error retrieving user info: {str(e)}")
            return None

class ChatLogger:
    """Logger for chat interactions"""
    def __init__(self, log_file: str):
        self.log_file = log_file
        
    def log_interaction(self, question: str, answer: str, language: Language, user_info: Optional[UserInfo] = None):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.log_file, 'a', encoding='utf-8') as f:
            user_context = ""
            if user_info:
                user_context = f"\nUser: {user_info.name} | Location: {user_info.location} | Crop: {user_info.crop_type}"
            f.write(f"\n[{timestamp}] [{language.value}]{user_context}\nQ: {question}\nA: {answer}\n{'-'*50}")

class ChatMemory:
    """Manages chat history"""
    def __init__(self, max_history: int = 3):
        self.max_history = max_history
        self.history = []
        
    def add_interaction(self, question: str, answer: str):
        self.history.append({"question": question, "answer": answer})
        if len(self.history) > self.max_history:
            self.history.pop(0)
            
    def get_history(self) -> List[Dict[str, str]]:
        return self.history
    
    def clear_history(self):
        self.history = []

class Translator:
    """Handles translation between English and Hindi using Gemini"""
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.generation_config = {
            "temperature": 0.1,  # Low temperature for more accurate translations
            "top_p": 0.95,
            "top_k": 64,
            "max_output_tokens": 8192,
        }
        self.model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            generation_config=self.generation_config
        )
        
    async def translate_to_hindi(self, text: str) -> str:
        """Translates English text to Hindi"""
        try:
            chat = self.model.start_chat(history=[])
            
            # First, set translation context
            system_prompt = """You are a professional translator specializing in English to Hindi translation.
            Follow these rules strictly:
            1. Translate the given text to fluent, natural Hindi
            2. Use Devanagari script only
            3. Keep technical terms in English but add Hindi translation in parentheses
            4. Maintain the original formatting and structure
            5. Ensure the translation is culturally appropriate for Indian farmers"""
            
            chat.send_message(system_prompt)
            
            # Send the text for translation
            translation_prompt = f"""Translate the following English text to Hindi:

            {text}
            
            Rules:
            - Translate everything to Hindi using Devanagari script
            - Keep agricultural/technical terms in both English and Hindi
            - Maintain any formatting, lists, or structure from the original
            - Make it sound natural and fluent in Hindi"""
            
            response = chat.send_message(translation_prompt)
            translated_text = response.text
            
            # Verify the translation contains Devanagari
            if not any(ord(c) >= 0x900 and ord(c) <= 0x97F for c in translated_text):
                # If no Devanagari found, try one more time with stronger prompt
                retry_prompt = """पिछला अनुवाद देवनागरी में नहीं था। कृपया इस टेक्स्ट का अनुवाद पूरी तरह से हिंदी में करें:

                {text}
                
                अनुवाद बिल्कुल हिंदी में होना चाहिए और देवनागरी लिपि में होना चाहिए।"""
                
                response = chat.send_message(retry_prompt)
                translated_text = response.text
            
            return translated_text
            
        except Exception as e:
            logging.error(f"Translation error: {str(e)}")
            return text  # Return original text if translation fails
    
    def is_hindi(self, text: str) -> bool:
        """Checks if text contains Hindi characters"""
        return any(ord(c) >= 0x900 and ord(c) <= 0x97F for c in text)
        
class QuestionGenerator:
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
        
    async def generate_questions(
        self, 
        question: str, 
        answer: str, 
        language: Language, 
        user_info: Optional[UserInfo] = None
    ) -> List[str]:
        try:
            chat = self.model.start_chat(history=[])
            
            # First, set the language context
            if language == Language.HINDI:
                system_prompt = """You are now in Hindi-only mode. 
                इस चैट में आप केवल हिंदी में प्रश्न तैयार करेंगे।
                सभी प्रश्न देवनागरी लिपि में होने चाहिए।
                अंग्रेजी का प्रयोग बिल्कुल नहीं करना है।"""
            else:
                system_prompt = "You are now in English-only mode. Generate all questions in English only."
                
            chat.send_message(system_prompt)
            
            # Build the main prompt with strong language enforcement
            language_instruction = (
                """आपको सभी प्रश्न केवल हिंदी में बनाने हैं। प्रत्येक प्रश्न देवनागरी लिपि में होना चाहिए।
                किसी भी स्थिति में अंग्रेजी का प्रयोग न करें।"""
                if language == Language.HINDI
                else "Generate all questions in English only."
            )
            
            user_context = ""
            if user_info:
                user_context = (
                    f"""
                    निम्नलिखित जानकारी के आधार पर प्रश्न तैयार करें:
                    - किसान: {user_info.name}
                    - स्थान: {user_info.location}
                    - उत्पाद खरीदा: {'हां' if user_info.has_purchased else 'नहीं'}
                    - फसल: {user_info.crop_type}
                    """
                    if language == Language.HINDI
                    else f"""
                    Consider this context while generating questions:
                    - Farmer: {user_info.name}
                    - Location: {user_info.location}
                    - Has purchased: {'Yes' if user_info.has_purchased else 'No'}
                    - Crop: {user_info.crop_type}
                    """
                )
            
            base_prompt = (
                f"""इस बातचीत के आधार पर प्रश्न तैयार करें:

                पिछला प्रश्न: {question}
                दिया गया उत्तर: {answer}

                {language_instruction}
        
                {user_context}

                कृपया 4 संबंधित प्रश्न तैयार करें जो एक किसान पूछ सकता है।
                प्रत्येक प्रश्न नई लाइन में लिखें और क्रमांक दें (1-4)।"""
                if language == Language.HINDI
                else f"""Based on this conversation:

                Previous question: {question}
                Given answer: {answer}

                {language_instruction}
               
                {user_context}

                Generate 4 relevant follow-up questions a farmer might ask.
                Number each question (1-4) and put each on a new line."""
            )
            
            response = chat.send_message(base_prompt).text
            
            questions = []
            for line in response.split('\n'):
                line = line.strip()
                if line and (line.startswith('1.') or line.startswith('2.') or 
                           line.startswith('3.') or line.startswith('4.')):
                    questions.append(line.split('.', 1)[1].strip())
            
            # Verify language and retry if needed for Hindi
            if language == Language.HINDI and not all(any(ord(c) >= 0x900 and ord(c) <= 0x97F for c in q) for q in questions):
                force_hindi = """आपके प्रश्न हिंदी में नहीं थे। कृपया सभी प्रश्न पूरी तरह से हिंदी में दोबारा तैयार करें।"""
                response = chat.send_message(force_hindi).text
                questions = []
                for line in response.split('\n'):
                    line = line.strip()
                    if line and (line.startswith('1.') or line.startswith('2.') or 
                               line.startswith('3.') or line.startswith('4.')):
                        questions.append(line.split('.', 1)[1].strip())
            
            default_questions = {
                Language.ENGLISH: [
                    "Can you provide more details about the product?",
                    "What are the application methods?",
                    "What results can I expect to see?",
                    "Is it safe for all soil types?"
                ],
                Language.HINDI: [
                    "क्या आप उत्पाद के बारे में और जानकारी दे सकते हैं?",
                    "इसे कैसे प्रयोग करें?",
                    "मुझे क्या परिणाम देखने को मिलेंगे?",
                    "क्या यह सभी प्रकार की मिट्टी के लिए सुरक्षित है?"
                ]
            }
            
            while len(questions) < 4:
                questions.append(default_questions[language][len(questions)])
            
            return questions[:4]
            
        except Exception as e:
            logging.error(f"Error generating questions: {str(e)}")
            return default_questions[language]

class ImageProcessor:
    """Handles image processing and analysis using Gemini"""
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.generation_config = {
            "temperature": 0.1,
            "top_p": 0.95,
            "top_k": 64,
            "max_output_tokens": 8192,
        }
        self.model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            generation_config=self.generation_config
        )
    
    async def process_image_query(
        self,
        image: bytes,
        query: str,
        language: Language,
        user_info: Optional[UserInfo] = None
    ) -> str:
        try:
            chat = self.model.start_chat(history=[])
            
            language_instruction = (
                "Respond in fluent Hindi, using Devanagari script." if language == Language.HINDI
                else "Respond in English."
            )
            
            user_context = ""
            if user_info:
                user_context = f"""
                Consider this user context while analyzing the image:
                - You are helping {user_info.name} from {user_info.location}
                - They {'' if user_info.has_purchased else 'have not '}purchased product
                - They are growing {user_info.crop_type}
                """
            
            # Upload image to Gemini
            image_part = {"mime_type": "image/jpeg", "data": image}
            
            prompt = f"""You are an expert agricultural consultant specializing in crop health and product bio-fertilizer.
            
            {language_instruction}
            
            {user_context}
            
            Analyze the image and address the user's query: {query}
            
            Focus on:
            1. Identifying any visible issues or concerns
            2. Providing practical solutions and recommendations
            3. Explaining how product might help (if relevant)
            4. Suggesting preventive measures for the future
            
            Be specific and actionable in your response."""
            
            response = chat.send_message([image_part, prompt])
            return response.text
            
        except Exception as e:
            logging.error(f"Error processing image query: {str(e)}")
            default_error = {
                Language.ENGLISH: "I apologize, but I'm having trouble processing the image. Please try again.",
                Language.HINDI: "क्षमा करें, मैं छवि को प्रोसेस करने में असमर्थ हूं। कृपया पुनः प्रयास करें।"
            }
            return default_error[language]


class GeminiRAG:
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.generation_config = {
            "temperature": 0.1,
            "top_p": 0.95,
            "top_k": 64,
            "max_output_tokens": 8192,
        }
        self.model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            generation_config=self.generation_config
        )
        self.image_processor = ImageProcessor(api_key)
        self.translator = Translator(api_key)
        
    async def get_answer(
        self, 
        question: str, 
        context: str, 
        language: Language,
        user_info: Optional[UserInfo] = None,
        image: Optional[bytes] = None
    ) -> str:
        if image:
            response = await self.image_processor.process_image_query(
                image,
                question,
                Language.ENGLISH,  # Always get English response first
                user_info
            )
            if language == Language.HINDI:
                return await self.translator.translate_to_hindi(response)
            return response
        
        try:
            # Get response in English first
            context_prompt = f"""
            Context: {context}
            
            Farmer Info:
            Name: {user_info.name if user_info else 'Unknown'}
            Location: {user_info.location if user_info else 'Unknown'}
            Crop: {user_info.crop_type if user_info else 'Unknown'}
            
            Farmer's Question: {question}
            """
            
            chat = self.model.start_chat(history=[])
            response = chat.send_message(context_prompt).text
            
            # Translate to Hindi if needed
            if language == Language.HINDI:
                return await self.translator.translate_to_hindi(response)
            
            return response
            
        except Exception as e:
            logging.error(f"Error generating answer: {str(e)}")
            return ("क्षमा करें, तकनीकी समस्या। कृपया पुनः प्रयास करें।" 
                    if language == Language.HINDI 
                    else "Sorry, technical error. Please try again.")

class CustomEmbeddings(Embeddings):
    """Custom embeddings using SentenceTransformer"""
    def __init__(self, model_name: str, device: str):
        self.model = SentenceTransformer(model_name)
        self.model.to(device)
        
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        with torch.no_grad():
            embeddings = self.model.encode(texts, convert_to_tensor=True)
            return embeddings.cpu().numpy().tolist()
            
    def embed_query(self, text: str) -> List[float]:
        with torch.no_grad():
            embedding = self.model.encode([text], convert_to_tensor=True)
            return embedding.cpu().numpy().tolist()[0]

class ProductDatabase:
    """Handles document storage and retrieval"""
    def __init__(self, config: ChatConfig):
        self.embeddings = CustomEmbeddings(
            model_name=config.embedding_model_name,
            device=config.device
        )
        self.vectorstore = None
        
    def process_markdown(self, markdown_content: str):
        """Process markdown content and create vector store"""
        try:
            sections = markdown_content.split('\n## ')
            documents = []
            
            if sections[0].startswith('# '):
                intro = sections[0].split('\n', 1)[1]
                documents.append({
                    "content": intro,
                    "section": "Introduction"
                })
            
            for section in sections[1:]:
                if section.strip():
                    title, content = section.split('\n', 1)
                    documents.append({
                        "content": content.strip(),
                        "section": title.strip()
                    })
            
            texts = [doc["content"] for doc in documents]
            metadatas = [{"section": doc["section"]} for doc in documents]
            
            self.vectorstore = FAISS.from_texts(
                texts=texts,
                embedding=self.embeddings,
                metadatas=metadatas
            )
            
        except Exception as e:
            raise Exception(f"Error processing markdown content: {str(e)}")
        
    def search(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """Search for relevant documents"""
        if not self.vectorstore:
            raise ValueError("Database not initialized. Please process documents first.")
        
        try:
            docs = self.vectorstore.similarity_search(query, k=k)
            return [{"content": doc.page_content, "metadata": doc.metadata} for doc in docs]
        except Exception as e:
            logging.error(f"Error during search: {str(e)}")
            return []
