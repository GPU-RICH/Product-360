import os
import logging
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
import torch
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
import google.generativeai as genai
from datetime import datetime
import json
from PIL import Image
import io

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
        
    def log_interaction(self, question: str, answer: str, user_info: Optional[UserInfo] = None):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.log_file, 'a', encoding='utf-8') as f:
            user_context = ""
            if user_info:
                user_context = f"\nUser: {user_info.name} | Location: {user_info.location} | Crop: {user_info.crop_type}"
            f.write(f"\n[{timestamp}]{user_context}\nQ: {question}\nA: {answer}\n{'-'*50}")

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

class QuestionGenerator:
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.generation_config = {
            "temperature": 0.1,
            "top_p": 0.95,
            "max_output_tokens": 8192,
        }
        self.model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            generation_config=self.generation_config
        )
        
        self.default_questions = [
            "इस उत्पाद को कब इस्तेमाल करना सबसे अच्छा रहेगा?",
            "क्या उत्पाद के इस्तेमाल के बाद आपने कोई परिणाम देखा है?",
            "क्या आपको अपनी फसल पर इसे लगाने को लेकर कोई चिंता है?",
            "क्या आप कीट नियंत्रण के अन्य तरीकों के बारे में जानना चाहेंगे?"
        ]
    
    async def generate_questions(
        self, 
        question: str, 
        answer: str, 
        user_info: Optional[UserInfo] = None
    ) -> List[str]:
        """Generate follow-up questions based on the conversation"""
        try:
            chat = self.model.start_chat(history=[])
            prompt = f"""Generate 4 simple, practical follow-up questions in Hindi (Devanagari script) based on this conversation with a farmer:

Question: {question}
Answer: {answer}

Focus the questions on:
1. समय और उपयोग (Timing and usage)
2. परिणाम और प्रभावशीलता (Results and effectiveness)
3. सुरक्षा संबंधी चिंताएं (Safety concerns)
4. अतिरिक्त सिफारिशें (Additional recommendations)

Keep the language simple and farmer-friendly. Format each question on a new line."""

            response = chat.send_message(prompt).text
            
            # Extract questions
            questions = [q.strip() for q in response.split('\n') if q.strip()]
            
            # Return default questions if we don't get exactly 4 valid questions
            if len(questions) != 4:
                return self.default_questions
            
            return questions
            
        except Exception as e:
            logging.error(f"Error generating questions: {str(e)}")
            return self.default_questions

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
        user_info: Optional[UserInfo] = None
    ) -> str:
        try:
            chat = self.model.start_chat(history=[])
            
            user_context = ""
            if user_info:
                user_context = f"""
                किसान की जानकारी:
                - नाम: {user_info.name}
                - स्थान: {user_info.location}
                - ग्राहक प्रतिक्रिया: {user_info.crop_type}
                - उत्पाद खरीदा: {'हाँ' if user_info.has_purchased else 'नहीं'}
                """
            
            image_part = {"mime_type": "image/jpeg", "data": image}
            
            prompt = f"""You are an expert agricultural consultant. Analyze the image and provide response in Hindi (Devanagari script).
            
            {user_context}
            
            किसान का प्रश्न: {query}
            
            इन बिंदुओं पर ध्यान दें:
            1. दिखाई देने वाली समस्याओं की पहचान
            2. व्यावहारिक समाधान
            3. उत्पाद कैसे मदद कर सकता है
            4. रोकथाम के उपाय
            5. तत्काल करने योग्य कदम
            
            सरल भाषा में एक संक्षिप्त और स्पष्ट जवाब दें।"""
            
            response = chat.send_message([image_part, prompt]).text
            return response
            
        except Exception as e:
            logging.error(f"Error processing image query: {str(e)}")
            return "क्षमा करें, छवि का विश्लेषण करने में समस्या हुई। कृपया पुनः प्रयास करें।"

    async def validate_image(self, image: bytes) -> bool:
        """Validate if the image is suitable for analysis"""
        try:
            img = Image.open(io.BytesIO(image))
            width, height = img.size
            if width < 100 or height < 100:
                return False
            if width > 4096 or height > 4096:
                return False
            return True
        except Exception as e:
            logging.error(f"Image validation error: {str(e)}")
            return False

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
    
    def create_context(self, relevant_docs: List[Dict[str, Any]]) -> str:
        """Creates a context string from relevant documents"""
        context_parts = []
        for doc in relevant_docs:
            context_parts.append(f"Section: {doc['metadata']['section']}\n{doc['content']}")
        return "\n\n".join(context_parts)

    async def get_answer(
        self, 
        question: str, 
        context: str,
        user_info: Optional[UserInfo] = None,
        image: Optional[bytes] = None
    ) -> str:
        if image:
            return await self.image_processor.process_image_query(image, question, user_info)
        
        try:
            chat = self.model.start_chat(history=[])
            
            prompt = f"""You are an agricultural expert. Provide response in Hindi (Devanagari script).

            संदर्भ जानकारी:
            {context}
            
            {f'''किसान की जानकारी:
            - नाम: {user_info.name}
            - स्थान: {user_info.location}
            - ग्राहक प्रतिक्रिया: {user_info.crop_type}
            - उत्पाद खरीदा: {'हाँ' if user_info.has_purchased else 'नहीं'}''' if user_info else ''}
            
            प्रश्न: {question}
            
            निर्देश:
            1. विशिष्ट और व्यावहारिक सलाह दें
            2. सरल किसान-हितैषी भाषा का प्रयोग करें
            3. जहाँ उपयुक्त हो उदाहरण दें
            4. तकनीकी शब्दों को समझाएं
            5. समाधान और सर्वोत्तम प्रथाओं पर ध्यान दें
            
            एक संक्षिप्त और व्यावहारिक उत्तर दें।"""
            
            response = chat.send_message(prompt).text
            return response
            
        except Exception as e:
            logging.error(f"Error generating answer: {str(e)}")
            return "क्षमा करें, तकनीकी त्रुटि हुई। कृपया पुनः प्रयास करें।"

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
                texts=texts,  # Added missing comma here
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
