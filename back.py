import os
import logging
import asyncio
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
import torch
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
import google.generativeai as genai
from datetime import datetime
import json
import PIL.Image
from PIL import Image
import io
import base64
from aiohttp import ClientSession
from functools import partial

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
    gemini_api_key: str = "AIzaSyBS_DFCJh82voYIKoglS-ow6ezGNg775pg"
    log_file: str = "chat_history.txt"
    user_data_file: str = "user_data.json"

class UserManager:
    """Manages user information storage and retrieval"""
    def __init__(self, user_data_file: str):
        self.user_data_file = user_data_file
        self.ensure_file_exists()
        self._lock = asyncio.Lock()
    
    def ensure_file_exists(self):
        """Create user data file if it doesn't exist"""
        if not os.path.exists(self.user_data_file):
            with open(self.user_data_file, 'w', encoding='utf-8') as f:
                json.dump({}, f)
    
    async def save_user_info(self, user_info: UserInfo) -> bool:
        """Save user information to JSON file asynchronously"""
        try:
            async with self._lock:
                async def write_to_file():
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
                
                # Run file operations in thread pool
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, write_to_file)
                return True
        except Exception as e:
            logging.error(f"Error saving user info: {str(e)}")
            return False
    
    async def get_user_info(self, mobile: str) -> Optional[UserInfo]:
        """Retrieve user information from JSON file asynchronously"""
        try:
            async with self._lock:
                def read_file():
                    with open(self.user_data_file, 'r', encoding='utf-8') as f:
                        return json.load(f)
                
                loop = asyncio.get_event_loop()
                data = await loop.run_in_executor(None, read_file)
                
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
        self._lock = asyncio.Lock()
        
    async def log_interaction(self, question: str, answer: str, user_info: Optional[UserInfo] = None):
        """Log chat interactions asynchronously"""
        async with self._lock:
            def write_log():
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                with open(self.log_file, 'a', encoding='utf-8') as f:
                    user_context = ""
                    if user_info:
                        user_context = f"\nUser: {user_info.name} | Location: {user_info.location} | Crop: {user_info.crop_type}"
                    f.write(f"\n[{timestamp}]{user_context}\nQ: {question}\nA: {answer}\n{'-'*50}")
            
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, write_log)

class ChatMemory:
    """Manages chat history"""
    def __init__(self, max_history: int = 3):
        self.max_history = max_history
        self.history = []
        self._lock = asyncio.Lock()
        
    async def add_interaction(self, question: str, answer: str):
        """Add interaction to history asynchronously"""
        async with self._lock:
            self.history.append({"question": question, "answer": answer})
            if len(self.history) > self.max_history:
                self.history.pop(0)
            
    def get_history(self) -> List[Dict[str, str]]:
        return self.history.copy()
    
    async def clear_history(self):
        """Clear chat history asynchronously"""
        async with self._lock:
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
        self._lock = asyncio.Lock()
        
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
        """Generate follow-up questions asynchronously"""
        async with self._lock:
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

                # Run Gemini operation in thread pool
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(
                    None,
                    partial(chat.send_message, prompt)
                )
                
                # Extract questions
                questions = [q.strip() for q in response.text.split('\n') if q.strip()]
                
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
        self._lock = asyncio.Lock()
    
    async def validate_image(self, image: bytes) -> bool:
        """Validate if the image is suitable for analysis asynchronously"""
        try:
            def process_image():
                img = Image.open(io.BytesIO(image))
                
                # Check image format
                if img.format not in ['JPEG', 'PNG']:
                    logging.warning(f"Invalid image format: {img.format}")
                    return False
                
                # Check dimensions
                width, height = img.size
                if width < 100 or height < 100:
                    logging.warning(f"Image too small: {width}x{height}")
                    return False
                if width > 4096 or height > 4096:
                    logging.warning(f"Image too large: {width}x{height}")
                    return False
                
                # Ensure the image can be converted to RGB
                if img.mode not in ['RGB', 'RGBA']:
                    try:
                        img = img.convert('RGB')
                    except Exception as e:
                        logging.error(f"Failed to convert image to RGB: {str(e)}")
                        return False
                
                # Check file size (max 4MB)
                img_byte_arr = io.BytesIO()
                img.save(img_byte_arr, format='JPEG' if img.format == 'JPEG' else 'PNG')
                if len(img_byte_arr.getvalue()) > 4 * 1024 * 1024:
                    logging.warning("Image file size too large")
                    return False
                
                return True
            
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, process_image)
                
        except Exception as e:
            logging.error(f"Image validation error: {str(e)}")
            return False

    async def process_image_query(
        self,
        image: bytes,
        query: str,
        user_info: Optional[UserInfo] = None,
        context: str = ""
    ) -> str:
        """Process image-based queries asynchronously"""
        async with self._lock:
            try:
                if not image:
                    return "कोई छवि नहीं मिली। कृपया एक छवि अपलोड करें।"

                # Validate image first
                is_valid = await self.validate_image(image)
                if not is_valid:
                    return "छवि का आकार या प्रारूप उपयुक्त नहीं है। कृपया 100x100 से 4096x4096 के बीच का आकार वाली JPG/PNG छवि अपलोड करें।"
                
                def process_image():
                    img = Image.open(io.BytesIO(image))
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    return img
                
                loop = asyncio.get_event_loop()
                img = await loop.run_in_executor(None, process_image)
                
                user_context = ""
                if user_info:
                    user_context = f"""
                    किसान की जानकारी:
                    - नाम: {user_info.name}
                    - स्थान: {user_info.location}
                    - फसल: {user_info.crop_type}
                    - उत्पाद खरीदा: {'हाँ' if user_info.has_purchased else 'नहीं'}
                    """
                
                prompt = f"""कृपया इस छवि का विश्लेषण करें और किसान की मदद करें।
                
                {user_context}
                
                संदर्भ जानकारी:
                {context}
                
                किसान का प्रश्न: {query}
                
                कृपया इन बिंदुओं पर ध्यान दें:
                1. छवि में दिखाई दे रही समस्या का विस्तृत विवरण
                2. संभावित कारण
                3. तत्काल समाधान
                4. भविष्य में बचाव के उपाय
                5. क्या उत्पाद इसमें मदद कर सकता है
                
                कृपया सरल हिंदी में जवाब दें जो एक किसान आसानी से समझ सके।"""

                # Convert image to bytes for API
                def prepare_image():
                    img_byte_arr = io.BytesIO()
                    img.save(img_byte_arr, format='JPEG')
                    return img_byte_arr.getvalue()
                
                img_bytes = await loop.run_in_executor(None, prepare_image)

                # Prepare content for the model
                contents = [
                    {
                        "parts": [
                            {"text": prompt},
                            {
                                "inline_data": {
                                    "mime_type": "image/jpeg",
                                    "data": base64.b64encode(img_bytes).decode('utf-8')
                                }
                            }
                        ]
                    }
                ]

                # Generate response
                response = await loop.run_in_executor(
                    None,
                    partial(self.model.generate_content, contents)
                )
                
                if response and response.text:
                    return response.text
                else:
                    raise ValueError("No response received from Gemini")
                
            except genai.types.generation_types.BlockedPromptException:
                logging.error("Blocked prompt exception")
                return "छवि में कुछ अनुपयुक्त सामग्री पाई गई। कृपया केवल फसल या खेती संबंधित छवियां अपलोड करें।"
            except Exception as e:
                logging.error(f"Error processing image query: {str(e)}", exc_info=True)
                return f"छवि का विश्लेषण करने में समस्या हुई। कृपया दूसरी छवि के साथ पुनः प्रयास करें। त्रुटि: {str(e)}"

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
        self._lock = asyncio.Lock()
    
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
        """Get answer asynchronously using either text or image processing"""
        async with self._lock:
            try:
                # If image is provided, use image processor
                if image:
                    return await self.image_processor.process_image_query(
                        image=image,
                        query=question,
                        user_info=user_info,
                        context=context
                    )
                
                # Text-only query processing
                def process_query():
                    chat = self.model.start_chat(history=[])
                    prompt = f"""You are an agricultural expert. Provide response in Hindi (Devanagari script).

                    संदर्भ जानकारी:
                    {context}
                    
                    {f'''किसान की जानकारी:
                    - नाम: {user_info.name}
                    - स्थान: {user_info.location}
                    - फसल: {user_info.crop_type}
                    - उत्पाद खरीदा: {'हाँ' if user_info.has_purchased else 'नहीं'}''' if user_info else ''}
                    
                    प्रश्न: {question}
                    
                    निर्देश:
                    1. विशिष्ट और व्यावहारिक सलाह दें
                    2. सरल किसान-हितैषी भाषा का प्रयोग करें
                    3. जहाँ उपयुक्त हो उदाहरण दें
                    4. तकनीकी शब्दों को समझाएं
                    5. समाधान और सर्वोत्तम प्रथाओं पर ध्यान दें"""
                    
                    response = chat.send_message(prompt)
                    return response.text
                
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(None, process_query)
                
            except Exception as e:
                logging.error(f"Error in get_answer: {str(e)}", exc_info=True)
                return "क्षमा करें, तकनीकी त्रुटि हुई। कृपया पुनः प्रयास करें।"

class CustomEmbeddings(Embeddings):
    """Custom embeddings using SentenceTransformer"""
    def __init__(self, model_name: str, device: str):
        self.model = SentenceTransformer(model_name)
        self.model.to(device)
        self._lock = asyncio.Lock()
        
    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed documents asynchronously"""
        async with self._lock:
            def process_embeddings():
                with torch.no_grad():
                    embeddings = self.model.encode(texts, convert_to_tensor=True)
                    return embeddings.cpu().numpy().tolist()
            
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, process_embeddings)
            
    async def embed_query(self, text: str) -> List[float]:
        """Embed query asynchronously"""
        async with self._lock:
            def process_embedding():
                with torch.no_grad():
                    embedding = self.model.encode([text], convert_to_tensor=True)
                    return embedding.cpu().numpy().tolist()[0]
            
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, process_embedding)

class ProductDatabase:
    """Handles document storage and retrieval"""
    def __init__(self, config: ChatConfig):
        self.embeddings = CustomEmbeddings(
            model_name=config.embedding_model_name,
            device=config.device
        )
        self.vectorstore = None
        self._lock = asyncio.Lock()
        
    async def process_markdown(self, markdown_content: str):
        """Process markdown content and create vector store asynchronously"""
        async with self._lock:
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
                
                # Process embeddings asynchronously
                embeddings = await self.embeddings.embed_documents(texts)
                
                def create_vectorstore():
                    return FAISS.from_texts(
                        texts=texts,
                        embedding=self.embeddings,
                        metadatas=metadatas
                    )
                
                loop = asyncio.get_event_loop()
                self.vectorstore = await loop.run_in_executor(None, create_vectorstore)
                
            except Exception as e:
                raise Exception(f"Error processing markdown content: {str(e)}")
        
    async def search(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """Search for relevant documents asynchronously"""
        if not self.vectorstore:
            raise ValueError("Database not initialized. Please process documents first.")
        
        async with self._lock:
            try:
                def perform_search():
                    docs = self.vectorstore.similarity_search(query, k=k)
                    return [{"content": doc.page_content, "metadata": doc.metadata} for doc in docs]
                
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(None, perform_search)
            except Exception as e:
                logging.error(f"Error during search: {str(e)}")
                return []
