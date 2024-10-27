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

class SimpleTranslator:
    """Simple translator that converts English responses to Hindi"""
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            generation_config={
                "temperature": 0.1,
                "top_p": 0.95,
                "max_output_tokens": 8192,
            }
        )
    
    async def to_hindi(self, english_text: str) -> str:
        """Translates English text to Hindi"""
        try:
            chat = self.model.start_chat(history=[])
            prompt = f"""Translate the following English text to Hindi. Use Devanagari script.
            Maintain the same formatting and structure.
            Make it natural and farmer-friendly.
            
            Text to translate:
            {english_text}"""
            
            response = chat.send_message(prompt)
            return response.text
        except Exception as e:
            logging.error(f"Translation error: {str(e)}")
            return "अनुवाद में त्रुटि। कृपया पुनः प्रयास करें।"
        
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
        
        # Simple, practical default questions for both languages
        self.default_questions = {
            Language.ENGLISH: [
                "What is the best time to apply this product?",
                "Have you seen any results after using the product?",
                "Do you have any concerns about applying it to your crops?",
                "Would you like to learn about other pest control methods?"
            ],
            Language.HINDI: [
                "इस उत्पाद को कब इस्तेमाल करना सबसे अच्छा रहेगा?",
                "क्या उत्पाद के इस्तेमाल के बाद आपने कोई परिणाम देखा है?",
                "क्या आपको अपनी फसल पर इसे लगाने को लेकर कोई चिंता है?",
                "क्या आप कीट नियंत्रण के अन्य तरीकों के बारे में जानना चाहेंगे?"
            ]
        }
    
    async def generate_questions(
        self, 
        question: str, 
        answer: str, 
        language: Language, 
        user_info: Optional[UserInfo] = None
    ) -> List[str]:
        """Generate follow-up questions based on the conversation"""
        try:
            # Generate in English first with a simple prompt
            chat = self.model.start_chat(history=[])
            prompt = f"""Based on this conversation with a farmer:

Question: {question}
Answer: {answer}

Generate 4 simple, practical follow-up questions focusing on:
1. Timing of application
2. Results and effectiveness
3. Safety concerns
4. Additional recommendations

Format each question on a new line starting with a number and a dot."""

            response = chat.send_message(prompt).text
            
            # Extract questions - keep it simple
            questions = []
            for line in response.split('\n'):
                line = line.strip()
                if line and any(line.startswith(f"{i}. ") for i in range(1, 5)):
                    question = line.split('. ', 1)[1].strip()
                    if question:
                        questions.append(question)
            
            # If we can't get enough good questions, use the defaults
            if len(questions) != 4:
                return self.default_questions[language]
                
            # If Hindi is requested, translate the questions
            if language == Language.HINDI:
                try:
                    hindi_prompt = f"""Translate these farming-related questions to simple, natural Hindi:

{questions[0]}
{questions[1]}
{questions[2]}
{questions[3]}"""
                    
                    hindi_response = chat.send_message(hindi_prompt).text
                    hindi_questions = [q.strip() for q in hindi_response.split('\n') if q.strip()]
                    
                    if len(hindi_questions) == 4:
                        return hindi_questions
                    else:
                        return self.default_questions[Language.HINDI]
                except:
                    return self.default_questions[Language.HINDI]
            
            return questions
            
        except Exception as e:
            logging.error(f"Error generating questions: {str(e)}")
            return self.default_questions[language]


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
        self.translator = SimpleTranslator(api_key)
    
    async def process_image_query(
        self,
        image: bytes,
        query: str,
        language: Language,
        user_info: Optional[UserInfo] = None
    ) -> str:
        try:
            chat = self.model.start_chat(history=[])
            
            # Always create prompt in English
            user_context = ""
            if user_info:
                user_context = f"""
                Consider this context while analyzing the image:
                - Farmer Name: {user_info.name}
                - Location: {user_info.location}
                - Current Crop: {user_info.crop_type}
                - Product Purchased: {'Yes' if user_info.has_purchased else 'No'}
                """
            
            # Upload image to Gemini
            image_part = {"mime_type": "image/jpeg", "data": image}
            
            prompt = f"""You are an expert agricultural consultant specializing in crop health.
            
            {user_context}
            
            Analyze the image and address this query: {query}
            
            Focus on:
            1. Identifying specific visible issues or concerns
            2. Providing practical, actionable solutions
            3. Explaining how the product can help (if relevant)
            4. Suggesting preventive measures
            5. Mentioning any immediate steps needed
            
            Guidelines:
            - Be specific and actionable
            - Give concise, clear recommendations
            - Use farmer-friendly language
            - Prioritize practical advice
            - Maintain a helpful, conversational tone
            
            Provide a brief, structured and easy-to-follow response. Always maintain a conversational tone."""
            
            # Get response in English
            response = chat.send_message([image_part, prompt]).text
            
            # Translate to Hindi if needed
            if language == Language.HINDI:
                try:
                    return await self.translator.to_hindi(response)
                except Exception as e:
                    logging.error(f"Translation error: {str(e)}")
                    return "क्षमा करें, छवि का विश्लेषण करने में समस्या हुई। कृपया पुनः प्रयास करें।"
            
            return response
            
        except Exception as e:
            logging.error(f"Error processing image query: {str(e)}")
            error_msg = "I apologize, but I'm having trouble analyzing the image. Please try again."
            
            if language == Language.HINDI:
                try:
                    return await self.translator.to_hindi(error_msg)
                except:
                    return "क्षमा करें, छवि का विश्लेषण करने में समस्या हुई। कृपया पुनः प्रयास करें।"
            
            return error_msg

    async def validate_image(self, image: bytes) -> bool:
        """Validate if the image is suitable for analysis"""
        try:
            img = Image.open(io.BytesIO(image))
            # Check if image is not too small or too large
            width, height = img.size
            if width < 100 or height < 100:
                return False
            if width > 4096 or height > 4096:
                return False
            # Add more validation as needed
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
        self.translator = SimpleTranslator(api_key)
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
        language: Language,
        user_info: Optional[UserInfo] = None,
        image: Optional[bytes] = None
    ) -> str:
        if image:
            try:
                # Process image query in English first
                chat = self.model.start_chat(history=[])
                
                image_part = {"mime_type": "image/jpeg", "data": image}
                prompt = f"""You are an expert agricultural consultant specializing in crop health.
                
                Analyze the image and address this query: {question}
                
                Consider this context:
                {context}
                
                {f'''User Context:
                - Farmer: {user_info.name}
                - Location: {user_info.location}
                - Growing: {user_info.crop_type}
                - Has purchased product: {'Yes' if user_info.has_purchased else 'No'}''' if user_info else ''}
                
                Focus on:
                1. Identifying visible issues or concerns
                2. Providing practical solutions and recommendations
                3. Explaining how product might help (if relevant)
                4. Suggesting preventive measures
                
                Provide a very brief, actionable response. Always try to keep the flow of a conversation."""
                
                response = chat.send_message([image_part, prompt]).text
                
                # Translate to Hindi if needed
                if language == Language.HINDI:
                    return await self.translator.to_hindi(response)
                return response
                
            except Exception as e:
                logging.error(f"Error processing image: {str(e)}")
                error_msg = "Sorry, I couldn't process the image. Please try again."
                if language == Language.HINDI:
                    return await self.translator.to_hindi(error_msg)
                return error_msg
        
        try:
            # Get response in English first
            chat = self.model.start_chat(history=[])
            
            prompt = f"""You are an agricultural expert specializing in crop protection.

            Context Information:
            {context}
            
            {f'''User Context:
            - Farmer: {user_info.name}
            - Location: {user_info.location}
            - Growing: {user_info.crop_type}
            - Has purchased product: {'Yes' if user_info.has_purchased else 'No'}''' if user_info else ''}
            
            Question: {question}
            
            Instructions:
            1. Provide specific, actionable advice
            2. Use clear, farmer-friendly language
            3. Include practical examples where relevant
            4. Explain any technical terms
            5. Focus on solutions and best practices
            
            Provide a very brief, actionable response addressing the farmer's question. Always try to keep the flow of a conversation."""
            
            response = chat.send_message(prompt).text
            
            # Translate to Hindi if needed
            if language == Language.HINDI:
                return await self.translator.to_hindi(response)
            
            return response
            
        except Exception as e:
            logging.error(f"Error generating answer: {str(e)}")
            error_msg = "Sorry, I encountered a technical error. Please try again."
            if language == Language.HINDI:
                return await self.translator.to_hindi(error_msg)
            return error_msg

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
