#CORE.PY
import os
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import torch
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
import google.generativeai as genai
from datetime import datetime
from enum import Enum

class Language(Enum):
    HINDI = "hindi"
    ENGLISH = "english"

@dataclass
class CustomerInfo:
    """Customer information storage"""
    mobile: str
    location: str
    purchase_status: str
    crop_type: str
    name: Optional[str] = None


import os
import json
from pathlib import Path
from typing import Optional, Dict, Any
import tempfile

class CustomerDatabase:
    """Handles customer information storage in JSON"""
    def __init__(self, file_path: str = None):
        if file_path is None:
            # Use system temp directory if no path provided
            temp_dir = tempfile.gettempdir()
            self.file_path = Path(temp_dir) / "customer_data.json"
        else:
            self.file_path = Path(file_path)
            
        # Create parent directories if they don't exist
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_file_exists()
    
    def _ensure_file_exists(self):
        """Create the JSON file if it doesn't exist"""
        try:
            if not self.file_path.exists():
                self.file_path.write_text("{}", encoding="utf-8")
        except Exception as e:
            logging.error(f"Error creating file: {str(e)}")
            # Use memory-only fallback if file operations fail
            self._data = {}
            self._use_memory = True
    
    def save_customer(self, customer_info: CustomerInfo) -> None:
        """Save customer information to JSON file"""
        try:
            # Read existing data
            data = self._read_data()
            
            # Update with new customer info
            data[customer_info.mobile] = {
                "location": customer_info.location,
                "purchase_status": customer_info.purchase_status,
                "crop_type": customer_info.crop_type,
                "name": customer_info.name,
                "last_updated": datetime.now().isoformat()
            }
            
            # Write back to file
            try:
                with self.file_path.open("w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=4)
            except Exception as e:
                logging.error(f"Error writing to file: {str(e)}")
                self._data = data  # Store in memory if file write fails
                self._use_memory = True
                
        except Exception as e:
            logging.error(f"Error saving customer data: {str(e)}")
            raise
    
    def get_customer(self, mobile: str) -> Optional[CustomerInfo]:
        """Retrieve customer information by mobile number"""
        try:
            data = self._read_data()
            if mobile in data:
                customer_data = data[mobile]
                return CustomerInfo(
                    mobile=mobile,
                    location=customer_data["location"],
                    purchase_status=customer_data["purchase_status"],
                    crop_type=customer_data["crop_type"],
                    name=customer_data.get("name")
                )
            return None
        except Exception as e:
            logging.error(f"Error retrieving customer data: {str(e)}")
            return None
    
    def _read_data(self) -> Dict[str, Any]:
        """Read the JSON file or return in-memory data"""
        if hasattr(self, '_use_memory') and self._use_memory:
            return getattr(self, '_data', {})
            
        try:
            with self.file_path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logging.error(f"Error reading customer data: {str(e)}")
            return {}

    def customer_exists(self, mobile: str) -> bool:
        """Check if a customer exists in the database"""
        data = self._read_data()
        return mobile in data
        
@dataclass
class ChatConfig:
    """Configuration for the chatbot"""
    embedding_model_name: str = 'all-MiniLM-L6-v2'
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    max_history: int = 3
    gemini_api_key: str = "AIzaSyBS_DFCJh82voYIKoglS-ow6ezGNg775pg"
    log_file: str = "chat_history.txt"
    
    # Bilingual greeting messages
    greetings = {
        Language.HINDI: """
        नमस्ते! मैं GAPL Starter प्रोडक्ट असिस्टेंट हूं। 
        कृपया निम्नलिखित जानकारी साझा करें:
        """,
        Language.ENGLISH: """
        Hello! I'm the GAPL Starter Product Assistant.
        Please share the following information:
        """
    }
    
    # Required customer information prompts
    customer_info_prompts = {
        Language.HINDI: {
            "mobile": "आपका मोबाइल नंबर क्या है?",
            "location": "आप कहाँ से हैं?",
            "purchase_status": "क्या आपने GAPL Starter खरीदा है?",
            "crop_type": "आप किस फसल के लिए इसका उपयोग करना चाहते हैं?",
            "name": "आपका नाम क्या है? (वैकल्पिक)"
        },
        Language.ENGLISH: {
            "mobile": "What is your mobile number?",
            "location": "Where are you from?",
            "purchase_status": "Have you purchased GAPL Starter?",
            "crop_type": "Which crop do you want to use it for?",
            "name": "What is your name? (optional)"
        }
    }

class ChatLogger:
    """Logger for chat interactions"""
    def __init__(self, log_file: str):
        self.log_file = log_file
        
    def log_interaction(self, question: str, answer: str, customer_info: CustomerInfo):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(f"\n[{timestamp}]\nCustomer: {customer_info.mobile}\nQ: {question}\nA: {answer}\n{'-'*50}")

class ChatMemory:
    """Manages chat history"""
    def __init__(self, max_history: int = 3):
        self.max_history = max_history
        self.history = []
        self.customer_info: Optional[CustomerInfo] = None
        self.language = Language.HINDI  # Default language
        
    def set_customer_info(self, customer_info: CustomerInfo):
        self.customer_info = customer_info
        
    def set_language(self, language: Language):
        self.language = language
        
    def add_interaction(self, question: str, answer: str):
        self.history.append({"question": question, "answer": answer})
        if len(self.history) > self.max_history:
            self.history.pop(0)
            
    def get_history(self) -> List[Dict[str, str]]:
        return self.history
    
    def clear_history(self):
        self.history = []
        self.customer_info = None

class QuestionGenerator:
    """Generates follow-up questions using Gemini"""
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
        
    async def generate_questions(self, question: str, answer: str, language: Language) -> List[str]:
        try:
            chat = self.model.start_chat(history=[])
            
            lang_instruction = "in Hindi using Devanagari script" if language == Language.HINDI else "in English"
            
            prompt = f"""Based on this product information interaction:
            
            Question: {question}
            Answer: {answer}
            
            Generate 4 relevant follow-up questions {lang_instruction} that a customer might ask about GAPL Starter.
            Focus on:
            - Application methods and timing
            - Benefits and effectiveness
            - Compatibility with specific crops
            - Scientific backing and results
            
            Return ONLY the numbered questions (1-4), one per line.
            """
            
            response = chat.send_message(prompt).text
            
            questions = []
            for line in response.split('\n'):
                line = line.strip()
                if line and (line.startswith('1.') or line.startswith('2.') or 
                           line.startswith('3.') or line.startswith('4.')):
                    questions.append(line.split('.', 1)[1].strip())
            
            # Default fallback questions in selected language
            fallback_questions = {
                Language.HINDI: [
                    "GAPL Starter को कैसे स्टोर करें?",
                    "क्या इसे अन्य उर्वरकों के साथ मिश्रित किया जा सकता है?",
                    "इसके उपयोग से क्या परिणाम मिलेंगे?",
                    "क्या यह सभी प्रकार की मिट्टी के लिए सुरक्षित है?"
                ],
                Language.ENGLISH: [
                    "How should I store GAPL Starter?",
                    "Can I mix it with other fertilizers?",
                    "What results can I expect?",
                    "Is it safe for all soil types?"
                ]
            }
            
            while len(questions) < 4:
                questions.append(fallback_questions[language][len(questions)])
            
            return questions[:4]
            
        except Exception as e:
            logging.error(f"Error generating questions: {str(e)}")
            return fallback_questions[language]

class GeminiRAG:
    """RAG implementation using Gemini"""
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
        
    def create_context(self, relevant_docs: List[Dict[str, Any]]) -> str:
        return "\n\n".join(doc['content'] for doc in relevant_docs)
        
    async def get_answer(self, question: str, context: str, language: Language, customer_info: CustomerInfo) -> str:
        try:
            chat = self.model.start_chat(history=[])
            
            lang_instruction = "in Hindi using Devanagari script only" if language == Language.HINDI else "in English"
            
            prompt = f"""You are an expert agricultural consultant specializing in GAPL Starter bio-fertilizer. 
            You have extensive hands-on experience with the product and deep knowledge of its applications and benefits.
            
            IMPORTANT: Respond {lang_instruction} regardless of the input language.
            
            Customer Information:
            - Location: {customer_info.location}
            - Crop: {customer_info.crop_type}
            - Purchase Status: {customer_info.purchase_status}
            
            Background information to inform your response:
            {context}

            Question from farmer: {question}

            Your response should be:
            - In {lang_instruction}
            - Confident and authoritative
            - Direct and practical
            - Focused on helping farmers succeed
            - Based on product expertise
            - Personalized to the customer's crop and location when relevant

            If you don't have enough specific information to answer the question, respond {lang_instruction} with something like:
            "As a GAPL Starter expert, I should note that while the product has broad applications, 
            I'd need to check the specific details about [missing information] to give you the most accurate guidance. 
            What I can tell you is..."
            """
            
            response = chat.send_message(prompt)
            return response.text
            
        except Exception as e:
            logging.error(f"Error generating answer: {str(e)}")
            default_error = {
                Language.HINDI: "क्षमा करें, मैं आपके प्रश्न को प्रोसेस नहीं कर पा रहा हूं। कृपया पुनः प्रयास करें।",
                Language.ENGLISH: "I apologize, but I'm having trouble processing your request. Please try again."
            }
            return default_error[language]

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
            # Split the content into sections
            sections = markdown_content.split('\n## ')
            documents = []
            
            # Process the first section (intro)
            if sections[0].startswith('# '):
                intro = sections[0].split('\n', 1)[1]
                documents.append({
                    "content": intro,
                    "section": "Introduction"
                })
            
            # Process remaining sections
            for section in sections[1:]:
                if section.strip():
                    title, content = section.split('\n', 1)
                    documents.append({
                        "content": content.strip(),
                        "section": title.strip()
                    })
            
            # Create vector store
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
