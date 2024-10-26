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

@dataclass
class FarmerInfo:
    """Storage for farmer information"""
    mobile: str
    location: str
    crop_type: str
    purchase_status: str
    name: Optional[str] = None

@dataclass
class ChatConfig:
    """Configuration for the chatbot"""
    embedding_model_name: str = 'all-MiniLM-L6-v2'
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    max_history: int = 3
    gemini_api_key: str = "AIzaSyBS_DFCJh82voYIKoglS-ow6ezGNg775pg"  # Replace with your API key
    log_file: str = "chat_history.txt"
    language: str = "english"
    farmer_info: Optional[FarmerInfo] = None

class ChatLogger:
    """Logger for chat interactions"""
    def __init__(self, log_file: str):
        self.log_file = log_file
        logging.basicConfig(
            filename=log_file,
            level=logging.INFO,
            format='%(asctime)s - %(message)s'
        )
        
    def log_interaction(self, question: str, answer: str, farmer_info: Optional[FarmerInfo] = None):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        farmer_details = ""
        if farmer_info:
            farmer_details = f"\nFarmer: {farmer_info.name or 'Anonymous'} | Location: {farmer_info.location} | Crop: {farmer_info.crop_type}"
        
        log_entry = f"\n[{timestamp}]{farmer_details}\nQ: {question}\nA: {answer}\n{'-'*50}"
        
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(log_entry)
        logging.info(f"Chat interaction logged - Question: {question[:50]}...")

class ChatMemory:
    """Manages chat history with farmer context"""
    def __init__(self, max_history: int = 3):
        self.max_history = max_history
        self.history = []
        
    def add_interaction(self, question: str, answer: str, farmer_info: Optional[FarmerInfo] = None):
        interaction = {
            "question": question,
            "answer": answer,
            "timestamp": datetime.now().isoformat()
        }
        if farmer_info:
            interaction["farmer_info"] = {
                "location": farmer_info.location,
                "crop_type": farmer_info.crop_type,
                "purchase_status": farmer_info.purchase_status
            }
        
        self.history.append(interaction)
        if len(self.history) > self.max_history:
            self.history.pop(0)
            
    def get_history(self) -> List[Dict[str, Any]]:
        return self.history
    
    def get_context_string(self) -> str:
        context = []
        for interaction in self.history:
            context.append(f"Q: {interaction['question']}\nA: {interaction['answer']}")
        return "\n\n".join(context)
    
    def clear_history(self):
        self.history = []

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
        
    async def generate_questions(self, 
                               question: str, 
                               answer: str, 
                               farmer_info: Optional[FarmerInfo] = None,
                               language: str = "english") -> List[str]:
        try:
            chat = self.model.start_chat(history=[])
            
            lang_instruction = "in Hindi" if language == "hindi" else "in English"
            crop_context = f"for {farmer_info.crop_type} farming" if farmer_info else ""
            
            prompt = f"""Based on this agricultural product interaction:
            
            Question: {question}
            Answer: {answer}
            
            Generate 4 relevant follow-up questions {lang_instruction} that a farmer might ask about GAPL Starter {crop_context}.
            Focus on:
            - Practical application methods and timing
            - Real benefits and effectiveness
            - Specific crop compatibility and usage
            - Results and experiences from other farmers
            
            Return ONLY the numbered questions (1-4), one per line.
            """
            
            response = chat.send_message(prompt).text
            
            questions = []
            for line in response.split('\n'):
                line = line.strip()
                if line and (line.startswith('1.') or line.startswith('2.') or 
                           line.startswith('3.') or line.startswith('4.')):
                    questions.append(line.split('.', 1)[1].strip())
            
            while len(questions) < 4:
                default_q = "कृपया GAPL Starter के बारे में और जानकारी दें?" if language == "hindi" else "Can you tell me more about GAPL Starter?"
                questions.append(default_q)
            
            return questions[:4]
            
        except Exception as e:
            logging.error(f"Error generating questions: {str(e)}")
            default_questions = {
                "hindi": [
                    "GAPL Starter को कैसे स्टोर करें?",
                    "क्या इसे अन्य उर्वरकों के साथ मिला सकते हैं?",
                    "इससे क्या परिणाम मिलेंगे?",
                    "क्या यह सभी प्रकार की मिट्टी के लिए सुरक्षित है?"
                ],
                "english": [
                    "How should I store GAPL Starter?",
                    "Can I mix it with other fertilizers?",
                    "What results can I expect to see?",
                    "Is it safe for all soil types?"
                ]
            }
            return default_questions["hindi" if language == "hindi" else "english"]

class GeminiRAG:
    """RAG implementation using Gemini with language support"""
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
        """Creates a context string from relevant documents"""
        return "\n\n".join(doc['content'] for doc in relevant_docs)
        
    async def get_answer(self, 
                        question: str, 
                        context: str, 
                        farmer_info: Optional[FarmerInfo] = None,
                        language: str = "english") -> str:
        try:
            chat = self.model.start_chat(history=[])
            
            lang_instruction = "Respond in Hindi, using simple farmer-friendly language." if language == "hindi" else "Use simple farmer-friendly language."
            
            farmer_context = ""
            if farmer_info:
                farmer_context = f"""You're speaking with a farmer from {farmer_info.location} who:
                - Grows {farmer_info.crop_type}
                - Is {farmer_info.purchase_status} GAPL Starter
                - Should be addressed as {farmer_info.name if farmer_info.name else 'respected farmer'}
                """
            
            prompt = f"""You are a friendly agricultural expert specializing in GAPL Starter bio-fertilizer. 
            
            {farmer_context}
            
            Background information:
            {context}

            Farmer's question: {question}

            Instructions:
            - {lang_instruction}
            - Be conversational and empathetic
            - Give practical, actionable advice
            - Share relevant farmer success stories when possible
            - Use local agricultural context when relevant
            - If uncertain about specific details, be honest and focus on what you know
            
            Remember to maintain a helpful, friendly tone throughout your response.
            """
            
            response = chat.send_message(prompt)
            return response.text
            
        except Exception as e:
            logging.error(f"Error generating answer: {str(e)}")
            error_msg = {
                "hindi": "क्षमा करें, मैं आपके प्रश्न को प्रोसेस करने में असमर्थ हूं। कृपया पुनः प्रयास करें।",
                "english": "I apologize, but I'm having trouble processing your request. Please try again."
            }
            return error_msg["hindi" if language == "hindi" else "english"]

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
