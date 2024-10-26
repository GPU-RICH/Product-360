import os
import json
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import torch
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
import google.generativeai as genai
from datetime import datetime

# Language translations
TRANSLATIONS = {
    "en": {
        "storage_prompt": "To provide you with the most accurate storage guidance for your specific situation, could you tell me where you're located?",
        "purchase_prompt": "Have you already purchased GAPL Starter or are you planning to?",
        "mobile_prompt": "To help you get the best support, could I have your mobile number?",
        "crop_prompt": "Which crop are you currently growing or planning to use GAPL Starter with?",
        "follow_up_prefix": "You might also want to know:",
    },
    "hi": {
        "storage_prompt": "आपको सटीक भंडारण मार्गदर्शन प्रदान करने के लिए, क्या आप मुझे बता सकते हैं कि आप कहाँ स्थित हैं?",
        "purchase_prompt": "क्या आपने GAPL Starter पहले से खरीदा है या खरीदने की योजना बना रहे हैं?",
        "mobile_prompt": "आपको बेहतर सहायता प्रदान करने के लिए, क्या मैं आपका मोबाइल नंबर जान सकता हूं?",
        "crop_prompt": "आप वर्तमान में कौन सी फसल उगा रहे हैं या GAPL Starter के साथ उपयोग करने की योजना बना रहे हैं?",
        "follow_up_prefix": "आप यह भी जानना चाह सकते हैं:",
    }
}

@dataclass
class ChatConfig:
    """Configuration for the chatbot"""
    embedding_model_name: str = 'all-MiniLM-L6-v2'
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    max_history: int = 3
    gemini_api_key: str = "YOUR_API_KEY"
    log_file: str = "chat_history.txt"
    user_data_file: str = "user_data.json"

class UserDataManager:
    """Manages user data collection and storage"""
    def __init__(self, data_file: str):
        self.data_file = data_file
        self.required_fields = ['mobile_number', 'location', 'purchase_status', 'crop']
        
    def save_user_data(self, user_id: str, data: Dict[str, Any]):
        try:
            if os.path.exists(self.data_file):
                with open(self.data_file, 'r') as f:
                    all_data = json.load(f)
            else:
                all_data = {}
            
            all_data[user_id] = data
            
            with open(self.data_file, 'w') as f:
                json.dump(all_data, f, indent=4)
                
        except Exception as e:
            logging.error(f"Error saving user data: {str(e)}")
    
    def get_user_data(self, user_id: str) -> Dict[str, Any]:
        try:
            if os.path.exists(self.data_file):
                with open(self.data_file, 'r') as f:
                    all_data = json.load(f)
                return all_data.get(user_id, {})
            return {}
        except Exception as e:
            logging.error(f"Error reading user data: {str(e)}")
            return {}

class ChatLogger:
    """Logger for chat interactions"""
    def __init__(self, log_file: str):
        self.log_file = log_file
        
    def log_interaction(self, question: str, answer: str, user_id: str, language: str):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(f"\n[{timestamp}] [User: {user_id}] [Lang: {language}]\nQ: {question}\nA: {answer}\n{'-'*50}")

class ChatMemory:
    """Manages chat history with user context"""
    def __init__(self, max_history: int = 3):
        self.max_history = max_history
        self.history = []
        self.user_context = {}
        
    def add_interaction(self, question: str, answer: str, user_context: Dict[str, Any] = None):
        self.history.append({
            "question": question,
            "answer": answer,
            "context": user_context or {}
        })
        if len(self.history) > self.max_history:
            self.history.pop(0)
    
    def update_user_context(self, context: Dict[str, Any]):
        self.user_context.update(context)
            
    def get_history(self) -> List[Dict[str, Any]]:
        return self.history
    
    def clear_history(self):
        self.history = []
        self.user_context = {}

class QuestionGenerator:
    """Generates follow-up questions using Gemini with language support"""
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
        
    async def generate_questions(self, question: str, answer: str, language: str = "en") -> List[str]:
        try:
            chat = self.model.start_chat(history=[])
            
            language_instruction = "in Hindi. Use Devanagari script." if language == "hi" else "in English."
            
            prompt = f"""Based on this product information interaction:
            
            Question: {question}
            Answer: {answer}
            
            Generate 4 relevant follow-up questions that a farmer might ask about GAPL Starter {language_instruction}
            Focus on:
            - Application methods and timing
            - Benefits and effectiveness
            - Compatibility with specific crops
            - Scientific backing and results
            
            Return ONLY the numbered questions (1-4), one per line.
            Make the questions sound natural and conversational, as if a farmer is asking them.
            """
            
            response = chat.send_message(prompt).text
            
            questions = []
            for line in response.split('\n'):
                line = line.strip()
                if line and (line.startswith('1.') or line.startswith('2.') or 
                           line.startswith('3.') or line.startswith('4.')):
                    questions.append(line.split('.', 1)[1].strip())
            
            default_question = "Can you tell me more about GAPL Starter?" if language == "en" else "क्या आप मुझे GAPL Starter के बारे में और बता सकते हैं?"
            while len(questions) < 4:
                questions.append(default_question)
            
            return questions[:4]
            
        except Exception as e:
            logging.error(f"Error generating questions: {str(e)}")
            return self._get_default_questions(language)
    
    def _get_default_questions(self, language: str) -> List[str]:
        if language == "en":
            return [
                "How should I store GAPL Starter?",
                "Can I mix it with other fertilizers?",
                "What results can I expect to see?",
                "Is it safe for all soil types?"
            ]
        else:
            return [
                "मैं GAPL Starter को कैसे स्टोर करूं?",
                "क्या मैं इसे अन्य उर्वरकों के साथ मिला सकता हूं?",
                "मुझे कैसे परिणाम दिखाई देंगे?",
                "क्या यह सभी प्रकार की मिट्टी के लिए सुरक्षित है?"
            ]

class GeminiRAG:
    """RAG implementation using Gemini with enhanced conversation and language support"""
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.generation_config = {
            "temperature": 0.2,
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
        
    async def get_answer(self, question: str, context: str, user_data: Dict[str, Any], language: str = "en") -> Dict[str, Any]:
        try:
            chat = self.model.start_chat(history=[])
            
            language_instruction = "Respond in Hindi using Devanagari script." if language == "hi" else "Respond in English."
            
            user_context = ""
            if user_data:
                user_context = f"""User Context:
                - Location: {user_data.get('location', 'Unknown')}
                - Crop: {user_data.get('crop', 'Unknown')}
                - Purchase Status: {user_data.get('purchase_status', 'Unknown')}
                """
            
            prompt = f"""{language_instruction}
            
            You are a friendly agricultural expert who specializes in GAPL Starter bio-fertilizer. 
            Speak naturally and conversationally, as if talking to a farmer friend.
            Use simple, clear language while maintaining expertise.
            
            {user_context}
            
            Background information:
            {context}

            Farmer's question: {question}

            If you don't have specific information, say something like:
            "I understand you want to know about [topic], but I should check the specific details to give you the most accurate advice. 
            What I can tell you from my experience is..."

            If user data is missing and it would help provide a better answer, naturally ask for it within your response.
            Use the translation dictionary for data collection prompts.
            
            Remember to:
            - Be friendly and conversational
            - Use practical examples
            - Relate to farming experience
            - Show you understand their concerns
            """
            
            response = chat.send_message(prompt)
            
            # Check if we should collect any user data
            data_collection = None
            if not user_data.get('location') and ('store' in question.lower() or 'storage' in question.lower()):
                data_collection = 'location'
            elif not user_data.get('crop') and ('apply' in question.lower() or 'use' in question.lower()):
                data_collection = 'crop'
            elif not user_data.get('purchase_status') and ('buy' in question.lower() or 'price' in question.lower()):
                data_collection = 'purchase_status'
            
            return {
                "answer": response.text,
                "collect_data": data_collection,
                "prompt_text": TRANSLATIONS[language].get(f"{data_collection}_prompt") if data_collection else None
            }
            
        except Exception as e:
            logging.error(f"Error generating answer: {str(e)}")
            return {
                "answer": "I apologize, but I'm having trouble processing your request. Please try again.",
                "collect_data": None,
                "prompt_text": None
            }

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
