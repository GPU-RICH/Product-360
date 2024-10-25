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
class ChatConfig:
    """Configuration for the chatbot"""
    embedding_model_name: str = 'all-MiniLM-L6-v2'
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    max_history: int = 3
    gemini_api_key: str = "AIzaSyBS_DFCJh82voYIKoglS-ow6ezGNg775pg"  # Replace with your API key
    log_file: str = "chat_history.txt"

class ChatLogger:
    """Logger for chat interactions"""
    def __init__(self, log_file: str):
        self.log_file = log_file
        
    def log_interaction(self, question: str, answer: str):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(f"\n[{timestamp}]\nQ: {question}\nA: {answer}\n{'-'*50}")

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
    """Generates follow-up questions using Gemini with prioritized metadata collection"""
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
            chat = self.model.start_chat(history=[])
            metadata = st.session_state.user_metadata
            message_count = st.session_state.message_counter
            
            # Prioritize metadata collection in first 3 interactions
            if message_count < 3:
                missing_metadata = []
                if not metadata.get('mobile_number'):
                    missing_metadata.append("To provide personalized assistance, could you share your contact number?")
                if not metadata.get('crop_name'):
                    missing_metadata.append("Which crops are you currently growing or planning to use GAPL Starter for?")
                if not metadata.get('location'):
                    missing_metadata.append("What's your pincode/location for region-specific recommendations?")
                
                if missing_metadata:
                    # For first interaction, return 1 product question and 2 metadata questions
                    if message_count == 0:
                        prompt = f"""Based on this interaction about GAPL Starter:
                        Question: {question}
                        Answer: {answer}
                        Generate 1 relevant follow-up question about GAPL Starter's benefits or application.
                        Return ONLY the question, without any prefixes."""
                        
                        response = chat.send_message(prompt).text.strip()
                        questions = [response.rstrip('?') + '?']
                        questions.extend(missing_metadata[:2])
                        return questions
                    
                    # For next interactions, include at least one metadata question
                    prompt = f"""Based on this interaction about GAPL Starter:
                    Question: {question}
                    Answer: {answer}
                    Generate 2 relevant follow-up questions about GAPL Starter.
                    Focus on practical application and benefits.
                    Return ONLY the questions, one per line."""
                    
                    response = chat.send_message(prompt).text
                    questions = [q.strip().rstrip('?') + '?' for q in response.split('\n') if q.strip()][:2]
                    questions.append(missing_metadata[0])
                    return questions
            
            # Regular question generation after metadata collection
            prompt = f"""Based on this interaction about GAPL Starter:
            Question: {question}
            Answer: {answer}
            Generate 3 relevant follow-up questions about GAPL Starter.
            Focus on practical application, benefits, and results.
            Return ONLY the questions, one per line."""
            
            response = chat.send_message(prompt).text
            questions = [q.strip().rstrip('?') + '?' for q in response.split('\n') if q.strip()]
            return questions[:3]
            
        except Exception as e:
            logging.error(f"Error generating questions: {str(e)}")
            return [
                "How should I apply GAPL Starter?",
                "What results can I expect?",
                "Could you share your contact number for personalized assistance?"
            ]

class GeminiRAG:
    """RAG implementation using Gemini with enhanced metadata handling"""
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
        
    def _process_metadata_response(self, question: str) -> Optional[str]:
        """Process and store metadata responses"""
        question_lower = question.lower().strip()
        metadata = st.session_state.user_metadata
        
        # Handle mobile number
        if question_lower.replace(" ", "").isdigit() and len(question_lower.replace(" ", "")) == 10:
            metadata['mobile_number'] = question_lower.replace(" ", "")
            return "Thank you for sharing your contact number. I'll ensure you receive personalized assistance for GAPL Starter. What would you like to know about its application?"
        
        # Handle crop information
        crop_keywords = ['growing', 'cultivating', 'farming', 'crop']
        if any(word in question_lower for word in crop_keywords) or len(question_lower.split()) <= 4:
            metadata['crop_name'] = question
            return f"Excellent! GAPL Starter has shown great results with {question}. Let me provide you with specific recommendations for your crop. Would you mind sharing your location for region-specific advice?"
        
        # Handle location
        location_keywords = ['pincode', 'location', 'district', 'village']
        if any(word in question_lower for word in location_keywords) or question_lower.isdigit():
            metadata['location'] = question
            return f"Thank you for sharing your location. I can now provide recommendations tailored to your region's conditions. What specific aspect of GAPL Starter would you like to know about?"
        
        return None
        
    async def get_answer(self, question: str, context: str) -> str:
        try:
            # First check if this is a metadata response
            metadata_response = self._process_metadata_response(question)
            if metadata_response:
                return metadata_response
            
            chat = self.model.start_chat(history=[])
            prompt = f"""You are an expert agricultural consultant specializing in GAPL Starter bio-fertilizer. 
            You have extensive hands-on experience with the product and deep knowledge of its applications and benefits.
            
            Background information:
            {context}

            Question from farmer: {question}

            Respond naturally as an expert would, keeping responses concise but informative. If you know the farmer's crop type
            from the metadata, include specific advice for that crop."""
            
            response = chat.send_message(prompt)
            return response.text
            
        except Exception as e:
            logging.error(f"Error generating answer: {str(e)}")
            return "I apologize, but I'm having trouble processing your request. Please try again."

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
