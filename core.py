import os
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
import torch
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
import google.generativeai as genai
from datetime import datetime

class Language(Enum):
    ENGLISH = "english"
    HINDI = "hindi"

@dataclass
class ChatConfig:
    """Configuration for the chatbot"""
    embedding_model_name: str = 'all-MiniLM-L6-v2'
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    max_history: int = 3
    gemini_api_key: str = "AIzaSyBS_DFCJh82voYIKoglS-ow6ezGNg775pg"  # Replace with your API key
    log_file: str = "chat_history.txt"
    default_language: Language = Language.ENGLISH

class ChatLogger:
    """Logger for chat interactions"""
    def __init__(self, log_file: str):
        self.log_file = log_file
        
    def log_interaction(self, question: str, answer: str, language: Language):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(f"\n[{timestamp}] [{language.value}]\nQ: {question}\nA: {answer}\n{'-'*50}")

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
            
            language_instruction = (
                "Return the questions in Hindi." if language == Language.HINDI
                else "Return the questions in English."
            )
            
            prompt = f"""Based on this product information interaction:
            
            Question: {question}
            Answer: {answer}
            
            Generate 4 relevant follow-up questions that a customer might ask about GAPL Starter.
            Focus on:
            - Application methods and timing
            - Benefits and effectiveness
            - Compatibility with specific crops
            - Scientific backing and results
            
            {language_instruction}
            Return ONLY the numbered questions (1-4), one per line.
            """
            
            response = chat.send_message(prompt).text
            
            questions = []
            for line in response.split('\n'):
                line = line.strip()
                if line and (line.startswith('1.') or line.startswith('2.') or 
                           line.startswith('3.') or line.startswith('4.')):
                    questions.append(line.split('.', 1)[1].strip())
            
            default_questions = {
                Language.ENGLISH: [
                    "Can you provide more details about GAPL Starter?",
                    "What are the application methods?",
                    "What results can I expect to see?",
                    "Is it safe for all soil types?"
                ],
                Language.HINDI: [
                    "क्या आप GAPL Starter के बारे में और जानकारी दे सकते हैं?",
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
        """Creates a context string from relevant documents"""
        return "\n\n".join(doc['content'] for doc in relevant_docs)
        
    async def get_answer(self, question: str, context: str, language: Language) -> str:
        try:
            chat = self.model.start_chat(history=[])
            
            language_instruction = (
                "Respond in fluent Hindi, using Devanagari script." if language == Language.HINDI
                else "Respond in English."
            )
            
            prompt = f"""You are an expert agricultural consultant specializing in GAPL Starter bio-fertilizer. 
            You have extensive hands-on experience with the product and deep knowledge of its applications and benefits.
            
            {language_instruction}
            
            Background information to inform your response:
            {context}

            Question from farmer: {question}

            Respond naturally as an expert would, without referencing any "provided information" or documentation.
            Your response should be:
            - Confident and authoritative
            - Direct and practical
            - Focused on helping farmers succeed
            - Based on product expertise

            If you don't have enough specific information to answer the question, respond appropriately in the specified language 
            acknowledging the limitations while sharing what you do know about the product.

            Remember to maintain a helpful, expert tone throughout your response.
            """
            
            response = chat.send_message(prompt)
            return response.text
            
        except Exception as e:
            logging.error(f"Error generating answer: {str(e)}")
            default_error = {
                Language.ENGLISH: "I apologize, but I'm having trouble processing your request. Please try again.",
                Language.HINDI: "क्षमा करें, मैं आपके प्रश्न को प्रोसेस करने में असमर्थ हूं। कृपया पुनः प्रयास करें।"
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
