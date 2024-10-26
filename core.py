# core.py (Part 1 - First half)

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
from googletrans import Translator

@dataclass
class ChatConfig:
    """Configuration for the chatbot"""
    embedding_model_name: str = 'all-MiniLM-L6-v2'
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    max_history: int = 3
    gemini_api_key: str = "YOUR_API_KEY"  # Replace with your API key
    log_file: str = "chat_history.txt"
    default_language: str = "english"

class LanguageTranslator:
    """Handles translation between English and Hindi"""
    def __init__(self):
        self.translator = Translator()
        
    def to_english(self, text: str) -> str:
        try:
            if not text:
                return text
            result = self.translator.translate(text, dest='en')
            return result.text
        except Exception as e:
            logging.error(f"Translation error: {str(e)}")
            return text
            
    def to_hindi(self, text: str) -> str:
        try:
            if not text:
                return text
            result = self.translator.translate(text, dest='hi')
            return result.text
        except Exception as e:
            logging.error(f"Translation error: {str(e)}")
            return text

class ChatLogger:
    """Logger for chat interactions with metadata"""
    def __init__(self, log_file: str):
        self.log_file = log_file
        
    def log_interaction(self, question: str, answer: str, metadata: Dict[str, Any] = None):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(f"\n[{timestamp}]\nQ: {question}\nA: {answer}")
            if metadata:
                f.write(f"\nMetadata: {metadata}")
            f.write(f"\n{'-'*50}")

class ChatMemory:
    """Manages chat history with metadata"""
    def __init__(self, max_history: int = 3):
        self.max_history = max_history
        self.history = []
        self.metadata = {}
        
    def add_interaction(self, question: str, answer: str, metadata: Dict[str, Any] = None):
        self.history.append({
            "question": question,
            "answer": answer,
            "metadata": metadata
        })
        if len(self.history) > self.max_history:
            self.history.pop(0)
            
    def get_history(self) -> List[Dict[str, Any]]:
        return self.history
    
    def clear_history(self):
        self.history = []
        
    def update_metadata(self, new_metadata: Dict[str, Any]):
        self.metadata.update(new_metadata)

class MetadataExtractor:
    """Extracts and validates user metadata from responses"""
    @staticmethod
    def extract_mobile(text: str) -> Optional[str]:
        # Remove all non-numeric characters
        numbers = ''.join(filter(str.isdigit, text))
        if len(numbers) == 10:
            return numbers
        return None
    
    @staticmethod
    def extract_location(text: str) -> Optional[str]:
        # Simple validation for now - could be enhanced with regex/pincode validation
        if len(text.strip()) > 2:
            return text.strip()
        return None
    
    @staticmethod
    def extract_crop(text: str) -> Optional[str]:
        # Basic crop name extraction
        if len(text.strip()) > 2:
            return text.strip()
        return None

# core.py (Part 2 - Second half)

class QuestionGenerator:
    """Generates follow-up questions with metadata collection priority"""
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
        self.translator = LanguageTranslator()
        
    async def generate_questions(self, question: str, answer: str, metadata: Dict[str, Any], 
                               language: str = "english") -> List[str]:
        try:
            chat = self.model.start_chat(history=[])
            missing_metadata = self._get_missing_metadata(metadata)
            
            # Generate context-aware questions
            if missing_metadata:
                questions = self._generate_metadata_questions(missing_metadata, question, answer)
            else:
                questions = self._generate_product_questions(question, answer)
            
            # Translate if needed
            if language.lower() == "hindi":
                questions = [self.translator.to_hindi(q) for q in questions]
                
            return questions[:3]
        except Exception as e:
            logging.error(f"Error generating questions: {str(e)}")
            return self._get_fallback_questions(language)
            
    def _get_missing_metadata(self, metadata: Dict[str, Any]) -> List[str]:
        missing = []
        if not metadata.get('mobile_number'):
            missing.append('mobile')
        if not metadata.get('location'):
            missing.append('location')
        if not metadata.get('crop_name'):
            missing.append('crop')
        return missing
        
    def _generate_metadata_questions(self, missing_fields: List[str], 
                                   question: str, answer: str) -> List[str]:
        metadata_questions = {
            'mobile': "Could you please share your contact number for personalized assistance?",
            'location': "Which area/district are you farming in?",
            'crop': "Which crops are you currently growing or planning to grow?"
        }
        
        questions = [metadata_questions[field] for field in missing_fields[:2]]
        
        # Add one product-related question
        prompt = f"""Based on: Q: {question} A: {answer}
        Generate 1 relevant follow-up question about GAPL Starter's benefits or application.
        Return ONLY the question."""
        
        try:
            response = self.model.generate_content(prompt).text.strip()
            questions.append(response)
        except:
            questions.append("How would you like to apply GAPL Starter in your field?")
            
        return questions
        
    def _generate_product_questions(self, question: str, answer: str) -> List[str]:
        prompt = f"""Based on this interaction about GAPL Starter:
        Question: {question}
        Answer: {answer}
        Generate 3 relevant follow-up questions focusing on:
        - Practical application
        - Benefits and results
        - Timing and usage
        Return ONLY the questions, one per line."""
        
        try:
            response = self.model.generate_content(prompt).text
            return [q.strip().rstrip('?') + '?' for q in response.split('\n') if q.strip()]
        except:
            return self._get_fallback_questions("english")
            
    def _get_fallback_questions(self, language: str) -> List[str]:
        questions = [
            "How should I apply GAPL Starter?",
            "What results can I expect?",
            "When is the best time to apply?"
        ]
        
        if language.lower() == "hindi":
            return [self.translator.to_hindi(q) for q in questions]
        return questions

class GeminiRAG:
    """RAG implementation with language support"""
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
        self.translator = LanguageTranslator()
        
    def create_context(self, relevant_docs: List[Dict[str, Any]]) -> str:
        """Create context from relevant documents"""
        context_parts = []
        for doc in relevant_docs:
            context_parts.append(f"Section: {doc['metadata']['section']}\n{doc['content']}")
        return "\n\n".join(context_parts)
        
    async def get_answer(self, question: str, context: str, 
                        metadata: Dict[str, Any], language: str = "english") -> str:
        try:
            # Translate question to English if needed
            if language.lower() == "hindi":
                question = self.translator.to_english(question)
            
            chat = self.model.start_chat(history=[])
            
            # Create prompt with metadata context
            prompt = self._create_prompt(question, context, metadata)
            
            response = chat.send_message(prompt).text
            
            # Translate response if needed
            if language.lower() == "hindi":
                response = self.translator.to_hindi(response)
                
            return response
            
        except Exception as e:
            logging.error(f"Error generating answer: {str(e)}")
            error_msg = "I apologize, but I'm having trouble processing your request. Please try again."
            return self.translator.to_hindi(error_msg) if language.lower() == "hindi" else error_msg
            
    def _create_prompt(self, question: str, context: str, metadata: Dict[str, Any]) -> str:
        # Create a more personalized prompt using available metadata
        crop_context = f" for {metadata['crop_name']}" if metadata.get('crop_name') else ""
        location_context = f" in {metadata['location']}" if metadata.get('location') else ""
        
        prompt = f"""You are an expert agricultural consultant specializing in GAPL Starter bio-fertilizer. 
        Provide specific advice{crop_context}{location_context}.
        
        Background information:
        {context}

        Question: {question}

        Respond naturally as an expert would, keeping these guidelines in mind:
        - Be concise but informative
        - Include specific dosage and application advice when relevant
        - Mention safety precautions where applicable
        - Use simple, clear language
        - If you don't know something specific, be honest about it"""
        
        return prompt


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
