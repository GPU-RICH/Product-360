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
import json

@dataclass
class ChatConfig:
    """Configuration for the chatbot"""
    embedding_model_name: str = 'all-MiniLM-L6-v2'
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    max_history: int = 4
    gemini_api_key: str = "AIzaSyBS_DFCJh82voYIKoglS-ow6ezGNg775pg"  # Replace with your API key
    log_file: str = "chat_history.txt"
    user_data_file: str = "farmer_data.json"
    default_language: str = "english"

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

class UserDataManager:
    """Manages user data persistence"""
    def __init__(self, data_file: str):
        self.data_file = data_file
        self.ensure_file_exists()

    def ensure_file_exists(self):
        if not os.path.exists(self.data_file):
            with open(self.data_file, 'w', encoding='utf-8') as f:
                json.dump({}, f)

    def save_user_data(self, user_id: str, data: Dict[str, Any]):
        try:
            with open(self.data_file, 'r+', encoding='utf-8') as f:
                all_data = json.load(f)
                all_data[user_id] = data
                f.seek(0)
                json.dump(all_data, f, indent=2)
                f.truncate()
        except Exception as e:
            logging.error(f"Error saving user data: {str(e)}")

    def get_user_data(self, user_id: str) -> Dict[str, Any]:
        try:
            with open(self.data_file, 'r', encoding='utf-8') as f:
                all_data = json.load(f)
                return all_data.get(user_id, {})
        except Exception as e:
            logging.error(f"Error reading user data: {str(e)}")
            return {}

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
    def extract_metadata(text: str) -> Dict[str, Any]:
        metadata = {}
        
        # Look for phone numbers (10 digits)
        import re
        phone_match = re.search(r'\b\d{10}\b', text)
        if phone_match:
            metadata['mobile_number'] = phone_match.group(0)
        
        # Look for location mentions
        # This is a simple implementation - could be enhanced with NER
        location_indicators = ['in', 'at', 'from', 'near']
        words = text.lower().split()
        for i, word in enumerate(words):
            if word in location_indicators and i + 1 < len(words):
                metadata['location'] = words[i + 1].capitalize()
                break
        
        # Look for crop mentions
        common_crops = ['wheat', 'rice', 'cotton', 'sugarcane', 'corn', 'maize']
        found_crops = [crop for crop in common_crops if crop in text.lower()]
        if found_crops:
            metadata['crop_name'] = found_crops[0].capitalize()
        
        return metadata

class GeminiRAG:
    """RAG implementation with natural language support"""
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.generation_config = {
            "temperature": 0.7,
            "top_p": 0.95,
            "top_k": 64,
            "max_output_tokens": 8192,
        }
        self.model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            generation_config=self.generation_config
        )
        
    def create_context(self, relevant_docs: List[Dict[str, Any]]) -> str:
        context_parts = []
        for doc in relevant_docs:
            context_parts.append(f"Section: {doc['metadata']['section']}\n{doc['content']}")
        return "\n\n".join(context_parts)
        
    async def get_answer(self, question: str, context: str, 
                        metadata: Dict[str, Any], language: str = "english") -> str:
        try:
            chat = self.model.start_chat(history=[])
            prompt = self._create_prompt(question, context, metadata, language)
            response = chat.send_message(prompt).text
            return response
        except Exception as e:
            logging.error(f"Error generating answer: {str(e)}")
            error_msg = "I'm having trouble processing your request. Please try again."
            return error_msg if language == "english" else "मुझे आपका सवाल समझने में दिक्कत हो रही है। कृपया दोबारा प्रयास करें।"
            
    def _create_prompt(self, question: str, context: str, metadata: Dict[str, Any], language: str) -> str:
        # Determine if we need to collect user information
        needs_info = not all(metadata.get(key) for key in ['mobile_number', 'location', 'crop_name'])
        
        base_prompt = {
            "english": """You are an expert agricultural consultant specializing in GAPL Starter bio-fertilizer. 
                      Respond in English. Keep responses natural and conversational.""",
            "hindi": """आप GAPL Starter जैविक उर्वरक के विशेषज्ञ कृषि सलाहकार हैं।
                     हिंदी में जवाब दें। जवाब सरल और बातचीत की तरह होना चाहिए।"""
        }

        info_collection = {
            "english": """If the user hasn't shared their details, naturally ask for relevant information like:
                      - Their location/district
                      - Which crops they're growing
                      - Their contact number
                      Make this part of your natural conversation, don't make it feel like a form.""",
            "hindi": """अगर उपयोगकर्ता ने अपनी जानकारी साझा नहीं की है, तो सहज रूप से पूछें:
                     - उनका स्थान/जिला
                     - वे कौन सी फसलें उगा रहे हैं
                     - उनका संपर्क नंबर
                     इसे सामान्य बातचीत का हिस्सा बनाएं।"""
        }

        prompt = f"""{base_prompt[language]}

        Background information:
        {context}

        Question: {question}

        Current user information:
        {metadata}

        Guidelines:
        - Be concise but informative
        - Include specific dosage and application advice when relevant
        - Mention safety precautions where applicable
        - Use simple, clear language
        - If you don't know something specific, be honest about it
        {info_collection[language] if needs_info else ''}
        """
        
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
        if not self.vectorstore:
            raise ValueError("Database not initialized. Please process documents first.")
        
        try:
            docs = self.vectorstore.similarity_search(query, k=k)
            return [{"content": doc.page_content, "metadata": doc.metadata} for doc in docs]
        except Exception as e:
            logging.error(f"Error during search: {str(e)}")
            return []
