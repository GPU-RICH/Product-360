import os
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import torch
from PIL import Image
from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPModel
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
import google.generativeai as genai
from datetime import datetime
import json
import numpy as np
from pathlib import Path
import pandas as pd

class Language(Enum):
    ENGLISH = "english"
    HINDI = "hindi"

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
    clip_model_name: str = 'openai/clip-vit-base-patch32'
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    max_history: int = 3
    gemini_api_key: str = "AIzaSyBS_DFCJh82voYIKoglS-ow6ezGNg775pg"  # Replace with your API key
    log_file: str = "chat_history.txt"
    user_data_file: str = "user_data.json"
    image_data_file: str = "image_data.json"
    default_language: Language = Language.ENGLISH

class ImageProcessor:
    """Handles image processing and similarity search"""
    def __init__(self, config: ChatConfig):
        self.config = config
        self.model = CLIPModel.from_pretrained(config.clip_model_name)
        self.processor = CLIPProcessor.from_pretrained(config.clip_model_name)
        self.model.to(config.device)
        self.image_database = self.load_image_database()

    def load_image_database(self) -> Dict:
        """Load image database from JSON file"""
        if Path(self.config.image_data_file).exists():
            with open(self.config.image_data_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}

    def save_image_database(self):
        """Save image database to JSON file"""
        with open(self.config.image_data_file, 'w', encoding='utf-8') as f:
            json.dump(self.image_database, f, indent=4)

    def generate_image_embedding(self, image: Image.Image) -> np.ndarray:
        """Generate CLIP embedding for an image"""
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.config.device) for k, v in inputs.items()}
        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)
        return image_features.cpu().numpy()

    def add_image(self, 
                 image_path: str, 
                 problem_type: str, 
                 description: str,
                 solution: str) -> bool:
        """Add new image and its information to the database"""
        try:
            image = Image.open(image_path)
            embedding = self.generate_image_embedding(image)
            
            image_id = f"img_{len(self.image_database)}"
            
            self.image_database[image_id] = {
                "path": str(image_path),
                "problem_type": problem_type,
                "description": description,
                "solution": solution,
                "embedding": embedding.tolist()
            }
            
            self.save_image_database()
            return True
        except Exception as e:
            logging.error(f"Error adding image: {str(e)}")
            return False

    def find_similar_images(self, query_image: Image.Image, k: int = 3) -> List[Dict]:
        """Find similar images using CLIP embeddings"""
        try:
            query_embedding = self.generate_image_embedding(query_image)
            
            similarities = []
            for image_id, image_data in self.image_database.items():
                stored_embedding = np.array(image_data["embedding"])
                similarity = np.dot(query_embedding.flatten(), stored_embedding.flatten())
                similarities.append((similarity, image_data))
            
            similarities.sort(key=lambda x: x[0], reverse=True)
            return [item[1] for item in similarities[:k]]
        except Exception as e:
            logging.error(f"Error finding similar images: {str(e)}")
            return []

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
    
    def save_user_info(self, user_info: UserInfo) -> bool:
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
        
    def log_interaction(self, 
                       question: str, 
                       answer: str, 
                       language: Language, 
                       user_info: Optional[UserInfo] = None,
                       has_image: bool = False):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.log_file, 'a', encoding='utf-8') as f:
            user_context = ""
            if user_info:
                user_context = f"\nUser: {user_info.name} | Location: {user_info.location} | Crop: {user_info.crop_type}"
            image_info = "\nImage query: Yes" if has_image else ""
            f.write(f"\n[{timestamp}] [{language.value}]{user_context}{image_info}\nQ: {question}\nA: {answer}\n{'-'*50}")

class ChatMemory:
    """Manages chat history"""
    def __init__(self, max_history: int = 3):
        self.max_history = max_history
        self.history = []
        
    def add_interaction(self, question: str, answer: str, has_image: bool = False):
        self.history.append({
            "question": question, 
            "answer": answer,
            "has_image": has_image
        })
        if len(self.history) > self.max_history:
            self.history.pop(0)
            
    def get_history(self) -> List[Dict[str, Any]]:
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
        
    async def generate_questions(
        self, 
        question: str, 
        answer: str, 
        language: Language, 
        user_info: Optional[UserInfo] = None,
        has_image: bool = False
    ) -> List[str]:
        try:
            chat = self.model.start_chat(history=[])
            
            language_instruction = (
                "Return the questions in Hindi." if language == Language.HINDI
                else "Return the questions in English."
            )
            
            user_context = ""
            if user_info:
                user_context = f"""
                Consider this user context while generating questions:
                - User Name: {user_info.name}
                - Location: {user_info.location}
                - Product Purchase Status: {"Has purchased" if user_info.has_purchased else "Has not purchased"}
                - Crop Type: {user_info.crop_type}
                """
            
            image_context = "The user shared an image of their crop problem. " if has_image else ""
            
            prompt = f"""Based on this product information interaction:
            
            {image_context}
            Question: {question}
            Answer: {answer}
            
            {user_context}
            
            Generate 4 relevant follow-up questions that a farmer might ask about this product.
            
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
                    "Can you provide more details about the product?",
                    "What are the application methods?",
                    "What results can I expect to see?",
                    "Is it safe for all soil types?"
                ],
                Language.HINDI: [
                    "क्या आप के बारे में और जानकारी दे सकते हैं?",
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
    """RAG implementation using Gemini with proper image handling"""
    def __init__(self, api_key: str, image_processor: ImageProcessor):
        genai.configure(api_key=api_key)
        self.generation_config = {
            "temperature": 0,
            "top_p": 0.95,
            "top_k": 64,
            "max_output_tokens": 8192,
            "response_mime_type": "text/plain",
        }
        self.model = genai.GenerativeModel(
            model_name="gemini-1.5-pro",
            generation_config=self.generation_config
        )
        self.image_processor = image_processor

    def create_context(self, relevant_docs: List[Dict[str, Any]]) -> str:
        """Creates a context string from relevant documents"""
        return "\n\n".join(doc['content'] for doc in relevant_docs)

    async def get_answer(
        self, 
        question: str, 
        context: str, 
        language: Language,
        user_info: Optional[UserInfo] = None,
        query_image: Optional[Image.Image] = None
    ) -> Tuple[str, List[Dict]]:
        try:
            language_instruction = (
                "Respond in fluent Hindi, using Devanagari script." if language == Language.HINDI
                else "Respond in English."
            )
            
            user_context = ""
            if user_info:
                user_context = f"""
                Consider this user context while generating your response:
                - You are talking to {user_info.name} from {user_info.location}
                - They {'' if user_info.has_purchased else 'have not '}purchased the product
                - They are growing {user_info.crop_type}
                """

            similar_images = []
            if query_image:
                # Save the PIL Image temporarily
                temp_path = "temp_image.jpg"
                query_image.save(temp_path, "JPEG")
                
                # Upload image to Gemini
                image_file = genai.upload_file(temp_path, mime_type="image/jpeg")
                
                # Find similar cases using CLIP
                similar_images = self.image_processor.find_similar_images(query_image)
                
                # Create image context from similar cases
                image_context = "\n\n".join([
                    f"Similar case found:\n"
                    f"Problem: {img['problem_type']}\n"
                    f"Description: {img['description']}\n"
                    f"Solution: {img['solution']}"
                    for img in similar_images
                ])

                # Start chat with image
                chat = self.model.start_chat(
                    history=[
                        {
                            "role": "user",
                            "parts": [
                                image_file,
                                f"""I am facing this issue with my {user_info.crop_type if user_info else 'crop'}.
                                Question: {question}
                                
                                {language_instruction}
                                {user_context}
                                
                                Consider this product information:
                                {context}
                                
                                Similar cases found:
                                {image_context}
                                
                                Provide a comprehensive response that:
                                1. Analyzes the crop problem in the image
                                2. Explains how the product can help
                                3. Gives specific application recommendations
                                4. Suggests preventive measures"""
                            ],
                        }
                    ]
                )
                
                # Get response
                response = chat.send_message("")
                
                # Clean up temp file
                os.remove(temp_path)
                
            else:
                # Text-only query
                chat = self.model.start_chat()
                response = chat.send_message(
                    f"""{question}
                    
                    {language_instruction}
                    {user_context}
                    
                    Consider this product information:
                    {context}
                    
                    Provide a detailed response focusing on how the product can help."""
                )

            return response.text, similar_images
            
        except Exception as e:
            logging.error(f"Error generating answer: {str(e)}")
            default_error = {
                Language.ENGLISH: "I apologize, but I'm having trouble processing your request. Please try again.",
                Language.HINDI: "क्षमा करें, मैं आपके प्रश्न को प्रोसेस करने में असमर्थ हूं। कृपया पुनः प्रयास करें।"
            }
            return default_error[language], []

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

def init_image_database_from_csv(csv_path: str, image_processor: ImageProcessor):
    """Initialize image database from existing CSV and image files"""
    try:
        # Read the CSV file
        df = pd.read_csv(csv_path, 
                        names=['affected_part', 'initial_diagnosis', 'symptoms', 
                              'damage_type', 'problem_image_url'],
                        sep='\t')
        
        print(f"Processing {len(df)} entries from CSV...")
        
        for idx, row in df.iterrows():
            try:
                # Get image path from URL (assuming last part is filename)
                image_filename = row['problem_image_url'].split('/')[-1]
                image_path = Path('images') / image_filename
                
                if image_path.exists():
                    # Create problem description
                    problem_description = (
                        f"Initial Diagnosis: {row['initial_diagnosis']}\n"
                        f"Symptoms: {row['symptoms']}"
                    )
                    
                    # Add to image database
                    success = image_processor.add_image(
                        image_path=str(image_path),
                        problem_type=row['affected_part'],
                        description=problem_description,
                        solution=row['damage_type']
                    )
                    
                    if success:
                        print(f"Processed image {idx + 1}/{len(df)}: {image_path}")
                    else:
                        print(f"Failed to process image {idx + 1}/{len(df)}: {image_path}")
                else:
                    print(f"Image file not found: {image_path}")
                    
            except Exception as e:
                print(f"Error processing row {idx}: {str(e)}")
                continue
                
        print("Database initialization completed!")
        
    except Exception as e:
        logging.error(f"Error reading CSV file: {str(e)}")
        raise
