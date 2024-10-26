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
import json

@dataclass
class ChatConfig:
    """Configuration for the chatbot"""
    embedding_model_name: str = 'all-MiniLM-L6-v2'
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    max_history: int = 3
    gemini_api_key: str = "AIzaSyBS_DFCJh82voYIKoglS-ow6ezGNg775pg"  # Replace with your API key
    log_file: str = "chat_history.txt"
    user_data_file: str = "user_data.json"

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
        
    async def generate_questions(self, question: str, answer: str) -> List[str]:
        try:
            chat = self.model.start_chat(history=[])
            prompt = f"""Based on this product information interaction:
            
            Question: {question}
            Answer: {answer}
            
            Generate 4 relevant follow-up questions that a customer might ask about GAPL Starter.
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
            
            while len(questions) < 4:
                questions.append("Can you provide more details about GAPL Starter?")
            
            return questions[:4]
            
        except Exception as e:
            logging.error(f"Error generating questions: {str(e)}")
            return [
                "How should I store GAPL Starter?",
                "Can I use it with other fertilizers?",
                "What results can I expect to see?",
                "Is it safe for all soil types?"
            ]

class GeminiRAG:
    """RAG implementation using Gemini"""
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.generation_config = {
            "temperature": 0.1,  # Slightly reduced for more consistent tone
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
        
    async def get_answer(self, question: str, context: str) -> str:
        try:
            chat = self.model.start_chat(history=[])
            prompt = f"""You are an expert agricultural consultant specializing in GAPL Starter bio-fertilizer. 
            You have extensive hands-on experience with the product and deep knowledge of its applications and benefits.
            
            Background information to inform your response:
            {context}

            Question from farmer: {question}

            Respond naturally as an expert would, without referencing any "provided information" or documentation.
            Your response should be:
            - Confident and authoritative
            - Direct and practical
            - Focused on helping farmers succeed
            - Based on product expertise

            If you don't have enough specific information to answer the question, say something like:
            "As an expert on GAPL Starter, I should note that while the product has broad applications, 
            I'd need to check the specific details about [missing information] to give you the most accurate guidance. 
            What I can tell you is..."

            Remember to maintain a helpful, expert tone throughout your response.
            """
            
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

def load_user_data(user_data_file: str) -> Dict[str, Any]:
    """Loads user data from a JSON file"""
    try:
        with open(user_data_file, 'r') as f:
            user_data = json.load(f)
        return user_data
    except FileNotFoundError:
        return {}

def save_user_data(user_data: Dict[str, Any], user_data_file: str):
    """Saves user data to a JSON file"""
    with open(user_data_file, 'w') as f:
        json.dump(user_data, f, indent=4)

def extract_user_info(question: str, user_data: Dict[str, Any]) -> Optional[Dict[str, str]]:
    """Extracts user information from the question"""
    extracted_info = {}
    
    if "mobile" in question.lower() or "phone" in question.lower():
        if "mobile" in question.lower():
            mobile_number = question.split("mobile")[1].strip()
        else:
            mobile_number = question.split("phone")[1].strip()
        if mobile_number.isdigit() and len(mobile_number) == 10:
            extracted_info["mobile_number"] = mobile_number
            user_data["mobile_number"] = mobile_number
    
    if "location" in question.lower() or "pincode" in question.lower():
        if "location" in question.lower():
            location = question.split("location")[1].strip()
        else:
            location = question.split("pincode")[1].strip()
        extracted_info["location"] = location
        user_data["location"] = location
    
    if "purchase" in question.lower() or "bought" in question.lower():
        if "purchase" in question.lower():
            purchase_status = question.split("purchase")[1].strip()
        else:
            purchase_status = question.split("bought")[1].strip()
        if purchase_status.lower() in ["yes", "no", "planning to purchase"]:
            extracted_info["purchase_status"] = purchase_status
            user_data["purchase_status"] = purchase_status
    
    if "crop" in question.lower():
        crop_name = question.split("crop")[1].strip()
        extracted_info["crop_name"] = crop_name
        user_data["crop_name"] = crop_name
    
    if "name" in question.lower():
        name = question.split("name")[1].strip()
        extracted_info["name"] = name
        user_data["name"] = name
    
    if extracted_info:
        save_user_data(user_data, config.user_data_file)
        return extracted_info
    else:
        return None

class ChatBot:
    """Main chatbot class"""
    def __init__(self, config: ChatConfig):
        self.config = config
        self.logger = ChatLogger(config.log_file)
        self.question_gen = QuestionGenerator(config.gemini_api_key)
        self.rag = GeminiRAG(config.gemini_api_key)
        self.db = ProductDatabase(config)
        self.chat_memory = ChatMemory(config.max_history)
        self.user_data = load_user_data(config.user_data_file)
        
    def load_database(self):
        """Loads the product database"""
        with open("STARTER.md", "r", encoding="utf-8") as f:
            markdown_content = f.read()
        self.db.process_markdown(markdown_content)
        
    async def process_question(self, question: str) -> str:
        """Processes the user's question"""
        try:
            extracted_info = extract_user_info(question, self.user_data)
            
            relevant_docs = self.db.search(question)
            context = self.rag.create_context(relevant_docs)
            answer = await self.rag.get_answer(question, context)
            follow_up_questions = await self.question_gen.generate_questions(question, answer)
            
            self.chat_memory.add_interaction(question, answer)
            self.logger.log_interaction(question, answer)
            
            # Add user info to the answer if available
            if extracted_info:
                answer += f"\n\nBased on your information, I can provide more tailored advice. "
                for key, value in extracted_info.items():
                    answer += f"You mentioned your {key} is {value}. "
            
            return answer, follow_up_questions
            
        except Exception as e:
            logging.error(f"Error processing question: {str(e)}")
            return "I apologize, but I'm having trouble processing your request. Please try again.", []

    def get_initial_questions(self) -> List[str]:
        """Returns a list of initial questions"""
        return [
            "What are the main benefits of GAPL Starter?",
            "How do I apply GAPL Starter correctly?",
            "Which crops is GAPL Starter suitable for?",
            "What is the recommended dosage?"
        ]

    def get_user_data(self) -> Dict[str, Any]:
        """Returns the user data"""
        return self.user_data

    def clear_chat_history(self):
        """Clears the chat history"""
        self.chat_memory.clear_history()

if __name__ == "__main__":
    config = ChatConfig()
    chatbot = ChatBot(config)
    chatbot.load_database()
    
    # Example usage
    while True:
        question = input("You: ")
        if question.lower() == "exit":
            break
        
        answer, follow_up_questions = asyncio.run(chatbot.process_question(question))
        print(f"Bot: {answer}")
        
        if follow_up_questions:
            print("Follow-up questions:")
            for i, q in enumerate(follow_up_questions):
                print(f"{i+1}. {q}")
