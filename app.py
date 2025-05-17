import streamlit as st
import os
from dotenv import load_dotenv
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import requests
import json
from functools import lru_cache
import time

# Load environment variables
load_dotenv()

# Page config
st.set_page_config(
    page_title="Constitutional Law Assistant",
    page_icon="⚖️",
    layout="wide"
)

# Initialize Groq API
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "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")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

class RAGChatbot:
    def __init__(self, faiss_path="faiss_index"):
        self.embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.index = faiss.read_index(os.path.join(faiss_path, "index.faiss"))
        with open(os.path.join(faiss_path, "metadata.pkl"), "rb") as f:
            self.metadata = pickle.load(f)
    
    @lru_cache(maxsize=100)
    def get_query_embedding(self, query: str):
        """Cache query embeddings for faster repeated searches"""
        query_embedding = self.embedder.encode([query])[0]
        faiss.normalize_L2(query_embedding.reshape(1, -1).astype('float32'))
        return query_embedding
    
    def get_relevant_chunks(self, query, k=3):
        # Get embedding (cached)
        query_embedding = self.get_query_embedding(query)
        
        # Search
        scores, indices = self.index.search(query_embedding.reshape(1, -1), k)
        chunks = [self.metadata['texts'][i] for i in indices[0]]
        metadatas = [self.metadata['metadatas'][i] for i in indices[0]]
        
        return chunks, metadatas, scores[0]
    
    def call_groq_api(self, prompt: str) -> str:
        """Call Groq API directly"""
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": "llama-3.1-8b-instant",
            "messages": [
                {"role": "system", "content": "You are an expert on constitutional law. Answer questions accurately based on the provided context."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.7,
            "max_tokens": 1000
        }
        
        try:
            response = requests.post(GROQ_API_URL, headers=headers, json=data)
            response.raise_for_status()
            result = response.json()
            return result['choices'][0]['message']['content']
        except Exception as e:
            return f"Error calling Groq API: {str(e)}"
    
    def generate_answer(self, query, context, language="English"):
        # Prepare the prompt
        prompt = f"""You are an expert on constitutional law. Answer questions precisely based on the provided context.
        Language: {language}
        
        Context:
        {context}
        
        Question: {query}
        
        Instructions:
        - Answer ONLY based on the provided context
        - Be concise but complete
        - If the context doesn't contain enough information, say so
        - Provide article references when available
        """
        
        # Call Groq API directly
        response = self.call_groq_api(prompt)
        return response

@st.cache_resource
def get_chatbot():
    return RAGChatbot(faiss_path="faiss_index")

def main():
    # Sidebar
    with st.sidebar:
        st.title("Settings")
        language = st.selectbox(
            "Select Language",
            ["English", "Turkish"],
            key="language_select"
        )
        
        st.markdown("---")
        st.markdown("### About")
        st.markdown("This application uses FAISS vector search and Groq LLM to provide answers about the Constitution.")
    
    # Main content
    title = "⚖️ Constitutional Law Assistant" if language == "English" else "⚖️ Anayasa Hukuku Asistanı"
    st.title(title)
    
    st.markdown(
        "Ask questions about the Constitution" if language == "English" 
        else "Anayasa hakkında sorular sorun"
    )
    
    # Initialize chatbot
    chatbot = get_chatbot()
    
    # Chat interface
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Type your question..." if language == "English" else "Sorunuzu yazın..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            
            # Measure inference time
            start_time = time.time()
            
            # Get relevant chunks
            chunks, metadatas, scores = chatbot.get_relevant_chunks(prompt)
            
            # Create context from relevant chunks
            context = "\n\n".join([f"[Score: {score:.2f}] {chunk}" for score, chunk in zip(scores, chunks)])
            
            # Generate answer
            response = chatbot.generate_answer(prompt, context, language)
            inference_time = time.time() - start_time
            
            # Display response
            message_placeholder.markdown(response)
            
            # Show metadata in expander
            with st.expander("Source Information" if language == "English" else "Kaynak Bilgisi"):
                st.markdown(f"Inference time: {inference_time:.2f} seconds")
                for i, (meta, score) in enumerate(zip(metadatas, scores)):
                    st.markdown(f"**Source {i+1}:** {meta['source']} ({meta['language']}) - Score: {score:.2f}")
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()