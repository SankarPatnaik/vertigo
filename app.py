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
from serpapi import GoogleSearch

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



    def get_google_results(self, query, max_results=3):
        """Get fallback context from Google if FAISS has no good matches"""
        api_key = os.getenv("SERPAPI_API_KEY")
        if not api_key:
            return "No Google context available (missing API key)."

        params = {
            "q": query,
            "num": max_results,
            "hl": "en",
            "api_key": api_key,
            "engine": "google"
        }

        try:
            search = GoogleSearch(params)
            results = search.get_dict()
            organic_results = results.get("organic_results", [])
            context = "\n\n".join([res.get("snippet", "") for res in organic_results if "snippet" in res])
            return context or "No relevant context found in Google results."
        except Exception as e:
            return f"Google Search error: {str(e)}"

    def call_groq_api(self, prompt: str) -> str:
        """Call Groq API directly"""
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }

        data = {
            "model": "llama-3.1-8b-instant",
            "messages": [
                {"role": "system",
                 "content": "You are an expert on constitutional law. Answer questions accurately based on the provided context."},
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


    def generate_answer(self, query, context, language="English", fallback=False):
        if fallback:
            context = self.get_google_results(query)

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

        return self.call_groq_api(prompt)


@st.cache_resource
def get_chatbot():
    return RAGChatbot(faiss_path="faiss_index")


def main():
    # Sidebar
    with st.sidebar:
        st.title("⚙️ Settings")
        language = st.selectbox(
            "🌐 Select Language",
            ["English", "Turkish"],
            key="language_select"
        )

        st.markdown("---")
        st.markdown("### 📘 About")
        st.markdown(
            "This chatbot uses **FAISS** for vector search and **Groq LLM** to answer queries about **Constitutional Law**.")
        st.markdown("Designed for legal professionals, students, and curious citizens.")

    # Page Title
    st.markdown(
        f"<h1 style='text-align: center; color: #4A90E2;'>⚖️ {'Constitutional Law Assistant' if language == 'English' else 'Anayasa Hukuku Asistanı'}</h1>",
        unsafe_allow_html=True
    )
    st.markdown(
        f"<p style='text-align: center;'> {'Ask anything about the Constitution below.' if language == 'English' else 'Anayasayla ilgili her şeyi sorabilirsiniz.'}</p>",
        unsafe_allow_html=True
    )

    chatbot = get_chatbot()

    if 'messages' not in st.session_state:
        st.session_state.messages = []
        st.markdown("👋 **Hello! Ask your first question about constitutional law.**")

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User input
    if prompt := st.chat_input(
            "💬 Type your question here..." if language == "English" else "💬 Sorunuzu buraya yazın..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="🧑‍⚖️"):
            st.markdown(prompt)

        with st.chat_message("assistant", avatar="🤖"):
            message_placeholder = st.empty()

            start_time = time.time()
            chunks, metadatas, scores = chatbot.get_relevant_chunks(prompt)
            context = "\n\n".join([f"[Score: {score:.2f}] {chunk}" for score, chunk in zip(scores, chunks)])

            with st.spinner("Generating a legal response..."):
                response = chatbot.generate_answer(prompt, context, language)
            inference_time = time.time() - start_time

            # Simulated typing animation
            typed_response = ""
            for char in response:
                typed_response += char
                time.sleep(0.005)
                message_placeholder.markdown(typed_response)

            with st.expander("📚 Source Information" if language == "English" else "📚 Kaynak Bilgisi"):
                st.markdown(f"⏱️ **Inference time:** {inference_time:.2f} seconds")
                for i, (meta, score) in enumerate(zip(metadatas, scores)):
                    st.markdown(
                        f"""
                        - **Source {i + 1}**  
                          📄 *{meta.get('source', 'N/A')}*  
                          🌍 *{meta.get('language', 'N/A')}*  
                          🧠 *Score:* `{score:.2f}`
                        """
                    )

        st.session_state.messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()
