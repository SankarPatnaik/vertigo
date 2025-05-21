import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
from typing import List, Dict

# Load environment variables
load_dotenv()

class FAISSVectorDBCreator:
    def __init__(self, embedding_model="sentence-transformers/all-MiniLM-L6-v2"):
        self.embedder = SentenceTransformer(embedding_model)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,  # Increased for better context
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )
        
    def load_pdfs(self, pdf_paths):
        all_documents = []
        for pdf_path in pdf_paths:
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
            # Add source metadata
            for doc in documents:
                doc.metadata['source'] = os.path.basename(pdf_path)
                doc.metadata['language'] = 'Turkish' if 'Turkish' in pdf_path else 'English'
            all_documents.extend(documents)
        return all_documents
    
    def split_documents(self, documents):
        return self.text_splitter.split_documents(documents)
    
    def create_faiss_index(self, chunks, save_path="faiss_index"):
        # Create directory if it doesn't exist
        os.makedirs(save_path, exist_ok=True)
        
        # Extract texts and metadata
        texts = [chunk.page_content for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]
        
        # Create embeddings (batch processing for better performance)
        embeddings = self.embedder.encode(texts, batch_size=32, show_progress_bar=True)
        
        # Create FAISS index (using IndexFlatIP for production)
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)  # Inner product for normalized vectors
        
        # Normalize vectors for cosine similarity
        faiss.normalize_L2(embeddings)
        index.add(embeddings.astype('float32'))
        
        # Save index and metadata
        faiss.write_index(index, os.path.join(save_path, "index.faiss"))
        with open(os.path.join(save_path, "metadata.pkl"), "wb") as f:
            pickle.dump({'texts': texts, 'metadatas': metadatas}, f)
        
        print(f"FAISS index created with {len(chunks)} chunks in '{save_path}'")
        return index


if __name__ == "__main__":
    # Initialize the creator
    creator = FAISSVectorDBCreator()

    # Define the folder containing PDFs
    pdf_folder_path = r"C:\Users\patna\OneDrive\Desktop\RAG_DLM\vertigo\Data"

    # Get all PDF files in the folder
    pdf_paths = [os.path.join(pdf_folder_path, f) for f in os.listdir(pdf_folder_path) if f.lower().endswith('.pdf')]

    # Load PDFs
    documents = creator.load_pdfs(pdf_paths)
    print(f"Loaded {len(documents)} pages from PDFs")

    # Split documents into chunks
    chunks = creator.split_documents(documents)
    print(f"Split into {len(chunks)} chunks")

    # Create FAISS index
    creator.create_faiss_index(chunks, save_path="faiss_index")