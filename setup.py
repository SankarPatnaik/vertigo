import os
import subprocess
import sys

def main():
    print("Setting up RAG Chatbot...")
    
    # Install requirements
    print("Installing requirements...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    
    # Create vector database
    print("Creating FAISS vector database...")
    import vector_db_creation
    creator = vector_db_creation.FAISSVectorDBCreator()
    
    pdf_paths = [
        r"C:\Users\patna\OneDrive\Desktop\RAG_DLM\vertigo\Data\circular.pdf",
        r"C:\Users\patna\OneDrive\Desktop\RAG_DLM\vertigo\Data\civil_code_1908.pdf",
        r"C:\Users\patna\OneDrive\Desktop\RAG_DLM\vertigo\Data\Anayasa_English.pdf",
        r"C:\Users\patna\OneDrive\Desktop\RAG_DLM\vertigo\Data\Anayasa-Turkish.pdf",
        r"C:\Users\patna\OneDrive\Desktop\RAG_DLM\vertigo\Data\At_A_Glance-ENG.pdf",
        r"C:\Users\patna\OneDrive\Desktop\RAG_DLM\vertigo\Data\Ebook-one-year-modi-2.0.pdf"

    ]
    
    documents = creator.load_pdfs(pdf_paths)
    chunks = creator.split_documents(documents)
    creator.create_faiss_index(chunks, save_path="faiss_index")
    
    print("Setup complete! You can now run: streamlit run app.py")

if __name__ == "__main__":
    main()