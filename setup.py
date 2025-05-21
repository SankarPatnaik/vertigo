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

    # Generic folder path
    pdf_folder_path = r"C:\Users\patna\OneDrive\Desktop\RAG_DLM\vertigo\Data"

    # Collect all PDF file paths from the folder

    pdf_paths = [os.path.join(pdf_folder_path, f) for f in os.listdir(pdf_folder_path) if f.lower().endswith('.pdf')]
    documents = creator.load_pdfs(pdf_paths)
    chunks = creator.split_documents(documents)
    creator.create_faiss_index(chunks, save_path="faiss_index")
    
    print("Setup complete! You can now run: streamlit run app.py")

if __name__ == "__main__":
    main()