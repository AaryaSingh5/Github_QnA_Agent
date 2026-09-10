import os
import argparse
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_community.vectorstores import FAISS

def ingest_repository(repo_path: str, save_path: str = "faiss_index"):
    """
    Ingests a code repository, splits the files into chunks,
    generates embeddings, and saves them to a local FAISS index.
    """
    print(f"Loading repository from: {repo_path}")
    
    loader = DirectoryLoader(
        repo_path,
        glob="**/*.*",
        exclude=["**/node_modules/**", "**/venv/**", "**/.git/**", "**/__pycache__/**", "**/faiss_index/**", "**/serverless_code_navigator/**"],
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"}
    )
    
    print("Parsing files into documents...")
    try:
        documents = loader.load()
    except Exception as e:
        print(f"Error loading documents: {e}")
        return
        
    print(f"Loaded {len(documents)} document(s).")

    if len(documents) == 0:
        print("No valid source code files found in the directory.")
        return

    print("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} document chunks.")

    hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if not hf_token or hf_token == "your_token_here":
        raise ValueError("Please set a valid HUGGINGFACEHUB_API_TOKEN in the .env file.")

    print("Loading embedding model via Free API (sentence-transformers/all-MiniLM-L6-v2)...")
    embeddings = HuggingFaceEndpointEmbeddings(
        model="sentence-transformers/all-MiniLM-L6-v2",
        task="feature-extraction",
        huggingfacehub_api_token=hf_token
    )

    print("Generating embeddings and building FAISS vector database...")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    vectorstore.save_local(save_path)
    print(f"Successfully saved FAISS index to {save_path}")

if __name__ == "__main__":
    load_dotenv()
    parser = argparse.ArgumentParser(description="Ingest a code repository for QA.")
    # By default, ingest the parent directory, since serverless_code_navigator is a subfolder
    parser.add_argument("--repo", type=str, default="..", help="Path to the repository to ingest.")
    parser.add_argument("--index", type=str, default="faiss_index", help="Path to save the FAISS index.")
    
    args = parser.parse_args()
    
    if not os.path.isdir(args.repo):
        print(f"Error: The path '{args.repo}' does not exist or is not a directory.")
    else:
        ingest_repository(args.repo, args.index)
