import os
import argparse
from dotenv import load_dotenv
from smolagents import CodeAgent, HfApiModel, tool
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEndpointEmbeddings

# Global variable to hold the retriever to avoid reloading it on every tool call
retriever = None

@tool
def search_codebase(query: str) -> str:
    """
    Searches the codebase for the given query and returns relevant code snippets.
    
    Args:
        query: The search term, keyword, or question to find in the codebase.
    """
    global retriever
    if retriever is None:
        return "Error: Vector database not initialized."
    
    docs = retriever.invoke(query)
    if not docs:
        return "No relevant code snippets found."
    
    context = ""
    for i, doc in enumerate(docs):
        source = doc.metadata.get("source", "Unknown")
        context += f"--- Snippet {i+1} from {source} ---\n{doc.page_content}\n\n"
        
    return context

def init_agent(index_path: str = "faiss_index"):
    global retriever
    
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"FAISS index not found at {index_path}. Please run ingest.py first.")

    hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if not hf_token or hf_token == "your_token_here":
        raise ValueError("Please set a valid HUGGINGFACEHUB_API_TOKEN in the .env file.")

    print("Loading embedding model via Free API...")
    embeddings = HuggingFaceEndpointEmbeddings(
        model="sentence-transformers/all-MiniLM-L6-v2",
        task="feature-extraction",
        huggingfacehub_api_token=hf_token
    )

    print("Loading FAISS vector database...")
    vectorstore = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    print("Initializing smolagents Brain with Hugging Face API...")
    # Using Qwen2.5-Coder-32B-Instruct as it's excellent for coding and available via HF free API
    model = HfApiModel(
        model_id="Qwen/Qwen2.5-Coder-32B-Instruct",
        token=hf_token
    )

    # Use a CodeAgent that can write python or call tools
    agent = CodeAgent(
        tools=[search_codebase],
        model=model,
        additional_authorized_imports=["os", "json"],
        max_steps=5
    )
    
    return agent

def interactive_session(index_path: str = "faiss_index"):
    try:
        agent = init_agent(index_path)
    except Exception as e:
        print(f"Failed to initialize agent: {e}")
        return

    print("\n" + "="*50)
    print("🤖 Serverless Code Navigator Agent Initialized!")
    print("Powered by smolagents & Hugging Face Free API")
    print("Ask any question about your codebase.")
    print("Type 'exit' or 'quit' to stop.")
    print("="*50 + "\n")

    while True:
        query = input("You: ")
        if query.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break
        
        if not query.strip():
            continue

        print("\nAgent is thinking and searching...")
        try:
            # Let the agent run autonomously to answer the query
            response = agent.run(query)
            print("\n🤖 Final Answer:")
            print(response)
            print("-" * 50 + "\n")
        except Exception as e:
            print(f"\nError generating answer: {e}\n")

if __name__ == "__main__":
    load_dotenv()
    parser = argparse.ArgumentParser(description="Query the ingested code repository.")
    parser.add_argument("--index", type=str, default="faiss_index", help="Path to the saved FAISS index.")
    
    args = parser.parse_args()
    interactive_session(args.index)
