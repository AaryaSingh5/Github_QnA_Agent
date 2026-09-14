import os
import argparse
import subprocess
import requests
from dotenv import load_dotenv
from smolagents import CodeAgent, HfApiModel, tool
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from ingest import ingest_repository

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

    print("Loading embedding model (sentence-transformers/all-MiniLM-L6-v2) locally...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

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

def prepare_repos(repo_urls: list) -> tuple[str, str]:
    """Prepares multiple repositories by cloning and returning the local base path and index path."""
    base_dir = "repos/multi_repo"
    os.makedirs(base_dir, exist_ok=True)
    
    for repo_path_or_url in repo_urls:
        if repo_path_or_url.startswith(("http://", "https://", "git@")):
            repo_name = repo_path_or_url.rstrip("/").split("/")[-1].replace(".git", "")
            local_repo_path = os.path.join(base_dir, repo_name)
            if not os.path.exists(local_repo_path):
                print(f"Cloning {repo_path_or_url} into {local_repo_path}...")
                subprocess.run(["git", "clone", repo_path_or_url, local_repo_path], check=True)
            else:
                print(f"Repository already cloned at {local_repo_path}")
        else:
            repo_name = os.path.basename(os.path.abspath(repo_path_or_url))
            local_repo_path = os.path.join(base_dir, repo_name)
            if not os.path.exists(local_repo_path) and os.path.exists(repo_path_or_url):
                print(f"Using local path: {repo_path_or_url}")
                local_repo_path = repo_path_or_url
                
    index_path = "multi_repo_faiss_index"
    if len(repo_urls) == 1:
        repo_name = repo_urls[0].rstrip("/").split("/")[-1].replace(".git", "")
        index_path = f"{repo_name}_faiss_index"
        
    if not os.path.exists(index_path):
        print(f"Index '{index_path}' not found. Running ingestion for {base_dir}...")
        ingest_repository(base_dir, save_path=index_path)
        
    return base_dir, index_path

def prepare_github_user(username: str) -> tuple[str, str]:
    """Fetches, clones, and prepares an index for all repositories of a GitHub user."""
    token = os.getenv("GITHUB_TOKEN")
    headers = {"Accept": "application/vnd.github.v3+json"}
    if token and token != "your_token_here":
        headers["Authorization"] = f"token {token}"
        
    print(f"Fetching repositories for user/org {username}...")
    repos = []
    page = 1
    while True:
        url = f"https://api.github.com/users/{username}/repos?per_page=100&page={page}"
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            print(f"Error fetching repos: {response.status_code} - {response.text}")
            break
        data = response.json()
        if not data:
            break
        repos.extend(data)
        page += 1
        
    if not repos:
        raise ValueError(f"No repositories found for {username}")
        
    print(f"Found {len(repos)} repositories.")
    for i, repo in enumerate(repos):
        print(f"[{i+1}] {repo['name']}")
    
    print("\nEnter the numbers of the repositories you want to index (comma-separated), or press Enter for all:")
    selection = input("> ").strip()
    if selection:
        try:
            indices = [int(x.strip()) - 1 for x in selection.split(",")]
            repos = [repos[i] for i in indices if 0 <= i < len(repos)]
            print(f"Selected {len(repos)} repositories.")
        except Exception as e:
            print("Invalid selection, proceeding with all repositories.")
            
    base_dir = os.path.join("repos", username)
    os.makedirs(base_dir, exist_ok=True)
    
    for repo in repos:
        repo_name = repo["name"]
        clone_url = repo["clone_url"]
        local_repo_path = os.path.join(base_dir, repo_name)
        
        if not os.path.exists(local_repo_path):
            print(f"Cloning {repo_name}...")
            subprocess.run(["git", "clone", clone_url, local_repo_path], check=False)
        else:
            print(f"Repository {repo_name} already cloned.")
            
    index_path = f"{username}_faiss_index"
    if not os.path.exists(index_path):
        print(f"Index '{index_path}' not found. Running unified ingestion for all repositories of {username}...")
        ingest_repository(base_dir, save_path=index_path)
        
    return base_dir, index_path

def interactive_session(index_path: str = "faiss_index", repos: list = None, github_user: str = None):
    if github_user:
        try:
            _, index_path = prepare_github_user(github_user)
        except Exception as e:
            print(f"Failed to prepare github user repositories: {e}")
            return
    elif repos:
        try:
            _, index_path = prepare_repos(repos)
        except Exception as e:
            print(f"Failed to prepare repositories: {e}")
            return

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
    parser.add_argument("--repos", nargs="+", type=str, help="One or more paths or URLs to the repositories to analyze.")
    parser.add_argument("--github-user", type=str, help="GitHub username to ingest all repositories for.")
    
    args = parser.parse_args()
    interactive_session(args.index, args.repos, args.github_user)
