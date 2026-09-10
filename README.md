# Serverless Code Navigator Agent

The **Serverless Code Navigator Agent** is an autonomous AI agent capable of reading and answering questions about your local codebase. It shifts all heavy computing to the cloud, operating with a **Zero Model Download** and **Zero Local GPU** architecture by leveraging Hugging Face's free Serverless APIs.

This tool is powered by [`smolagents`](https://github.com/huggingface/smolagents) to give the AI the autonomy to call its own tools, retrieve local code context, and reason about it before responding.

---

## 🏗️ How It Works

The architecture consists of four main components interacting seamlessly:

1. **The Ingestor**: Uses LangChain Document Loaders to scan your local repository and chop code files into readable 1,000-character chunks.
2. **The Indexer**: Converts those code chunks into embeddings using a free Hugging Face feature-extraction API (`sentence-transformers/all-MiniLM-L6-v2`) and temporarily stores them in a local FAISS database (acting as "RAM").
3. **The Tool (`search_codebase`)**: A custom Python function given to the AI. It takes a keyword, searches the FAISS memory, and returns relevant code snippets.
4. **The Brain**: Powered by `smolagents.CodeAgent` and a free cloud LLM (like `Qwen/Qwen2.5-Coder-32B-Instruct`). The agent is given access to your codebase tool. If it doesn't find the answer on the first search, it will automatically adjust its query and search again until it gets it right.

---

## 🚀 Setup & Installation

### Prerequisites
- Python 3.8+
- A [Hugging Face API Token](https://huggingface.co/settings/tokens) (Free tier works perfectly!)

### 1. Install Dependencies
Clone this repository and install the required packages:

```bash
pip install -r requirements.txt
```

### 2. Add Your Hugging Face Token
Open the `.env` file and paste your Hugging Face API token:

```env
HUGGINGFACEHUB_API_TOKEN=hf_your_token_here
```

---

## 💻 Usage

### Step 1: Ingest Your Repository
First, you need to parse and embed your local codebase. By default, this script targets the parent directory of where it runs, but you can pass any directory.

```bash
python ingest.py --repo /path/to/your/codebase
```
*This will create a local `faiss_index` folder containing the embedded chunks.*

### Step 2: Start the AI Agent
Run the interactive agent loop. It will load the FAISS database and initialize the Hugging Face Serverless model.

```bash
python agent.py
```

### Example Interaction:
```text
🤖 Serverless Code Navigator Agent Initialized!
Powered by smolagents & Hugging Face Free API
Ask any question about your codebase.
Type 'exit' or 'quit' to stop.
==================================================

You: How does the embedding generation work?

Agent is thinking and searching...
[Agent autonomously uses search_codebase("embedding generation")]

🤖 Final Answer:
The embedding generation is handled in `ingest.py`. It uses `HuggingFaceEndpointEmbeddings` with the `sentence-transformers/all-MiniLM-L6-v2` model via the Hugging Face Free API. The embeddings are then stored locally in a FAISS vector database.
```

---

## 🛠️ Built With
- [smolagents](https://github.com/huggingface/smolagents) - Autonomous AI Tool Calling
- [LangChain](https://github.com/langchain-ai/langchain) - Code Splitting & Vector Store Loaders
- [FAISS](https://github.com/facebookresearch/faiss) - In-Memory Vector Search
- [Hugging Face](https://huggingface.co/) - Free Serverless Inference & Embeddings
