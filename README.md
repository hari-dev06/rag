# Enterprise Document Q&A (RAG Pipeline)

I built this Retrieval-Augmented Generation (RAG) pipeline to solve a specific problem: querying enterprise PDF repositories reliably on a zero-dollar budget.

It uses an entirely free-tier tech stack. To compensate for the latency and strict rate limits of free LLM APIs, the backend is built heavily around asynchronous execution, parallelized retrieval strategies, and strict in-memory caching.

Live Link: 

## Tech Stack

* **Backend:** FastAPI (fully async `httpx` architecture)
* **Orchestration:** LangChain
* **LLMs:** OpenRouter (Gemma-4-26b for chat, Llama-Nemotron-1b for embeddings)
* **Vector Store:** FAISS (Isolated per-document indices)
* **Frontend:** Streamlit

## Engineering Highlights

### Parallel Dual-Retrieval

Standard semantic search often fails on complex queries. To fix this, the system executes Query Rewriting and HyDE (Hypothetical Document Embeddings) simultaneously using `asyncio.gather`. The backend merges the results from both strategies and mathematically deduplicates the text chunks before sending them to the context window.

### Smart Routing & Cache Optimization

When the system holds more than 5 documents, sweeping the entire vector database becomes inefficient. Instead, an LLM router reads isolated summary files (`meta.json`) generated during ingestion to determine which specific FAISS index to target. The pipeline caches FAISS indices and metadata in memory after the initial query, eliminating redundant disk I/O.

### Context Compression & Fallback Logic

* **Auto-Summarization:** The `InMemoryChatMessageHistory` is programmed to compress the conversation context exactly at 10 messages. This prevents context-window bloat and maintains fast API response times.
* **Web Search Fallback:** If the local documents cannot answer a factual query, the system intercepts the failure and triggers a live web crawl using the `ddgs` (DuckDuckGo) library to construct an answer. Opinion-based queries bypass this fallback.

## Repository Structure

```text
rag-app/
├── backend/
│   ├── ingest.py         # PDF chunking (2000 chars/400 overlap) and FAISS indexing
│   ├── rag_chain.py      # Core orchestration (Rewrite, HyDE, Async Embeddings, Fallback)
│   ├── main.py           # FastAPI application and endpoint routers
│   └── vector_store/     # Local binary index data partitioned per file
└── frontend/
    └── app.py            # Streamlit dashboard and UI state management

```

## Local Development Setup

**1. Environment setup**
Ensure you have Python 3.10+ installed. Clone the repository and configure your virtual environment:

```bash
git clone <repository-url>
cd rag-app
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

```

Create a `.env` file in the root directory and add your free-tier API key:

```env
OPENROUTER_API_KEY=your_api_key_here

```

**2. Start the Backend**
The FastAPI server handles all heavy lifting, including ingestion and vector math.

```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload

```

**3. Start the Frontend**
Open a new terminal window, activate your environment, and start the UI:

```bash
streamlit run frontend/app.py

```

## API Endpoints

If you want to bypass the frontend entirely, the API exposes the following routes:

* `POST /ingest` - Accepts a PDF, runs chunking, generates abstractive summaries, and builds the FAISS index.
* `POST /chat` - Accepts a query string and `session_id`, returning the LLM response and source citations.
* `GET /documents` - Returns all ingested files and their computed metadata.
* `GET /health` - Returns current memory states and active session counts.

## Future Roadmap

While currently optimized for local execution on a free tier, the architecture is designed to scale. Next steps for production deployment include:

1. Swapping `InMemoryChatMessageHistory` for a persistent Redis cluster.
2. Migrating from local FAISS files to a managed vector database like Pinecone or Qdrant.
3. Wrapping the backend and frontend into a multi-stage Docker Compose network.
