import os
import time
import json
import asyncio
import httpx
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain.embeddings.base import Embeddings
from ddgs import DDGS



load_dotenv()

VECTOR_STORE_ROOT = Path("vector_store")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
CHAT_MODEL = "openrouter/free"
SUMMARIZE_AFTER = 10
ROUTING_THRESHOLD = 5

sessions: dict[str, dict] = {}
_metadata_cache: list[dict] = []
_index_cache: dict[str, FAISS] = {}


# Embeddings 

class OpenRouterEmbeddings(Embeddings):
    def __init__(self, model: str, api_key: str):
        self.model = model
        self.api_key = api_key

    def _embed(self, texts: list[str]) -> list[list[float]]:
        with httpx.Client() as client:
            response = client.post(
                "https://openrouter.ai/api/v1/embeddings",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={"model": self.model, "input": texts},
                timeout=30
            )
            response.raise_for_status()
            return [d["embedding"] for d in response.json()["data"]]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text])[0]


embeddings = OpenRouterEmbeddings(
    model="nvidia/llama-nemotron-embed-vl-1b-v2:free",
    api_key=OPENROUTER_API_KEY
)


# Session management

def get_session(session_id: str) -> dict:
    if session_id not in sessions:
        sessions[session_id] = {
            "history": InMemoryChatMessageHistory(),
            "summary": ""
        }
    return sessions[session_id]


# Async LLM call 

async def llm_call(messages: list[dict], retries: int = 3) -> str:
    async with httpx.AsyncClient() as client:
        for attempt in range(retries):
            response = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={"model": CHAT_MODEL, "messages": messages},
                timeout=60
            )
            if response.status_code == 429:
                wait = 2 ** attempt
                print(f"Rate limited. Waiting {wait}s...")
                await asyncio.sleep(wait)
                continue
            if not response.is_success:
                print(f"ERROR {response.status_code}: {response.text}")
                response.raise_for_status()
            return response.json()["choices"][0]["message"].get("content") or ""
    return ""


# Summarization memory 

async def maybe_summarize(session: dict) -> None:
    messages = session["history"].messages
    if len(messages) < SUMMARIZE_AFTER:
        return

    convo = "\n".join([
        f"{'User' if m.type == 'human' else 'AI'}: {m.content}"
        for m in messages
    ])

    new_summary = await llm_call([{
        "role": "user",
        "content": f"Summarize this conversation concisely, preserving key facts and context:\n\n{convo}"
    }])

    session["summary"] = new_summary
    session["history"] = InMemoryChatMessageHistory()
    print("[Memory summarized]")


# Metadata + index cache 

def load_all_metadata() -> list[dict]:
    global _metadata_cache
    if _metadata_cache:
        return _metadata_cache
    metas = []
    for folder in VECTOR_STORE_ROOT.iterdir():
        meta_file = folder / "meta.json"
        if meta_file.exists():
            with open(meta_file) as f:
                metas.append(json.load(f))
    _metadata_cache = metas
    return metas


def get_index(filename: str) -> FAISS | None:
    if filename not in _index_cache:
        index_path = str(VECTOR_STORE_ROOT / filename)
        if not (VECTOR_STORE_ROOT / filename / "index.faiss").exists():
            return None
        _index_cache[filename] = FAISS.load_local(
            index_path, embeddings, allow_dangerous_deserialization=True
        )
    return _index_cache[filename]


# Query rewriting

async def rewrite_query(query: str) -> str:
    rewritten = await llm_call([{
        "role": "user",
        "content": f"""Rewrite this question into a clear, specific search query optimized for document retrieval.
Return ONLY the rewritten query, nothing else.

Question: {query}"""
    }])
    result = rewritten.strip()
    print(f"[Rewritten query]: {result}")
    return result if result else query
    


# HyDE 

async def generate_hypothetical_answer(query: str) -> str:
    hypothesis = await llm_call([{
        "role": "user",
        "content": f"""Write a hypothetical answer to this question as if you found it in a document.
Be specific and detailed. Return ONLY the hypothetical answer.

Question: {query}"""
    }])
    print(f"[HyDE hypothesis]: {hypothesis[:100]}...")
    return hypothesis.strip() or query


# Retrieval 

def retrieve_chunks(query: str, filenames: list[str], k: int = 10) -> list:
    import concurrent.futures

    def search_index(filename):
        store = get_index(filename)
        if not store:
            return []
        return store.similarity_search(query, k=k)

    all_chunks = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = executor.map(search_index, filenames)
        for chunks in results:
            all_chunks.extend(chunks)

    # Deduplicate
    seen = set()
    unique = []
    for c in all_chunks:
        key = c.page_content[:100]
        if key not in seen:
            seen.add(key)
            unique.append(c)
    return unique


# Web search fallback

async def web_search_fallback(query: str) -> str:
    print("[Falling back to web search]")
    try:
        # The new standard way to call it
        results = DDGS().text(query, max_results=5)
        
        if not results:
            return "No results found on web either."

        context = "\n\n".join([
            f"[{r.get('title', 'No Title')}] {r.get('body', '')}"
            for r in results
        ])

        answer = await llm_call([{
            "role": "user",
            "content": f"""Answer this question using the web search results below.
Cite sources by title.

Question: {query}

Web results:
{context}"""
        }])
        return f"[Web search result]\n{answer}"
    except Exception as e:
        return f"Web search failed: {e}"

# LLM routing 

async def route_query_llm(query: str, metas: list[dict]) -> list[str]:
    doc_list = "\n".join([
        f"{i+1}. {m['filename']}: {m['summary']}"
        for i, m in enumerate(metas)
    ])

    answer = await llm_call([{
        "role": "user",
        "content": f"""Given this query: "{query}"

Which documents are most relevant? Reply ONLY with comma-separated numbers.

Documents:
{doc_list}"""
    }])

    selected = []
    for part in answer.replace(" ", "").split(","):
        try:
            idx = int(part.strip()) - 1
            if 0 <= idx < len(metas):
                selected.append(metas[idx]["filename"])
        except ValueError:
            continue

    return selected if selected else [m["filename"] for m in metas]


# Prompt builder 

def build_prompt(query: str, chunks: list, session: dict) -> list:
    if chunks:
        context = "\n\n".join([
            f"[Source: {c.metadata.get('source', 'unknown')} | Page {c.metadata.get('page', '?')}]\n{c.page_content}"
            for c in chunks
        ])
    else:
        context = "No relevant documents found."

    system = f"""You are a document Q&A assistant helping a student prepare for exams.
Answer using the context below. Always cite sources as [Source | Page X].
List ALL items found — do not summarize or truncate lists.
If asked for opinions or recommendations, use the document content to reason and give a helpful answer.
Only respond with "NOT_FOUND_IN_DOCS" if the topic is completely absent from the context.

CONTEXT:
{context}"""

    messages = [{"role": "system", "content": system}]

    if session["summary"]:
        messages.append({
            "role": "system",
            "content": f"Previous conversation summary:\n{session['summary']}"
        })

    for msg in session["history"].messages:
        if msg.type == "human":
            messages.append({"role": "user", "content": msg.content})
        elif msg.type == "ai":
            messages.append({"role": "assistant", "content": msg.content})

    messages.append({"role": "user", "content": query})
    return messages


# Main chat

async def chat(query: str, session_id: str = "default") -> dict:
    session = get_session(session_id)
    metas = load_all_metadata()

    # Step 1: rewrite + HyDE in parallel
    rewritten, hypothesis = await asyncio.gather(
        rewrite_query(query),
        generate_hypothetical_answer(query)
    )

    # Step 2: route
    if len(metas) > ROUTING_THRESHOLD:
        print(f"[Routing via LLM — {len(metas)} docs]")
        selected_files = await route_query_llm(rewritten, metas)
    else:
        print(f"[Searching all {len(metas)} docs directly]")
        selected_files = [m["filename"] for m in metas]

    # Step 3: retrieve using BOTH rewritten query and HyDE, merge results
    chunks_rewritten = retrieve_chunks(rewritten, selected_files, k=10)
    chunks_hyde = retrieve_chunks(hypothesis, selected_files, k=6)

    # Merge + deduplicate
    seen = set()
    chunks = []
    for c in chunks_rewritten + chunks_hyde:
        key = c.page_content[:100]
        if key not in seen:
            seen.add(key)
            chunks.append(c)

    # Step 4: answer
    messages = build_prompt(query, chunks, session)
    answer = await llm_call(messages)

    # Step 5: web fallback only for factual queries
    opinion_keywords = ["important", "rank", "best", "recommend", "suggest", "which", "opinion"]
    is_opinion = any(kw in query.lower() for kw in opinion_keywords)

    if "NOT_FOUND_IN_DOCS" in answer and not is_opinion:
        answer = await web_search_fallback(query)
    elif "NOT_FOUND_IN_DOCS" in answer and is_opinion:
        answer = "Cannot determine from documents — documents only contain question banks without explicit weightage or ranking information."

    # Step 6: update memory
    session["history"].add_user_message(query)
    session["history"].add_ai_message(answer)
    await maybe_summarize(session)

    sources = list({
        f"{c.metadata.get('source', 'unknown')} | Page {c.metadata.get('page', '?')}"
        for c in chunks
    })

    return {
        "answer": answer,
        "sources": sources,
        "searched_docs": selected_files,
        "session_id": session_id
    }



# CLI

async def main():
    import uuid
    session_id = str(uuid.uuid4())
    print(f"Session: {session_id}\n")

    while True:
        q = input("You: ")
        if q.lower() in ("exit", "quit"):
            break
        result = await chat(q, session_id)
        print(f"\nAI: {result['answer']}")
        print(f"Sources: {result['sources']}")
        print(f"Searched: {result['searched_docs']}\n")


if __name__ == "__main__":
    asyncio.run(main())