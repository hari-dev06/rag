import os
import json
import requests
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings.base import Embeddings

load_dotenv()

VECTOR_STORE_ROOT = Path("vector_store")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
CHAT_MODEL = "openrouter/free"


class OpenRouterEmbeddings(Embeddings):
    def __init__(self, model: str, api_key: str):
        self.model = model
        self.api_key = api_key

    def _embed(self, texts: list[str]) -> list[list[float]]:
        response = requests.post(
            "https://openrouter.ai/api/v1/embeddings",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            },
            json={"model": self.model, "input": texts}
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


def summarize_pdf(text_sample: str, filename: str) -> str:
    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": CHAT_MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": f"Summarize in 2-3 sentences what this document is about. Be specific about topics covered.\n\nFilename: {filename}\n\nContent sample:\n{text_sample[:3000]}"
                }
            ]
        }
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"].get("content") or ""


def ingest_pdf(pdf_path: str) -> dict:
    pdf_path = Path(pdf_path)
    filename = pdf_path.name
    store_path = VECTOR_STORE_ROOT / filename

    # Load
    loader = PyPDFLoader(str(pdf_path))
    documents = loader.load()

    # Chunk
    splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=400,
    separators=["\n\n", "\n", ".", " "]
    )
    chunks = splitter.split_documents(documents)

    # Embed + store in per-PDF index
    store_path.mkdir(parents=True, exist_ok=True)
    faiss_index_path = str(store_path)

    if (store_path / "index.faiss").exists():
        store = FAISS.load_local(faiss_index_path, embeddings, allow_dangerous_deserialization=True)
        store.add_documents(chunks)
    else:
        store = FAISS.from_documents(chunks, embeddings)

    store.save_local(faiss_index_path)

    # Generate summary for routing
    full_text = "\n".join([d.page_content for d in documents])
    summary = summarize_pdf(full_text, filename)

    # Save metadata
    meta = {
        "filename": filename,
        "pages": len(documents),
        "chunks": len(chunks),
        "summary": summary
    }
    with open(store_path / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    return meta


def list_ingested() -> list[dict]:
    if not VECTOR_STORE_ROOT.exists():
        return []
    result = []
    for folder in VECTOR_STORE_ROOT.iterdir():
        meta_file = folder / "meta.json"
        if meta_file.exists():
            with open(meta_file) as f:
                result.append(json.load(f))
    return result


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python ingest.py <pdf_path>")
        sys.exit(1)

    result = ingest_pdf(sys.argv[1])
    print(f"\nIngested: {result['filename']}")
    print(f"Pages: {result['pages']} | Chunks: {result['chunks']}")
    print(f"Summary: {result['summary']}")

    print("\n--- All ingested docs ---")
    for doc in list_ingested():
        print(f"- {doc['filename']}: {doc['summary'][:100]}...")