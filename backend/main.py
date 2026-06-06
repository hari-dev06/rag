import uuid
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path
import shutil

from ingest import ingest_pdf, list_ingested
from rag_chain import chat, sessions, _metadata_cache

app = FastAPI(title="RAG Document Q&A")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


# Models

class ChatRequest(BaseModel):
    query: str
    session_id: str | None = None

class ChatResponse(BaseModel):
    answer: str
    sources: list[str]
    searched_docs: list[str]
    session_id: str


# Routes 

@app.get("/")
def root():
    return {"status": "ok"}


@app.post("/ingest")
async def ingest(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files allowed")

    # Save upload
    save_path = UPLOAD_DIR / file.filename
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Ingest
    try:
        result = ingest_pdf(str(save_path))
        # Bust metadata cache so new doc available immediately
        _metadata_cache.clear()
        return {"status": "success", **result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/documents")
def list_documents():
    return {"documents": list_ingested()}


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest):
    session_id = req.session_id or str(uuid.uuid4())
    try:
        result = await chat(req.query, session_id)
        return ChatResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/sessions/{session_id}")
def get_session_info(session_id: str):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    session = sessions[session_id]
    return {
        "session_id": session_id,
        "message_count": len(session["history"].messages),
        "has_summary": bool(session["summary"])
    }


@app.delete("/sessions/{session_id}")
def clear_session(session_id: str):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    del sessions[session_id]
    return {"status": "cleared", "session_id": session_id}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "documents": len(list_ingested()),
        "active_sessions": len(sessions)
    }