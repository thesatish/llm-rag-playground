import os
from typing import Optional, List
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rag_engine import RAGEngine

app = FastAPI(title="Minimal RAG System", version="1.0.0")

# Enable CORS for local testing if needed
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = RAGEngine()

# Static directories
STATIC_DIR = os.path.join(os.path.dirname(__file__), "frontend")
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class RetrieveRequest(BaseModel):
    query: str
    top_k: int = 4


class AskRequest(BaseModel):
    query: str
    top_k: int = 4
    model: Optional[str] = None


class ModelSelectRequest(BaseModel):
    model: str


@app.get("/", response_class=HTMLResponse)
async def serve_index():
    index_file = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_file):
        with open(index_file, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>Minimal RAG System - Frontend not found</h1>"


@app.get("/api/status")
async def get_status():
    doc_data = engine.list_documents()
    is_empty = doc_data["total_docs"] == 0
    return {
        "status": "ready",
        "is_empty": is_empty,
        "badge_text": "Empty" if is_empty else f"{doc_data['total_docs']} docs",
        "total_docs": doc_data["total_docs"],
        "total_chunks": doc_data["total_chunks"],
        "active_model": engine.active_model,
        "vector_db": "ChromaDB (Local)"
    }


@app.get("/api/documents")
async def get_documents():
    data = engine.list_documents()
    return data


@app.post("/api/upload")
async def upload_document(file: UploadFile = File(...)):
    filename = file.filename
    if not filename:
        raise HTTPException(status_code=400, detail="Filename cannot be empty")

    ext = os.path.splitext(filename)[1].lower()
    allowed_exts = [".pdf", ".txt", ".md", ".markdown"]
    if ext not in allowed_exts:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format '{ext}'. Supported: PDF, TXT, MD"
        )

    try:
        content = await file.read()
        res = engine.index_document(filename, content)
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/clear")
async def clear_corpus():
    engine.clear_corpus()
    return {"status": "success", "message": "Corpus cleared successfully."}


@app.delete("/api/documents/{filename}")
async def delete_document(filename: str):
    success = engine.delete_document(filename)
    if success:
        return {"status": "success", "message": f"Document '{filename}' deleted."}
    raise HTTPException(status_code=404, detail=f"Document '{filename}' not found.")


@app.post("/api/retrieve")
async def retrieve_chunks(req: RetrieveRequest):
    if not req.query.strip():
        return {"results": [], "query": req.query, "count": 0}
    try:
        results = engine.retrieve(req.query, top_k=req.top_k)
        return {
            "results": results,
            "query": req.query,
            "count": len(results)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/ask")
async def ask_rag(req: AskRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query question cannot be empty")
    try:
        res = engine.ask(req.query, top_k=req.top_k, model=req.model)
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/models")
async def get_models():
    models = [
        {"id": "gpt-5.4-mini", "name": "gpt-5.4-mini", "description": "Screenshot Default Model"},
        {"id": "gemini-3.5-flash-lite", "name": "gemini-3.5-flash-lite", "description": "Google Cloud Gemini Flash (Fast & High Intelligence)"},
        {"id": "qwen3:4b", "name": "qwen3:4b", "description": "Local Ollama 4B Model"}
    ]
    return {
        "active_model": engine.active_model,
        "models": models
    }


@app.post("/api/model")
async def set_model(req: ModelSelectRequest):
    engine.active_model = req.model
    return {"status": "success", "active_model": engine.active_model}

