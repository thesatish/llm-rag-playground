import os
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from src.runner import AdvancedRAGRunner
from config import config

app = FastAPI(
    title="Advanced RAG Production Engine",
    description="Interactive platform demonstrating all 13 core pillars of Advanced Retrieval-Augmented Generation.",
    version="2.0.0"
)

# Global runner instance
runner = AdvancedRAGRunner()

# Ensure sample documents loaded on launch if store is empty
if runner.pipeline.vector_store.count() == 0:
    runner.load_sample_documents()

# Static files directory
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/")
def serve_index():
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "Advanced RAG API is running. Access /docs for Swagger documentation."}

# Pydantic Request Models
class SearchRequest(BaseModel):
    query: str
    top_k: int = 5
    where_filter: Optional[Dict[str, Any]] = None

class HybridSearchRequest(BaseModel):
    query: str
    method: str = "rrf"  # "rrf" or "convex"
    alpha: float = 0.5
    top_k: int = 5
    candidate_k: int = 15

class RerankRequest(BaseModel):
    query: str
    candidates: List[Dict[str, Any]]
    top_k: int = 4

class QueryPipelineRequest(BaseModel):
    query: str
    top_k: int = 4
    candidate_k: int = 12
    hybrid_method: str = "rrf"
    alpha: float = 0.5
    use_cache: bool = True
    use_hyde: bool = False
    use_multiquery: bool = False

class IngestTextRequest(BaseModel):
    doc_id: str
    text: str
    source_name: str = "manual_entry.txt"
    chunk_size: Optional[int] = 500
    chunk_overlap: Optional[int] = 100

# REST Endpoints
@app.get("/api/stats")
def get_stats():
    return runner.get_corpus_stats()

@app.post("/api/ingest/text")
def ingest_text(req: IngestTextRequest):
    res = runner.pipeline.ingest_document(
        doc_id=req.doc_id,
        text=req.text,
        source_name=req.source_name
    )
    return res

@app.post("/api/ingest/file")
async def ingest_file(file: UploadFile = File(...)):
    contents = await file.read()
    try:
        text = contents.decode("utf-8")
    except UnicodeDecodeError:
        # Fallback for binary / pdf
        import io
        import pypdf
        reader = pypdf.PdfReader(io.BytesIO(contents))
        text = "\n".join([page.extract_text() or "" for page in reader.pages])

    doc_id = os.path.splitext(file.filename)[0]
    res = runner.pipeline.ingest_document(
        doc_id=doc_id,
        text=text,
        source_name=file.filename
    )
    return res

@app.post("/api/search/semantic")
def search_semantic(req: SearchRequest):
    results = runner.pipeline.semantic_search.search(
        query=req.query,
        top_k=req.top_k,
        where_filter=req.where_filter
    )
    return {"query": req.query, "count": len(results), "results": results}

@app.post("/api/search/bm25")
def search_bm25(req: SearchRequest):
    results = runner.pipeline.keyword_search.search(
        query=req.query,
        top_k=req.top_k
    )
    return {"query": req.query, "count": len(results), "results": results}

@app.post("/api/search/hybrid")
def search_hybrid(req: HybridSearchRequest):
    if req.method == "convex":
        results = runner.pipeline.hybrid_search.search_convex(
            query=req.query,
            alpha=req.alpha,
            top_k=req.top_k,
            fetch_k=req.candidate_k
        )
    else:
        results = runner.pipeline.hybrid_search.search_rrf(
            query=req.query,
            top_k=req.top_k,
            fetch_k=req.candidate_k
        )
    return {"query": req.query, "method": req.method, "results": results}

@app.post("/api/rerank")
def rerank(req: RerankRequest):
    results = runner.pipeline.reranker.rerank(
        query=req.query,
        candidates=req.candidates,
        top_k=req.top_k
    )
    return {"query": req.query, "count": len(results), "results": results}

@app.post("/api/query")
def run_query_pipeline(req: QueryPipelineRequest):
    res = runner.pipeline.run(
        query=req.query,
        top_k=req.top_k,
        candidate_k=req.candidate_k,
        hybrid_method=req.hybrid_method,
        alpha=req.alpha,
        use_cache=req.use_cache,
        use_hyde=req.use_hyde,
        use_multiquery=req.use_multiquery
    )
    return res

@app.post("/api/evaluate")
def run_evaluation():
    return runner.run_evaluation_benchmark()

@app.post("/api/corpus/clear")
def clear_corpus():
    runner.pipeline.vector_store.clear()
    runner.pipeline.keyword_search.corpus.clear()
    return {"status": "cleared", "total_chunks": 0}

@app.post("/api/corpus/reload-samples")
def reload_samples():
    runner.pipeline.vector_store.clear()
    res = runner.load_sample_documents()
    return res

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host=config.HOST, port=config.PORT, reload=True)

