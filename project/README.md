# Minimal RAG System Playground

A complete, full-stack Retrieval-Augmented Generation (RAG) system with a local vector database, document indexing (PDF, TXT, MD), semantic retrieval inspector, and grounded LLM generation, designed to match the playground interface.

---

## Features

1. **Corpus Management & Document Indexing**:
   - Ingests **PDF**, **TXT**, and **Markdown** documents.
   - Intelligent text extraction and sentence/paragraph-aware chunking with customizable overlap.
   - Vector database storage powered locally by **ChromaDB**.
   - Embeddings powered by local **Ollama (`qwen3-embedding:0.6b`)** with fallback to Google Gemini.
   - Status indicators (`Empty`, `Indexed`), one-click **Refresh**, and **Clear corpus**.

2. **Retrieve Inspector**:
   - Inspect raw chunk matches before generation.
   - Tunable **Top K** chunks selector.
   - Displays similarity scores, source file names, chunk indices, and formatted text previews.

3. **Grounded Q&A (Ask RAG)**:
   - Grounded generation passing retrieved top-K context into the prompt to prevent hallucinations.
   - Sources and citations inspector showing exactly which chunks and files grounded the response.
   - Model switcher supporting **gpt-5.4-mini**, **gemini-3.5-flash-lite**, and local **qwen3:4b**.

---

## Quick Start

### 1. Start the Server
Run the startup script:
```bash
./run.sh
```
Or directly with Python:
```bash
python3 -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

### 2. Open the Playground
Navigate to:
```
http://localhost:8000
```

### 3. Test with Sample Documents
Sample documents are included in `sample_docs/`:
- `sample_docs/rag_fundamentals.md`
Upload and index the file, test retrieval queries (e.g. `What are the key components of RAG?`), and ask grounded questions!

