import os
import re
import io
import time
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
import requests
from openai import OpenAI
import chromadb
from chromadb.config import Settings
import pypdf

load_dotenv()

BASE_URL = os.getenv("BASE_URL", "http://localhost:11434/v1")
OLLAMA_API = os.getenv("OLLAMA_API", "http://localhost:11434")
OLLAMA_API_KEY = os.getenv("API_KEY", "ollama")
OLLAMA_MODEL = os.getenv("MODEL", "qwen3:4b")
EMBEDDING_MODEL = os.getenv("EMODEL", "qwen3-embedding:0.6b")

GEMINI_BASE_URL = os.getenv("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")
GEMINI_API_KEY = os.getenv("API_KEY_G", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")

CHROMA_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")

class RAGEngine:
    def __init__(self, persist_dir: str = CHROMA_DIR):
        self.persist_dir = persist_dir
        os.makedirs(self.persist_dir, exist_ok=True)
        self.chroma_client = chromadb.PersistentClient(path=self.persist_dir)
        self.collection = self.chroma_client.get_or_create_collection(
            name="rag_corpus",
            metadata={"hnsw:space": "cosine"}
        )
        self.active_model = GEMINI_MODEL if GEMINI_API_KEY else OLLAMA_MODEL

    def get_embedding(self, text: str) -> List[float]:
        """Compute embeddings using local Ollama model or Gemini fallback."""
        text = text.replace("\n", " ").strip()
        if not text:
            text = " "

        # 1. Try local Ollama embedding
        try:
            res = requests.post(
                f"{OLLAMA_API}/api/embeddings",
                json={"model": EMBEDDING_MODEL, "prompt": text},
                timeout=10
            )
            if res.status_code == 200:
                emb = res.json().get("embedding")
                if emb:
                    return emb
        except Exception as e:
            print(f"[Embedding] Ollama error: {e}")

        # 2. Try Gemini embedding if key is available
        if GEMINI_API_KEY:
            try:
                g_client = OpenAI(base_url=GEMINI_BASE_URL, api_key=GEMINI_API_KEY)
                res = g_client.embeddings.create(
                    model="text-embedding-004",
                    input=text
                )
                return res.data[0].embedding
            except Exception as e:
                print(f"[Embedding] Gemini error: {e}")

        # 3. Fallback to basic normalized frequency vector (dim 1024)
        print("[Embedding] Fallback deterministic vector")
        import hashlib
        h = hashlib.sha256(text.encode("utf-8")).digest()
        vec = [(b / 255.0) - 0.5 for b in h]
        while len(vec) < 1024:
            vec.extend(vec[:1024 - len(vec)])
        return vec[:1024]

    def extract_text(self, filename: str, content_bytes: bytes) -> str:
        """Extract plain text from PDF, TXT, or MD files."""
        ext = os.path.splitext(filename)[1].lower()
        if ext == ".pdf":
            reader = pypdf.PdfReader(io.BytesIO(content_bytes))
            pages = []
            for i, page in enumerate(reader.pages):
                page_text = page.extract_text() or ""
                if page_text.strip():
                    pages.append(page_text.strip())
            return "\n\n".join(pages)
        elif ext in [".txt", ".md", ".markdown"]:
            return content_bytes.decode("utf-8", errors="replace")
        else:
            return content_bytes.decode("utf-8", errors="replace")

    def chunk_text(self, text: str, chunk_size: int = 500, chunk_overlap: int = 80) -> List[str]:
        """Split text into overlapping chunks respecting sentence and paragraph boundaries."""
        text = text.strip()
        if not text:
            return []
        if len(text) <= chunk_size:
            return [text]

        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        chunks = []
        current_chunk = ""

        for para in paragraphs:
            if len(current_chunk) + len(para) + 2 <= chunk_size:
                current_chunk = (current_chunk + "\n\n" + para).strip()
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                # If paragraph itself is too large, split by sentences
                if len(para) > chunk_size:
                    sentences = re.split(r'(?<=[.!?])\s+', para)
                    s_chunk = ""
                    for s in sentences:
                        if len(s_chunk) + len(s) + 1 <= chunk_size:
                            s_chunk = (s_chunk + " " + s).strip()
                        else:
                            if s_chunk:
                                chunks.append(s_chunk)
                            s_chunk = s
                    if s_chunk:
                        current_chunk = s_chunk
                    else:
                        current_chunk = ""
                else:
                    # Start new chunk with slight overlap if possible
                    overlap_prefix = current_chunk[-chunk_overlap:] if len(current_chunk) >= chunk_overlap else ""
                    current_chunk = (overlap_prefix + " " + para).strip() if overlap_prefix else para

        if current_chunk and current_chunk not in chunks:
            chunks.append(current_chunk)

        # Cleanup and ensure minimum chunk length
        valid_chunks = [c.strip() for c in chunks if len(c.strip()) > 10]
        return valid_chunks if valid_chunks else [text[:chunk_size]]

    def index_document(self, filename: str, content_bytes: bytes) -> Dict[str, Any]:
        """Parse, chunk, embed, and index a document into ChromaDB."""
        text = self.extract_text(filename, content_bytes)
        if not text.strip():
            raise ValueError(f"No extractable text found in '{filename}'.")

        # Delete any prior chunks for this file to allow re-indexing
        self.delete_document(filename)

        chunks = self.chunk_text(text)
        if not chunks:
            raise ValueError("Document contains no valid chunks.")

        ids = []
        embeddings = []
        metadatas = []
        documents = []

        now_ts = int(time.time())
        clean_name = re.sub(r'[^a-zA-Z0-9_\-\.]', '_', filename)

        for i, chunk in enumerate(chunks):
            chunk_id = f"{clean_name}_chunk_{i}"
            emb = self.get_embedding(chunk)

            ids.append(chunk_id)
            embeddings.append(emb)
            documents.append(chunk)
            metadatas.append({
                "doc_name": filename,
                "chunk_index": i,
                "total_chunks": len(chunks),
                "timestamp": now_ts,
                "char_length": len(chunk)
            })

        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )

        return {
            "filename": filename,
            "chunks_count": len(chunks),
            "char_count": len(text),
            "status": "success"
        }

    def list_documents(self) -> Dict[str, Any]:
        """Return indexed documents and chunk statistics."""
        data = self.collection.get(include=["metadatas"])
        docs_map = {}

        if data and "metadatas" in data and data["metadatas"]:
            for meta in data["metadatas"]:
                doc_name = meta.get("doc_name", "unknown")
                if doc_name not in docs_map:
                    docs_map[doc_name] = {
                        "filename": doc_name,
                        "chunks_count": 0,
                        "timestamp": meta.get("timestamp", 0)
                    }
                docs_map[doc_name]["chunks_count"] += 1

        docs_list = list(docs_map.values())
        docs_list.sort(key=lambda x: x["timestamp"], reverse=True)

        total_chunks = sum(d["chunks_count"] for d in docs_list)
        return {
            "documents": docs_list,
            "total_docs": len(docs_list),
            "total_chunks": total_chunks
        }

    def delete_document(self, filename: str) -> bool:
        """Remove all chunks associated with a specific document."""
        data = self.collection.get(where={"doc_name": filename})
        if data and data.get("ids"):
            self.collection.delete(ids=data["ids"])
            return True
        return False

    def clear_corpus(self) -> bool:
        """Clear the entire vector database collection."""
        try:
            self.chroma_client.delete_collection("rag_corpus")
        except Exception:
            pass
        self.collection = self.chroma_client.get_or_create_collection(
            name="rag_corpus",
            metadata={"hnsw:space": "cosine"}
        )
        return True

    def retrieve(self, query: str, top_k: int = 4) -> List[Dict[str, Any]]:
        """Perform vector similarity search for raw chunk matches."""
        query = query.strip()
        if not query:
            return []

        doc_count = self.collection.count()
        if doc_count == 0:
            return []

        actual_k = min(top_k, doc_count)
        q_emb = self.get_embedding(query)

        results = self.collection.query(
            query_embeddings=[q_emb],
            n_results=actual_k,
            include=["documents", "metadatas", "distances"]
        )

        matches = []
        if results and "documents" in results and results["documents"]:
            docs = results["documents"][0]
            metas = results["metadatas"][0] if "metadatas" in results else [{}] * len(docs)
            distances = results["distances"][0] if "distances" in results else [0.0] * len(docs)

            for i in range(len(docs)):
                dist = distances[i]
                # Cosine distance to similarity: similarity = 1 - distance (or max(0, 1 - dist))
                sim_score = max(0.0, 1.0 - dist)
                matches.append({
                    "chunk_id": results["ids"][0][i],
                    "text": docs[i],
                    "metadata": metas[i],
                    "doc_name": metas[i].get("doc_name", "Unknown"),
                    "chunk_index": metas[i].get("chunk_index", 0),
                    "similarity": round(sim_score, 4),
                    "distance": round(dist, 4)
                })

        return matches

    def ask(self, question: str, top_k: int = 4, model: Optional[str] = None) -> Dict[str, Any]:
        """Perform grounded Q&A against retrieved chunks."""
        matches = self.retrieve(question, top_k=top_k)
        if not matches:
            return {
                "answer": "No relevant documents found in the corpus. Please upload and index documents first.",
                "sources": [],
                "model_used": model or self.active_model
            }

        # Format context
        context_blocks = []
        for i, m in enumerate(matches, 1):
            context_blocks.append(
                f"--- SOURCE [{i}] (File: {m['doc_name']}, Chunk #{m['chunk_index'] + 1}, Relevance: {m['similarity']:.2f}) ---\n{m['text']}"
            )
        context_str = "\n\n".join(context_blocks)

        prompt = f"""You are a precise, helpful AI answering questions grounded strictly in the provided document context.

CONTEXT INFORMATION:
{context_str}

USER QUESTION:
{question}

INSTRUCTIONS:
1. Answer the question factually based ONLY on the context provided above.
2. If the context does not contain the answer, say "The provided documents do not contain enough information to answer this question."
3. Highlight key facts and cite which source(s) provide the evidence (e.g., [Source 1]).
4. Maintain a professional, clear, and concise tone.
"""

        chosen_model = model or self.active_model

        # Execute generation
        answer = ""
        # 1. If Gemini
        if "gemini" in chosen_model.lower() and GEMINI_API_KEY:
            try:
                client = OpenAI(base_url=GEMINI_BASE_URL, api_key=GEMINI_API_KEY)
                resp = client.chat.completions.create(
                    model=chosen_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2
                )
                answer = resp.choices[0].message.content or ""
            except Exception as e:
                print(f"[LLM] Gemini error: {e}")

        # 2. If Ollama or fallback
        if not answer:
            try:
                # Direct call to Ollama generate API
                ollama_res = requests.post(
                    f"{OLLAMA_API}/api/generate",
                    json={
                        "model": OLLAMA_MODEL,
                        "prompt": prompt,
                        "stream": False,
                        "options": {"temperature": 0.2, "num_predict": 512}
                    },
                    timeout=60
                )
                if ollama_res.status_code == 200:
                    answer = ollama_res.json().get("response", "")
            except Exception as e:
                print(f"[LLM] Ollama error: {e}")

        # 3. Final fallback
        if not answer:
            answer = f"Synthesized from context:\n\nBased on {len(matches)} retrieved chunks from {', '.join(set(m['doc_name'] for m in matches))}:\n\n" + matches[0]["text"]

        return {
            "answer": answer.strip(),
            "sources": matches,
            "model_used": chosen_model
        }

