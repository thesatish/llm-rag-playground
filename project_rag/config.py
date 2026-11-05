import os
from pathlib import Path
from dotenv import load_dotenv

# Search for .env in current directory, project_rag, or parent directory
BASE_DIR = Path(__file__).resolve().parent
PARENT_DIR = BASE_DIR.parent

if (BASE_DIR / ".env").exists():
    load_dotenv(BASE_DIR / ".env")
elif (PARENT_DIR / ".env").exists():
    load_dotenv(PARENT_DIR / ".env")
else:
    load_dotenv()

class RAGConfig:
    """Central configuration for Advanced RAG pipeline and server."""
    # Ollama settings
    OLLAMA_BASE_URL: str = os.getenv("BASE_URL", "http://localhost:11434/v1")
    OLLAMA_API_BASE: str = os.getenv("OLLAMA_API", "http://localhost:11434")
    OLLAMA_API_KEY: str = os.getenv("API_KEY", "ollama")
    OLLAMA_MODEL: str = os.getenv("MODEL", "qwen3:4b")
    OLLAMA_EMBED_MODEL: str = os.getenv("EMODEL", "qwen3-embedding:0.6b")

    # Gemini settings
    GEMINI_BASE_URL: str = os.getenv("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")
    GEMINI_API_KEY: str = os.getenv("API_KEY_G", "")
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

    # Preferred Provider ('gemini' or 'ollama')
    # If Gemini API key is present, default to Gemini for LLM completions for speed, Ollama for embeddings
    PREFERRED_LLM_PROVIDER: str = os.getenv("PREFERRED_LLM_PROVIDER", "gemini" if os.getenv("API_KEY_G") else "ollama")
    PREFERRED_EMBED_PROVIDER: str = os.getenv("PREFERRED_EMBED_PROVIDER", "ollama")

    # Chunking defaults
    DEFAULT_CHUNK_SIZE: int = int(os.getenv("RAG_CHUNK_SIZE", "500"))
    DEFAULT_CHUNK_OVERLAP: int = int(os.getenv("RAG_CHUNK_OVERLAP", "100"))

    # Retrieval & Reranking parameters
    DEFAULT_TOP_K_RETRIEVAL: int = int(os.getenv("RAG_TOP_K_RETRIEVAL", "10"))
    DEFAULT_TOP_K_RERANK: int = int(os.getenv("RAG_TOP_K_RERANK", "4"))
    DEFAULT_HYBRID_ALPHA: float = float(os.getenv("RAG_HYBRID_ALPHA", "0.5"))
    DEFAULT_RRF_K: int = int(os.getenv("RAG_RRF_K", "60"))

    # Vector store path
    CHROMA_PERSIST_DIR: str = str(BASE_DIR / os.getenv("RAG_PERSIST_DIR", "chroma_db"))
    COLLECTION_NAME: str = "advanced_rag_collection"

    # Server settings
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8001"))

config = RAGConfig()

