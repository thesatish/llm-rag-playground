import os
import shutil
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
from config import config
from .chunking import Chunk

class ChromaVectorStore:
    """
    Topic 7: Building a Vector Store with ChromaDB.
    Encapsulates ChromaDB persistent storage, collection management,
    HNSW indexing configuration, vector insertion, and metadata filtering.
    """
    def __init__(
        self,
        persist_dir: Optional[str] = None,
        collection_name: Optional[str] = None,
        space: str = "cosine"
    ):
        self.persist_dir = persist_dir or config.CHROMA_PERSIST_DIR
        self.collection_name = collection_name or config.COLLECTION_NAME
        self.space = space

        os.makedirs(self.persist_dir, exist_ok=True)
        self.client = chromadb.PersistentClient(path=self.persist_dir)
        self.collection = self._get_or_create_collection()

    def _get_or_create_collection(self):
        """Creates or retrieves a ChromaDB collection with configured distance metric."""
        return self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": self.space}
        )

    def add_chunks(self, chunks: List[Chunk], embeddings: List[List[float]]):
        """
        Adds or upserts chunk embeddings, texts, and metadata into ChromaDB.
        Avoids external model downloads by passing explicit pre-computed vectors.
        """
        if not chunks:
            return

        if len(chunks) != len(embeddings):
            raise ValueError(f"Chunk count ({len(chunks)}) and Embedding count ({len(embeddings)}) mismatch.")

        ids = [c.chunk_id for c in chunks]
        documents = [c.text for c in chunks]

        # Chroma metadata values must be primitive types (str, int, float, bool)
        metadatas = []
        for c in chunks:
            sanitized_meta = {}
            for k, v in c.metadata.items():
                if isinstance(v, (str, int, float, bool)):
                    sanitized_meta[k] = v
                elif v is None:
                    continue
                else:
                    sanitized_meta[k] = str(v)
            metadatas.append(sanitized_meta)

        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )

    def query_vectors(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        where_filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Queries ChromaDB using dense vector embedding and optional metadata filter.
        """
        count = self.collection.count()
        if count == 0:
            return []

        actual_k = min(top_k, count)
        kwargs: Dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": actual_k,
            "include": ["documents", "metadatas", "distances"]
        }
        if where_filter:
            kwargs["where"] = where_filter

        results = self.collection.query(**kwargs)

        formatted: List[Dict[str, Any]] = []
        if not results or not results["ids"] or not results["ids"][0]:
            return formatted

        ids = results["ids"][0]
        docs = results["documents"][0] if results.get("documents") else []
        metas = results["metadatas"][0] if results.get("metadatas") else []
        dists = results["distances"][0] if results.get("distances") else []

        for i in range(len(ids)):
            distance = dists[i] if i < len(dists) else 0.0
            # For cosine distance: distance in [0, 2], similarity = 1 - (distance / 2) or 1 - distance
            similarity = max(0.0, 1.0 - distance)
            formatted.append({
                "chunk_id": ids[i],
                "text": docs[i] if i < len(docs) else "",
                "metadata": metas[i] if i < len(metas) else {},
                "score": float(similarity),
                "distance": float(distance),
                "retrieval_type": "semantic"
            })

        return formatted

    def get_chunk(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """Fetch a specific chunk by ID."""
        res = self.collection.get(ids=[chunk_id], include=["documents", "metadatas"])
        if res and res["ids"] and len(res["ids"]) > 0:
            return {
                "chunk_id": res["ids"][0],
                "text": res["documents"][0] if res.get("documents") else "",
                "metadata": res["metadatas"][0] if res.get("metadatas") else {}
            }
        return None

    def get_all_chunks(self) -> List[Dict[str, Any]]:
        """Retrieve all stored documents and metadata in the collection."""
        res = self.collection.get(include=["documents", "metadatas"])
        chunks = []
        if not res or not res.get("ids"):
            return chunks

        for i, cid in enumerate(res["ids"]):
            chunks.append({
                "chunk_id": cid,
                "text": res["documents"][i] if res.get("documents") else "",
                "metadata": res["metadatas"][i] if res.get("metadatas") else {}
            })
        return chunks

    def count(self) -> int:
        return self.collection.count()

    def clear(self):
        """Clears all vectors in the collection."""
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass
        self.collection = self._get_or_create_collection()

