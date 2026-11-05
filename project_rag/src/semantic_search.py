from typing import List, Dict, Any, Optional
from .embeddings import EmbeddingEngine
from .vector_store import ChromaVectorStore

class SemanticSearchEngine:
    """
    Topic 4 & 8: Semantic Search with Dense Embeddings and ChromaDB.
    Bridges the vocabulary mismatch problem by mapping queries into dense vector space
    and querying ChromaDB's HNSW Approximate Nearest Neighbors (ANN) index.
    """
    def __init__(self, vector_store: ChromaVectorStore, embedding_engine: EmbeddingEngine):
        self.vector_store = vector_store
        self.embedding_engine = embedding_engine

    def search(
        self,
        query: str,
        top_k: int = 5,
        where_filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Executes semantic search:
        1. Embeds query text into high-dimensional vector space.
        2. Queries ChromaDB for nearest neighbor vectors.
        3. Returns ranked results with similarity scores and metadata.
        """
        if not query.strip():
            return []

        query_emb = self.embedding_engine.embed_query(query)
        results = self.vector_store.query_vectors(
            query_embedding=query_emb,
            top_k=top_k,
            where_filter=where_filter
        )

        for rank, item in enumerate(results, start=1):
            item["rank"] = rank
            item["retrieval_type"] = "semantic_dense"
            item["search_type"] = "semantic_dense"

        return results
