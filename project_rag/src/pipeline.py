import time
from typing import List, Dict, Any, Optional
from .chunking import Chunk, RecursiveCharacterChunker, MarkdownSemanticChunker
from .embeddings import EmbeddingEngine
from .vector_store import ChromaVectorStore
from .keyword_search import BM25SearchEngine
from .semantic_search import SemanticSearchEngine
from .hybrid_search import HybridSearchEngine
from .reranker import ReRanker
from .retrieval_intelligence import RetrievalIntelligence
from .generator import AnswerGenerator
from config import config

class AdvancedRAGPipeline:
    """
    Topic 10: Building the Complete Advanced RAG Retrieval Pipeline.
    Orchestrates the entire end-to-end multi-stage architecture:
      1. Ingestion: Semantic Chunking -> Vector Store + BM25 Inverted Index
      2. Intelligent Pre-Retrieval: Semantic Cache -> Query Routing -> Multi-Query / HyDE
      3. First-Stage Retrieval: Hybrid Search (ChromaDB Dense + BM25 Sparse via RRF)
      4. Second-Stage Re-Ranking: Cross-Encoder / Heuristic Scorer
      5. Post-Retrieval Intelligence: CRAG Document Grader
      6. Synthesis: Grounded Generator with Citations & Guardrails
    """
    def __init__(
        self,
        persist_dir: Optional[str] = None,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None
    ):
        self.chunk_size = chunk_size or config.DEFAULT_CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or config.DEFAULT_CHUNK_OVERLAP

        # Subsystems
        self.chunker = MarkdownSemanticChunker(
            max_chunk_size=self.chunk_size,
            overlap=self.chunk_overlap
        )
        self.embedding_engine = EmbeddingEngine()
        self.vector_store = ChromaVectorStore(persist_dir=persist_dir)
        self.semantic_search = SemanticSearchEngine(
            vector_store=self.vector_store,
            embedding_engine=self.embedding_engine
        )
        self.keyword_search = BM25SearchEngine()
        self.hybrid_search = HybridSearchEngine(
            semantic_engine=self.semantic_search,
            keyword_engine=self.keyword_search,
            default_alpha=config.DEFAULT_HYBRID_ALPHA,
            rrf_k=config.DEFAULT_RRF_K
        )
        self.reranker = ReRanker(strategy="cross_scorer")
        self.intelligence = RetrievalIntelligence(embedding_engine=self.embedding_engine)
        self.generator = AnswerGenerator()

        # Re-index existing vector store chunks into BM25 on initialization
        self._sync_bm25_from_vector_store()

    def _sync_bm25_from_vector_store(self):
        """Loads existing chunks from ChromaDB and builds the in-memory BM25 index."""
        raw_chunks = self.vector_store.get_all_chunks()
        if raw_chunks:
            chunks = [
                Chunk(
                    chunk_id=c["chunk_id"],
                    doc_id=c["metadata"].get("doc_id", "doc"),
                    text=c["text"],
                    metadata=c["metadata"]
                )
                for c in raw_chunks
            ]
            self.keyword_search.index_documents(chunks)

    def ingest_document(
        self,
        doc_id: str,
        text: str,
        source_name: str = "document",
        file_type: str = "md"
    ) -> Dict[str, Any]:
        """
        Ingestion Pipeline:
        1. Breaks document into enriched chunks using structural/markdown chunker.
        2. Generates dense embeddings.
        3. Upserts chunks into ChromaDB.
        4. Re-indexes BM25 sparse index.
        """
        start_time = time.time()
        chunks = self.chunker.chunk(text, doc_id=doc_id, source_name=source_name)
        if not chunks:
            return {"status": "empty", "chunks_indexed": 0, "latency_sec": 0.0}

        texts = [c.text for c in chunks]
        embeddings = self.embedding_engine.embed_documents(texts)

        self.vector_store.add_chunks(chunks, embeddings)
        self._sync_bm25_from_vector_store()

        elapsed = round(time.time() - start_time, 3)
        return {
            "status": "success",
            "doc_id": doc_id,
            "source_name": source_name,
            "chunks_indexed": len(chunks),
            "latency_sec": elapsed
        }

    def run(
        self,
        query: str,
        top_k: int = 4,
        candidate_k: int = 12,
        use_cache: bool = True,
        use_hyde: bool = False,
        use_multiquery: bool = False,
        hybrid_method: str = "rrf",
        alpha: float = 0.5,
        where_filter: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Executes complete multi-stage RAG query pipeline with component profiling.
        """
        pipeline_start = time.time()
        telemetry: Dict[str, float] = {}

        # Stage 0: Semantic Cache Check
        if use_cache:
            cached = self.intelligence.check_semantic_cache(query)
            if cached:
                cached["telemetry"]["pipeline_total_sec"] = round(time.time() - pipeline_start, 4)
                return cached

        # Stage 1: Query Routing
        t0 = time.time()
        routing = self.intelligence.route_query(query)
        print("\n routing::::", routing)
        print()
        telemetry["routing_sec"] = round(time.time() - t0, 4)

        if routing["route"] == "DIRECT_LLM":
            # Direct response without retrieval
            answer_payload = self.generator.generate_answer(query, context_chunks=[])
            return {
                "query": query,
                "route": routing,
                "answer": answer_payload["answer"],
                "citations": [],
                "retrieved_chunks": [],
                "telemetry": {"total_sec": round(time.time() - pipeline_start, 4)}
            }

        # Stage 2: Query Transformation
        search_queries = [query]
        t1 = time.time()
        if use_multiquery:
            search_queries = self.intelligence.expand_query_multiquery(query)
        elif use_hyde:
            search_queries = [self.intelligence.generate_hyde_document(query)]
        telemetry["query_transformation_sec"] = round(time.time() - t1, 4)

        # Stage 3: First-Stage Hybrid Retrieval
        t2 = time.time()
        print("Stage 2: search_queries:::::", search_queries )
        candidate_pool: List[Dict[str, Any]] = []
        for sq in search_queries:
            if hybrid_method == "rrf":
                res = self.hybrid_search.search_rrf(sq, top_k=candidate_k, where_filter=where_filter)
            else:
                res = self.hybrid_search.search_convex(sq, alpha=alpha, top_k=candidate_k, where_filter=where_filter)

            # Deduplicate by chunk_id
            seen = {c["chunk_id"] for c in candidate_pool}
            for item in res:
                if item["chunk_id"] not in seen:
                    candidate_pool.append(item)
                    seen.add(item["chunk_id"])

        telemetry["hybrid_retrieval_sec"] = round(time.time() - t2, 4)

        # Stage 4: Second-Stage Cross-Encoder Re-Ranking
        t3 = time.time()
        reranked_chunks = self.reranker.rerank(query, candidate_pool, top_k=top_k)
        print("statge 4 reranked_chunks:::::", reranked_chunks)
        telemetry["reranking_sec"] = round(time.time() - t3, 4)

        # Stage 5: Corrective RAG (CRAG) Document Filtering
        t4 = time.time()
        final_chunks = self.intelligence.grade_documents_crag(query, reranked_chunks)
        print("Statge 5 : final_chunks:::::::", final_chunks)
        telemetry["crag_filter_sec"] = round(time.time() - t4, 4)

        # Stage 6: Grounded Generation with Citations
        t5 = time.time()
        print("query::::", query)
        
        gen_result = self.generator.generate_answer(query, context_chunks=final_chunks)
        
        print("Statge 6: gen_result::::::::", gen_result)
        
        telemetry["generation_sec"] = round(time.time() - t5, 4)

        total_elapsed = round(time.time() - pipeline_start, 4)
        telemetry["pipeline_total_sec"] = total_elapsed

        response_payload = {
            "query": query,
            "route": routing,
            "search_queries": search_queries,
            "answer": gen_result["answer"],
            "citations": gen_result["citations"],
            "retrieved_chunks": final_chunks,
            "candidate_pool_size": len(candidate_pool),
            "model_used": gen_result["model"],
            "usage": gen_result.get("usage", {}),
            "telemetry": telemetry,
            "cache_hit": False,
            "reranked_chunks" : reranked_chunks,
            "gen_result" : gen_result,
            "final_chunks" :final_chunks,
            
            
        }

        # Store in Semantic Cache
        if use_cache and gen_result.get("grounded", False):
            self.intelligence.store_in_semantic_cache(query, response_payload)

        return response_payload

