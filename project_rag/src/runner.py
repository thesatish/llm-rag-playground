import os
import glob
from pathlib import Path
from typing import List, Dict, Any, Optional
from .pipeline import AdvancedRAGPipeline
from .evaluation import RAGEvaluator
from config import config

class AdvancedRAGRunner:
    """
    Topic 12: Building the Final Runner for Our Advanced RAG Application.
    Provides a production-grade operational facade for:
      - Corpus ingestion and initialization
      - Interactive single and batch query executions
      - Comprehensive IR and RAG Triad evaluations
      - Health checks, telemetry reports, and configuration verification
    """
    def __init__(self, persist_dir: Optional[str] = None):
        self.pipeline = AdvancedRAGPipeline(persist_dir=persist_dir)
        self.evaluator = RAGEvaluator()

    def load_sample_documents(self, sample_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Scans sample_data directory and ingests all .md and .txt files.
        """
        if sample_dir is None:
            sample_dir = str(Path(__file__).resolve().parent.parent / "sample_data")

        files = glob.glob(os.path.join(sample_dir, "*.*"))
        ingested = []

        for fpath in files:
            fname = os.path.basename(fpath)
            ext = os.path.splitext(fname)[1].lower()
            if ext in [".md", ".txt"]:
                with open(fpath, "r", encoding="utf-8") as f:
                    content = f.read()
                doc_id = os.path.splitext(fname)[0]
                res = self.pipeline.ingest_document(
                    doc_id=doc_id,
                    text=content,
                    source_name=fname,
                    file_type=ext.replace(".", "")
                )
                ingested.append(res)

        return {
            "documents_loaded": len(ingested),
            "total_chunks": self.pipeline.vector_store.count(),
            "details": ingested
        }

    def ask(
        self,
        query: str,
        top_k: int = 4,
        hybrid_method: str = "rrf",
        alpha: float = 0.5,
        use_cache: bool = True,
        use_hyde: bool = False,
        use_multiquery: bool = False
    ) -> Dict[str, Any]:
        """Runs the query through the pipeline and formats the output."""
        return self.pipeline.run(
            query=query,
            top_k=top_k,
            use_cache=use_cache,
            use_hyde=use_hyde,
            use_multiquery=use_multiquery,
            hybrid_method=hybrid_method,
            alpha=alpha
        )

    def get_corpus_stats(self) -> Dict[str, Any]:
        """Returns statistics on current corpus, chunk counts, and collection status."""
        count = self.pipeline.vector_store.count()
        all_chunks = self.pipeline.vector_store.get_all_chunks()

        docs = set()
        total_tokens = 0
        for c in all_chunks:
            meta = c.get("metadata", {})
            docs.add(meta.get("source", "unknown"))
            total_tokens += meta.get("token_count", 0)

        return {
            "total_chunks": count,
            "unique_documents": len(docs),
            "document_sources": list(docs),
            "estimated_total_tokens": total_tokens,
            "vector_store_space": self.pipeline.vector_store.space,
            "embedding_dimension": self.pipeline.embedding_engine.dimension or "Dynamic",
            "model": self.pipeline.generator.model_name
        }

    def run_evaluation_benchmark(self, custom_test_cases: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Runs evaluation suite testing Hit Rate @ 5, MRR @ 5, NDCG @ 5, Context Relevance,
        Faithfulness, and Answer Relevance.
        """
        # If no custom test cases provided, generate standard evaluation suite
        test_cases = custom_test_cases or [
            {
                "query": "How does Reciprocal Rank Fusion combine search rankings?",
                "expected_chunk_id": "rag_deep_dive_c004"
            },
            {
                "query": "What is the lost in the middle phenomenon in context chunking?",
                "expected_chunk_id": "rag_deep_dive_c001"
            },
            {
                "query": "Explain how HNSW graph layers enable approximate nearest neighbor search.",
                "expected_chunk_id": "rag_deep_dive_c007"
            }
        ]

        return self.evaluator.evaluate_batch(test_cases, self.pipeline)

