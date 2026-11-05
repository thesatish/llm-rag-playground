import unittest
import shutil
import tempfile
import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.chunking import FixedSizeChunker, RecursiveCharacterChunker, MarkdownSemanticChunker, Chunk
from src.embeddings import EmbeddingEngine
from src.vector_store import ChromaVectorStore
from src.keyword_search import BM25SearchEngine
from src.semantic_search import SemanticSearchEngine
from src.hybrid_search import HybridSearchEngine
from src.reranker import ReRanker
from src.evaluation import RAGEvaluator
from src.retrieval_intelligence import RetrievalIntelligence
from src.generator import AnswerGenerator
from src.pipeline import AdvancedRAGPipeline
from src.runner import AdvancedRAGRunner

class TestAdvancedRAG(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.mkdtemp()
        cls.runner = AdvancedRAGRunner(persist_dir=cls.temp_dir)
        cls.runner.load_sample_documents()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.temp_dir, ignore_errors=True)

    def test_topic_1_and_2_chunking_and_metadata(self):
        """Test Chunking strategies, sliding window stride, and metadata preservation."""
        text = "Sentence one. Sentence two. Sentence three. Sentence four. Sentence five."
        chunker = FixedSizeChunker(chunk_size=35, chunk_overlap=10)
        chunks = chunker.chunk(text, doc_id="test_doc", source_name="test.txt")

        self.assertGreaterEqual(len(chunks), 2)
        for c in chunks:
            self.assertEqual(c.metadata["doc_id"], "test_doc")
            self.assertIn("char_start", c.metadata)
            self.assertIn("char_end", c.metadata)
            self.assertEqual(c.metadata["overlap"], 10)

        # Markdown chunker
        md_text = "# Title\n\nIntro text.\n\n## Section 1\n\nSection content details."
        md_chunker = MarkdownSemanticChunker(max_chunk_size=100, overlap=20)
        md_chunks = md_chunker.chunk(md_text, doc_id="md_doc")
        self.assertGreaterEqual(len(md_chunks), 2)
        headings = [c.metadata.get("section_heading") for c in md_chunks]
        self.assertIn("Section 1", headings)

    def test_topic_3_embeddings_and_similarity(self):
        """Test embedding engine, dimension consistency, L2 normalization, and cosine similarity."""
        engine = EmbeddingEngine()
        vec1 = engine.embed_query("Artificial intelligence and machine learning")
        vec2 = engine.embed_query("Deep learning neural networks and AI")
        vec3 = engine.embed_query("Baking fresh sourdough bread in an oven")

        self.assertGreater(len(vec1), 0)
        self.assertEqual(len(vec1), len(vec2))
        self.assertEqual(len(vec2), len(vec3))

        sim_high = engine.cosine_similarity(vec1, vec2)
        sim_low = engine.cosine_similarity(vec1, vec3)

        self.assertGreater(sim_high, sim_low)

    def test_topic_7_and_8_chromadb_vector_store(self):
        """Test ChromaDB persistent storage, insertion, and querying."""
        engine = EmbeddingEngine()
        vs = self.runner.pipeline.vector_store
        self.assertGreater(vs.count(), 0)

        # Query vector store directly
        q_emb = engine.embed_query("HNSW graphs skip list")
        matches = vs.query_vectors(q_emb, top_k=2)
        self.assertEqual(len(matches), 2)
        self.assertIn("score", matches[0])
        self.assertIn("chunk_id", matches[0])

    def test_topic_4_semantic_search(self):
        """Test dense semantic search."""
        results = self.runner.pipeline.semantic_search.search("How does HNSW graph navigation work?", top_k=3)
        self.assertGreater(len(results), 0)
        self.assertEqual(results[0]["rank"], 1)
        self.assertEqual(results[0]["retrieval_type"], "semantic_dense")

    def test_topic_5_keyword_bm25_and_hybrid(self):
        """Test BM25 lexical search and Hybrid Search (RRF & Convex)."""
        # BM25
        bm25_res = self.runner.pipeline.keyword_search.search("Reciprocal Rank Fusion", top_k=3)
        self.assertGreater(len(bm25_res), 0)
        self.assertEqual(bm25_res[0]["retrieval_type"], "keyword_bm25")

        # Hybrid RRF
        rrf_res = self.runner.pipeline.hybrid_search.search_rrf("Reciprocal Rank Fusion", top_k=3)
        self.assertGreater(len(rrf_res), 0)
        self.assertEqual(rrf_res[0]["retrieval_type"], "hybrid_rrf")
        self.assertIn("breakdown", rrf_res[0])

        # Hybrid Convex
        convex_res = self.runner.pipeline.hybrid_search.search_convex("BM25 Okapi", alpha=0.6, top_k=3)
        self.assertGreater(len(convex_res), 0)
        self.assertIn("hybrid_convex", convex_res[0]["retrieval_type"])

    def test_topic_6_reranker(self):
        """Test cross-encoder candidate re-ranking."""
        query = "What is the sliding window stride formula in chunking?"
        candidates = self.runner.pipeline.hybrid_search.search_rrf(query, top_k=5)
        reranked = self.runner.pipeline.reranker.rerank(query, candidates, top_k=3)

        self.assertLessEqual(len(reranked), 3)
        for r in reranked:
            self.assertIn("rerank_score", r)
            self.assertIn("rank_delta", r)
            self.assertEqual(r["stage"], "reranked")

    def test_topic_9_evaluation_metrics(self):
        """Test IR metrics (Hit Rate, MRR, NDCG) and RAG Triad."""
        evaluator = RAGEvaluator()
        retrieved = ["doc_c001", "doc_c002", "doc_c003"]
        ground_truth = "doc_c002"

        hr = evaluator.calculate_hit_rate(retrieved, ground_truth, k=3)
        mrr = evaluator.calculate_mrr(retrieved, ground_truth, k=3)
        ndcg = evaluator.calculate_ndcg(retrieved, ground_truth, k=3)

        self.assertEqual(hr, 1.0)
        self.assertEqual(mrr, 0.5)  # Rank 2 => 1/2 = 0.5
        self.assertGreater(ndcg, 0.0)

        # RAG Triad
        chunks = [{"text": "HNSW graphs use multiple layers for fast approximate nearest neighbor search."}]
        cr = evaluator.evaluate_context_relevance("HNSW nearest neighbor search", chunks)
        self.assertGreater(cr, 0.5)

        faith = evaluator.evaluate_faithfulness("HNSW graphs use multiple layers for fast nearest neighbor search.", chunks)
        self.assertGreater(faith, 0.7)

    def test_topic_10_and_13_retrieval_intelligence(self):
        """Test query routing, expansion, HyDE, and semantic cache."""
        intel = self.runner.pipeline.intelligence

        # Intent Routing
        route_chitchat = intel.route_query("Hello there!")
        self.assertEqual(route_chitchat["route"], "DIRECT_LLM")

        route_rag = intel.route_query("Explain vector distance metrics in ChromaDB.")
        self.assertIn("HYBRID", route_rag["route"])

        # Multi-query
        variations = intel.expand_query_multiquery("What is RRF in search?")
        self.assertGreaterEqual(len(variations), 1)

        # Semantic Cache
        q = "What is HNSW?"
        payload = {"answer": "HNSW is a graph index.", "grounded": True, "telemetry": {}}
        intel.store_in_semantic_cache(q, payload)
        cached = intel.check_semantic_cache("What is HNSW?")
        self.assertIsNotNone(cached)
        self.assertTrue(cached.get("cache_hit"))

    def test_topic_11_and_12_answer_generator_and_runner(self):
        """Test Answer Generator citation parsing and end-to-end Runner execution."""
        res = self.runner.ask("What is the formula for sliding window stride in chunking?", top_k=2)

        self.assertIn("answer", res)
        self.assertGreater(len(res["answer"]), 0)
        self.assertIn("telemetry", res)
        self.assertIn("pipeline_total_sec", res["telemetry"])
        self.assertGreater(len(res["retrieved_chunks"]), 0)

        # Corpus Stats
        stats = self.runner.get_corpus_stats()
        self.assertGreater(stats["total_chunks"], 0)
        self.assertGreater(stats["unique_documents"], 0)

if __name__ == "__main__":
    unittest.main()
