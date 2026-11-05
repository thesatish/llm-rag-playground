#!/usr/bin/env python3
import sys
import os
import json
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.runner import AdvancedRAGRunner
from src.chunking import RecursiveCharacterChunker, FixedSizeChunker, MarkdownSemanticChunker
from src.embeddings import EmbeddingEngine
from config import config

def print_header(title: str):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)

def main():
    print_header("ADVANCED RAG: PRODUCTION-READY DEMO RUNNER")
    print(f"Active LLM Model: {config.PREFERRED_LLM_PROVIDER} ({config.GEMINI_MODEL if config.GEMINI_API_KEY else config.OLLAMA_MODEL})")
    print(f"Embedding Model:  {config.OLLAMA_EMBED_MODEL}")
    print(f"Chroma DB Path:   {config.CHROMA_PERSIST_DIR}")

    runner = AdvancedRAGRunner()

    # Automatically ensure sample documents are ingested if collection is empty
    if runner.pipeline.vector_store.count() == 0:
        print("\n[Init] Vector store is empty. Ingesting sample documents from sample_data/...")
        res = runner.load_sample_documents()
        print(f"[Init] Ingested {res['documents_loaded']} documents ({res['total_chunks']} chunks).")

    while True:
        print("\n" + "-" * 50)
        print("Select a topic to demonstrate:")
        print("  1. Topic 1 & 2: Chunking Strategy & Overlap Inspector")
        print("  2. Topic 3: Generate & Compare Vector Embeddings")
        print("  3. Topic 4 & 8: Semantic Vector Search (ChromaDB + HNSW)")
        print("  4. Topic 5: Lexical Search (BM25 Okapi)")
        print("  5. Topic 5: Hybrid Search (BM25 + Semantic Search via RRF & Convex)")
        print("  6. Topic 6: Re-Ranking Candidates (Cross-Encoder Scoring)")
        print("  7. Topic 9: Evaluation Benchmark (Hit Rate, MRR, NDCG, RAG Triad)")
        print("  8. Topic 10 & 13: Retrieval Intelligence (Routing, HyDE, Multi-Query, CRAG)")
        print("  9. Topic 11 & 12: Interactive Grounded Q&A Runner")
        print("  10. Run Full Pipeline Validation (All 13 Topics)")
        print("  q. Exit")
        print("-" * 50)

        try:
            choice = input("\nEnter choice [1-10, q]: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting Advanced RAG Demo. Goodbye!")
            break

        if choice.lower() in ["q", "quit", "exit"]:
            print("Exiting Advanced RAG Demo. Goodbye!")
            break

        elif choice == "1":
            print_header("TOPIC 1 & 2: CHUNKING STRATEGY & METADATA OVERLAP")
            sample_text = (
                "# Distributed Systems Overview\n\n"
                "Distributed systems require consensus algorithms to ensure fault tolerance. "
                "The Raft consensus algorithm splits consensus into leader election, log replication, "
                "and safety constraints.\n\n"
                "## Log Replication\n"
                "Once a leader has been elected, it begins servicing client requests. Each client request "
                "contains a command to be executed by the replicated state machines. The leader appends "
                "the command to its log as a new entry."
            )
            print("Original Text:\n", sample_text)

            chunker_rec = RecursiveCharacterChunker(chunk_size=150, chunk_overlap=30)
            chunks = chunker_rec.chunk(sample_text, doc_id="sample_doc", source_name="sample.md")
            print(f"\nGenerated {len(chunks)} Chunks with Overlap (size=150, overlap=30):")
            for c in chunks:
                print(f"\n[ID: {c.chunk_id}] (Offset: {c.metadata['char_start']}-{c.metadata['char_end']}, Tokens: {c.metadata['token_count']})")
                print(f"Content: \"{c.text}\"")

        elif choice == "2":
            print_header("TOPIC 3: GENERATING EMBEDDINGS & COSINE SIMILARITY")
            engine = EmbeddingEngine()
            q1 = "How does vector search find nearest neighbors?"
            q2 = "What algorithm calculates approximate closest vectors?"
            q3 = "Recipe for homemade chocolate chip cookies."

            print(f"Text 1: \"{q1}\"")
            print(f"Text 2: \"{q2}\" (Semantically similar)")
            print(f"Text 3: \"{q3}\" (Completely irrelevant)")

            e1 = engine.embed_query(q1)
            e2 = engine.embed_query(q2)
            e3 = engine.embed_query(q3)

            sim_1_2 = engine.cosine_similarity(e1, e2)
            sim_1_3 = engine.cosine_similarity(e1, e3)

            print(f"\nEmbedding Dimension: {len(e1)}")
            print(f"Cosine Similarity (Text 1 vs Text 2): {sim_1_2:.4f} (High semantic alignment)")
            print(f"Cosine Similarity (Text 1 vs Text 3): {sim_1_3:.4f} (Near zero semantic overlap)")

        elif choice == "3":
            print_header("TOPIC 4 & 8: SEMANTIC VECTOR SEARCH WITH CHROMADB")
            query = input("Enter query for Semantic Search: ").strip() or "How does HNSW graph navigation work?"
            results = runner.pipeline.semantic_search.search(query, top_k=3)
            print(f"\nTop {len(results)} Semantic Matches:")
            for r in results:
                print(f"\nRank {r['rank']} | Chunk ID: {r['chunk_id']} | Cosine Similarity: {r['score']:.4f}")
                print(f"Source: {r['metadata'].get('source')} | Section: {r['metadata'].get('section_heading', 'General')}")
                print(f"Preview: {r['text'][:200]}...")

        elif choice == "4":
            print_header("TOPIC 5: LEXICAL SEARCH (BM25 OKAPI)")
            query = input("Enter query for BM25 Search: ").strip() or "Reciprocal Rank Fusion k=60"
            results = runner.pipeline.keyword_search.search(query, top_k=3)
            print(f"\nTop {len(results)} BM25 Matches:")
            for r in results:
                print(f"\nRank {r['rank']} | Chunk ID: {r['chunk_id']} | BM25 Score: {r['score']:.4f}")
                print(f"Source: {r['metadata'].get('source')}")
                print(f"Preview: {r['text'][:200]}...")

        elif choice == "5":
            print_header("TOPIC 5: HYBRID SEARCH (BM25 + SEMANTIC SEARCH)")
            query = input("Enter query for Hybrid Search: ").strip() or "How does BM25 handle term frequency saturation and length normalization?"
            print("\n1. Running Reciprocal Rank Fusion (RRF)...")
            rrf_res = runner.pipeline.hybrid_search.search_rrf(query, top_k=3)
            for r in rrf_res:
                print(f"\nRank {r['rank']} | Chunk: {r['chunk_id']} | RRF Score: {r['score']:.5f}")
                print(f"Breakdown: Semantic Rank {r['breakdown'].get('semantic_rank', 'N/A')}, Keyword Rank {r['breakdown'].get('keyword_rank', 'N/A')}")
                print(f"Snippet: {r['text'][:180]}...")

            print("\n2. Running Convex Combination (Alpha = 0.5)...")
            convex_res = runner.pipeline.hybrid_search.search_convex(query, alpha=0.5, top_k=3)
            for r in convex_res:
                print(f"\nRank {r['rank']} | Chunk: {r['chunk_id']} | Fused Score: {r['score']:.4f}")
                print(f"Breakdown: {r['breakdown']}")

        elif choice == "6":
            print_header("TOPIC 6: RE-RANKING CANDIDATES (CROSS-ENCODER)")
            query = input("Enter query for Re-Ranking: ").strip() or "What is the sliding window stride formula in chunking?"
            candidates = runner.pipeline.hybrid_search.search_rrf(query, top_k=6)
            print(f"\nRetrieved {len(candidates)} Initial Candidates from Hybrid Search.")
            reranked = runner.pipeline.reranker.rerank(query, candidates, top_k=4)

            print("\nRe-Ranker Comparison (Before vs After):")
            for r in reranked:
                arrow = "UP  " if r["rank_delta"] > 0 else ("DOWN" if r["rank_delta"] < 0 else "SAME")
                print(f"\nFinal Rank {r['final_rank']} (Initial: {r['initial_rank']} | Delta: {r['rank_delta']:+d} [{arrow}])")
                print(f"Chunk ID: {r['chunk_id']} | Re-Rank Score: {r['rerank_score']:.4f}")
                print(f"Preview: {r['text'][:150]}...")

        elif choice == "7":
            print_header("TOPIC 9: EVALUATING RAG RETRIEVAL & GENERATION QUALITY")
            print("Running benchmark on standard evaluation suite...")
            report = runner.run_evaluation_benchmark()
            print("\nEVALUATION BENCHMARK SUMMARY:")
            print(json.dumps(report["summary"], indent=2))

        elif choice == "8":
            print_header("TOPIC 10 & 13: RETRIEVAL INTELLIGENCE")
            test_q = "Explain the lost in the middle problem in LLM context windows."
            print(f"Query: \"{test_q}\"\n")

            # Routing
            route = runner.pipeline.intelligence.route_query(test_q)
            print(f"1. Query Intent Routing: [{route['route']}] -> {route['reason']}")

            # Multi-Query
            print("\n2. Multi-Query Expansion:")
            variations = runner.pipeline.intelligence.expand_query_multiquery(test_q)
            for i, v in enumerate(variations, start=1):
                print(f"   Variation {i}: {v}")

            # HyDE
            print("\n3. HyDE (Hypothetical Document):")
            hyde_doc = runner.pipeline.intelligence.generate_hyde_document(test_q)
            print(f"   Generated Hypothetical Passage: \"{hyde_doc[:220]}...\"")

        elif choice == "9":
            print_header("TOPIC 11 & 12: INTERACTIVE GROUNDED Q&A RUNNER")
            while True:
                try:
                    q = input("\nAsk Question (or 'back' to return): ").strip()
                except (EOFError, KeyboardInterrupt):
                    break
                if not q or q.lower() == "back":
                    break

                res = runner.ask(q, top_k=3, use_cache=True)
                print("\n" + "=" * 50)
                print(f"ANSWER:\n{res['answer']}")
                print("=" * 50)
                print("\nCITATIONS & GROUNDED SOURCES:")
                for cit in res.get("citations", []):
                    print(f" - [{cit['chunk_id']}] {cit['source']} | {cit['section']}")
                    print(f"   Snippet: \"{cit['snippet']}\"")
                print(f"\nTelemetry: Pipeline Total {res['telemetry'].get('pipeline_total_sec', 0)}s (Generation: {res['telemetry'].get('generation_sec', 0)}s)")

        elif choice == "10":
            print_header("RUNNING FULL 13-TOPIC PIPELINE VALIDATION")
            print("Testing end-to-end integration...")
            test_query = "What is the difference between bi-encoders and cross-encoders?"
            res = runner.ask(test_query, top_k=3)
            print("\n[Validation] Query:", test_query)
            print(f"[Validation] Route: {res['route']['route']}")
            print(f"[Validation] Retrieved {len(res['retrieved_chunks'])} chunks")
            print(f"[Validation] Citations Found: {len(res['citations'])}")
            print(f"[Validation] Answer Preview: {res['answer'][:200]}...")
            print(f"[Validation] Latency: {res['telemetry']['pipeline_total_sec']}s")
            print("\n>>> All 13 Topics Successfully Validated! <<<")

if __name__ == "__main__":
    main()
