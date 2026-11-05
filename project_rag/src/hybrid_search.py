from typing import List, Dict, Any, Optional
from collections import defaultdict
from config import config
from .semantic_search import SemanticSearchEngine
from .keyword_search import BM25SearchEngine

class HybridSearchEngine:
    """
    Topic 5: Building Hybrid Search: Combining Keyword and Semantic Retrieval.
    Combines dense semantic retrieval (ChromaDB) and sparse lexical retrieval (BM25)
    using Reciprocal Rank Fusion (RRF) and Convex Linear Combination (Alpha-weighted).
    """
    def __init__(
        self,
        semantic_engine: SemanticSearchEngine,
        keyword_engine: BM25SearchEngine,
        default_alpha: float = 0.5,
        rrf_k: int = 60
    ):
        self.semantic_engine = semantic_engine
        self.keyword_engine = keyword_engine
        self.default_alpha = default_alpha
        self.rrf_k = rrf_k

    @staticmethod
    def _normalize_scores(results: List[Dict[str, Any]]) -> Dict[str, float]:
        """Min-Max normalizes scores into [0, 1] range."""
        if not results:
            return {}
        scores = [r["score"] for r in results]
        min_s = min(scores)
        max_s = max(scores)
        if max_s == min_s:
            return {r["chunk_id"]: 1.0 for r in results}
        return {r["chunk_id"]: (r["score"] - min_s) / (max_s - min_s) for r in results}

    def search_rrf(
        self,
        query: str,
        top_k: int = 5,
        fetch_k: int = 15,
        where_filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Reciprocal Rank Fusion (RRF):
        RRF_Score(d) = 1/(k + rank_dense) + 1/(k + rank_sparse)
        Robust against differing score scales and distributions.
        """
        semantic_results = self.semantic_engine.search(query, top_k=fetch_k, where_filter=where_filter)
        keyword_results = self.keyword_engine.search(query, top_k=fetch_k)

        all_docs: Dict[str, Dict[str, Any]] = {}
        rrf_scores: Dict[str, float] = defaultdict(float)
        rank_breakdown: Dict[str, Dict[str, Any]] = defaultdict(dict)

        # 1. Process Semantic Results
        for item in semantic_results:
            cid = item["chunk_id"]
            all_docs[cid] = item
            rank = item["rank"]
            contribution = 1.0 / (self.rrf_k + rank)
            rrf_scores[cid] += contribution
            rank_breakdown[cid]["semantic_rank"] = rank
            rank_breakdown[cid]["semantic_score"] = round(item["score"], 4)

        # 2. Process Keyword Results
        for item in keyword_results:
            cid = item["chunk_id"]
            if cid not in all_docs:
                all_docs[cid] = item
            rank = item["rank"]
            contribution = 1.0 / (self.rrf_k + rank)
            rrf_scores[cid] += contribution
            rank_breakdown[cid]["keyword_rank"] = rank
            rank_breakdown[cid]["keyword_score"] = round(item["score"], 4)

        sorted_cids = sorted(rrf_scores.keys(), key=lambda c: rrf_scores[c], reverse=True)[:top_k]

        final_results: List[Dict[str, Any]] = []
        for rank, cid in enumerate(sorted_cids, start=1):
            doc = all_docs[cid].copy()
            doc["score"] = float(rrf_scores[cid])
            doc["rank"] = rank
            doc["retrieval_type"] = "hybrid_rrf"
            doc["breakdown"] = rank_breakdown[cid]
            final_results.append(doc)

        return final_results

    def search_convex(
        self,
        query: str,
        alpha: Optional[float] = None,
        top_k: int = 5,
        fetch_k: int = 15,
        where_filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Convex Linear Combination:
        Score = alpha * DenseNorm + (1 - alpha) * SparseNorm
        alpha = 1.0 => Pure Dense Semantic Search
        alpha = 0.0 => Pure Lexical BM25 Search
        alpha = 0.5 => Balanced Hybrid
        """
        if alpha is None:
            alpha = self.default_alpha
        alpha = max(0.0, min(1.0, alpha))

        semantic_results = self.semantic_engine.search(query, top_k=fetch_k, where_filter=where_filter)
        keyword_results = self.keyword_engine.search(query, top_k=fetch_k)

        all_docs: Dict[str, Dict[str, Any]] = {}
        for r in semantic_results:
            all_docs[r["chunk_id"]] = r
        for r in keyword_results:
            if r["chunk_id"] not in all_docs:
                all_docs[r["chunk_id"]] = r

        norm_semantic = self._normalize_scores(semantic_results)
        norm_keyword = self._normalize_scores(keyword_results)

        combined_scores: Dict[str, float] = {}
        breakdown: Dict[str, Dict[str, Any]] = {}

        for cid in all_docs:
            s_sem = norm_semantic.get(cid, 0.0)
            s_key = norm_keyword.get(cid, 0.0)
            fused = (alpha * s_sem) + ((1.0 - alpha) * s_key)
            combined_scores[cid] = fused
            breakdown[cid] = {
                "alpha": alpha,
                "norm_semantic": round(s_sem, 4),
                "norm_keyword": round(s_key, 4),
                "fused_score": round(fused, 4)
            }

        sorted_cids = sorted(combined_scores.keys(), key=lambda c: combined_scores[c], reverse=True)[:top_k]

        final_results: List[Dict[str, Any]] = []
        for rank, cid in enumerate(sorted_cids, start=1):
            doc = all_docs[cid].copy()
            doc["score"] = float(combined_scores[cid])
            doc["rank"] = rank
            doc["retrieval_type"] = f"hybrid_convex_alpha_{alpha:.2f}"
            doc["breakdown"] = breakdown[cid]
            final_results.append(doc)

        return final_results

