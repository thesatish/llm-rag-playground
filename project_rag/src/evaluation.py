import re
import math
from typing import List, Dict, Any, Optional

class RAGEvaluator:
    """
    Topic 9: Evaluating and Improving RAG Retrieval Quality.
    Implements standard Information Retrieval (IR) and RAG generation metrics:
      - Hit Rate @ K
      - Mean Reciprocal Rank (MRR @ K)
      - NDCG @ K (Normalized Discounted Cumulative Gain)
      - The RAG Triad: Context Relevance, Faithfulness/Groundedness, Answer Relevance
    """

    @staticmethod
    def calculate_hit_rate(retrieved_ids: List[str], ground_truth_id: str, k: int = 5) -> float:
        """
        Calculates Hit Rate @ K: 1.0 if ground_truth_id in top-K, 0.0 otherwise.
        """
        top_k_ids = retrieved_ids[:k]
        return 1.0 if ground_truth_id in top_k_ids else 0.0

    @staticmethod
    def calculate_mrr(retrieved_ids: List[str], ground_truth_id: str, k: int = 5) -> float:
        """
        Calculates Reciprocal Rank @ K: 1 / rank of first relevant result (or 0 if not found).
        """
        for rank, cid in enumerate(retrieved_ids[:k], start=1):
            if cid == ground_truth_id:
                return 1.0 / rank
        return 0.0

    @staticmethod
    def calculate_ndcg(retrieved_ids: List[str], ground_truth_id: str, k: int = 5) -> float:
        """
        Calculates Normalized Discounted Cumulative Gain (NDCG @ K) for binary relevance.
        """
        top_k = retrieved_ids[:k]
        dcg = 0.0
        for i, cid in enumerate(top_k, start=1):
            rel = 1.0 if cid == ground_truth_id else 0.0
            if rel > 0:
                dcg += (2.0 ** rel - 1.0) / math.log2(i + 1)

        idcg = (2.0 ** 1.0 - 1.0) / math.log2(1 + 1)  # Ideal rank 1
        return dcg / idcg if idcg > 0 else 0.0

    @staticmethod
    def evaluate_context_relevance(query: str, context_chunks: List[Dict[str, Any]]) -> float:
        """
        RAG Triad Metric 1: Context Relevance.
        Evaluates the proportion of sentences in the context that share semantic overlap with query.
        """
        if not context_chunks:
            return 0.0

        q_terms = set(re.findall(r"\b\w{3,}\b", query.lower()))
        if not q_terms:
            return 1.0

        relevant_chunks = 0
        for c in context_chunks:
            c_text = c.get("text", "").lower()
            overlap = sum(1 for t in q_terms if t in c_text)
            if overlap >= min(2, len(q_terms)):
                relevant_chunks += 1

        return round(relevant_chunks / len(context_chunks), 3)

    @staticmethod
    def evaluate_faithfulness(answer: str, context_chunks: List[Dict[str, Any]]) -> float:
        """
        RAG Triad Metric 2: Groundedness / Faithfulness.
        Evaluates whether factual claims in answer are substantiated by context.
        Zero hallucination check.
        """
        if not answer or not context_chunks:
            return 0.0

        combined_context = " ".join([c.get("text", "") for c in context_chunks]).lower()
        sentences = [s.strip() for s in re.split(r"[.!?]\s+", answer) if len(s.strip()) > 15]
        if not sentences:
            return 1.0

        grounded_sentences = 0
        for s in sentences:
            s_words = set(re.findall(r"\b\w{4,}\b", s.lower()))
            if not s_words:
                grounded_sentences += 1
                continue
            matched_words = sum(1 for w in s_words if w in combined_context)
            ratio = matched_words / len(s_words)
            if ratio >= 0.5:
                grounded_sentences += 1

        return round(grounded_sentences / len(sentences), 3)

    @staticmethod
    def evaluate_answer_relevance(query: str, answer: str) -> float:
        """
        RAG Triad Metric 3: Answer Relevance.
        Checks if the generated answer directly addresses the terms in the query.
        """
        q_words = set(re.findall(r"\b\w{3,}\b", query.lower()))
        if not q_words:
            return 1.0
        ans_lower = answer.lower()
        matched = sum(1 for w in q_words if w in ans_lower)
        return round(min(1.0, matched / max(1, len(q_words) * 0.7)), 3)

    def evaluate_batch(self, test_cases: List[Dict[str, Any]], pipeline: Any) -> Dict[str, Any]:
        """
        Runs comprehensive evaluation over a test dataset:
        Each test_case: {"query": str, "expected_chunk_id": str}
        """
        hit_rates = []
        mrrs = []
        ndcgs = []
        context_rels = []
        faithfulnesses = []
        ans_rels = []

        details = []

        for tc in test_cases:
            q = tc["query"]
            target_id = tc["expected_chunk_id"]

            res = pipeline.run(q, top_k=5, use_hyde=False, use_multiquery=False)
            retrieved_ids = [c["chunk_id"] for c in res["retrieved_chunks"]]

            hr = self.calculate_hit_rate(retrieved_ids, target_id, k=5)
            mrr = self.calculate_mrr(retrieved_ids, target_id, k=5)
            ndcg = self.calculate_ndcg(retrieved_ids, target_id, k=5)
            cr = self.evaluate_context_relevance(q, res["retrieved_chunks"])
            faith = self.evaluate_faithfulness(res["answer"], res["retrieved_chunks"])
            ar = self.evaluate_answer_relevance(q, res["answer"])

            hit_rates.append(hr)
            mrrs.append(mrr)
            ndcgs.append(ndcg)
            context_rels.append(cr)
            faithfulnesses.append(faith)
            ans_rels.append(ar)

            details.append({
                "query": q,
                "target_id": target_id,
                "retrieved_ids": retrieved_ids,
                "hit_rate@5": hr,
                "mrr@5": mrr,
                "ndcg@5": ndcg,
                "context_relevance": cr,
                "faithfulness": faith,
                "answer_relevance": ar
            })

        n = max(1, len(test_cases))
        return {
            "summary": {
                "mean_hit_rate@5": round(sum(hit_rates) / n, 4),
                "mean_mrr@5": round(sum(mrrs) / n, 4),
                "mean_ndcg@5": round(sum(ndcgs) / n, 4),
                "mean_context_relevance": round(sum(context_rels) / n, 4),
                "mean_faithfulness": round(sum(faithfulnesses) / n, 4),
                "mean_answer_relevance": round(sum(ans_rels) / n, 4),
                "total_queries": len(test_cases)
            },
            "queries": details
        }

