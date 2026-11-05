import re
import math
from typing import List, Dict, Any, Optional
from openai import OpenAI
from config import config

class ReRanker:
    """
    Topic 6: Understanding Re-Ranking: Finding the Most Relevant Results.
    Two-stage retrieval:
      Stage 1 -> High Recall (Retrieve candidate pool of ~10-30 chunks via Hybrid Search)
      Stage 2 -> High Precision (Re-score candidate pool using deep cross-attention / cross-interaction)
    """
    def __init__(
        self,
        strategy: str = "cross_scorer",
        llm_model: Optional[str] = None
    ):
        self.strategy = strategy
        self.llm_model = llm_model or (config.GEMINI_MODEL if config.GEMINI_API_KEY else config.OLLAMA_MODEL)
        self.gemini_key = config.GEMINI_API_KEY
        self.gemini_url = config.GEMINI_BASE_URL
        self.ollama_url = config.OLLAMA_BASE_URL

    def _cross_attention_heuristic(self, query: str, doc_text: str) -> float:
        """
        Simulates cross-encoder cross-attention interaction:
        Computes query-document token co-occurrence, contiguous n-gram matching,
        exact phrase alignment, and term proximity density.
        """
        q_tokens = re.findall(r"\b\w+\b", query.lower())
        d_tokens = re.findall(r"\b\w+\b", doc_text.lower())
        if not q_tokens or not d_tokens:
            return 0.0

        q_set = set(q_tokens)
        d_text_lower = doc_text.lower()
        query_lower = query.lower()

        # 1. Exact phrase occurrence bonus
        exact_match_score = 3.0 if query_lower in d_text_lower else 0.0

        # 2. Token overlap ratio
        matched_tokens = [t for t in q_tokens if t in d_tokens]
        overlap_ratio = len(matched_tokens) / len(q_tokens)

        # 3. Bigram and trigram phrase overlap
        ngram_score = 0.0
        for n in (2, 3):
            if len(q_tokens) >= n:
                for i in range(len(q_tokens) - n + 1):
                    ngram = " ".join(q_tokens[i:i+n])
                    if ngram in d_text_lower:
                        ngram_score += (1.5 * n)

        # 4. Term proximity bonus (how close query terms appear to one another)
        proximity_score = 0.0
        indices = [i for i, t in enumerate(d_tokens) if t in q_set]
        if len(indices) >= 2:
            spans = [indices[j+1] - indices[j] for j in range(len(indices)-1)]
            avg_span = sum(spans) / len(spans)
            if avg_span <= 5:
                proximity_score = 2.0
            elif avg_span <= 12:
                proximity_score = 1.0

        raw_score = (overlap_ratio * 4.0) + exact_match_score + ngram_score + proximity_score
        # Sigmoid normalization into [0, 1]
        normalized = 1.0 / (1.0 + math.exp(-raw_score / 3.0))
        return float(normalized)

    def _rerank_with_llm(self, query: str, candidates: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
        """Zero-shot LLM re-ranker as an advanced cross-encoder judge."""
        try:
            if self.gemini_key:
                client = OpenAI(base_url=self.gemini_url, api_key=self.gemini_key)
            else:
                client = OpenAI(base_url=self.ollama_url, api_key="ollama")

            prompt = f"""You are a Cross-Encoder Re-Ranking model. Evaluate the relevance of the following candidate text passages to the Query.
Query: "{query}"

Candidate Passages:
"""
            for i, c in enumerate(candidates):
                prompt += f"\n[Passage {i+1} - ID: {c['chunk_id']}]\n{c['text'][:300]}...\n"

            prompt += """
Rate each candidate on a scale of 0.0 (completely irrelevant) to 1.0 (perfectly relevant and directly answers query).
Output ONLY lines formatted like:
[Passage 1] Score: 0.95
[Passage 2] Score: 0.32
"""
            res = client.chat.completions.create(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=256
            )
            output = res.choices[0].message.content or ""
            score_map = {}
            for line in output.splitlines():
                match = re.search(r"Passage\s+(\d+).*?(\d+\.\d+)", line)
                if match:
                    idx = int(match.group(1)) - 1
                    val = float(match.group(2))
                    if 0 <= idx < len(candidates):
                        score_map[candidates[idx]["chunk_id"]] = val

            for c in candidates:
                c["rerank_score"] = score_map.get(c["chunk_id"], self._cross_attention_heuristic(query, c["text"]))

        except Exception:
            # Fallback to cross attention heuristic
            for c in candidates:
                c["rerank_score"] = self._cross_attention_heuristic(query, c["text"])

        sorted_candidates = sorted(candidates, key=lambda x: x.get("rerank_score", 0.0), reverse=True)[:top_k]
        return sorted_candidates

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int = 4
    ) -> List[Dict[str, Any]]:
        """
        Re-scores candidates and re-orders them by semantic relevance.
        Computes rank deltas (e.g., candidate moved from Rank 6 -> Rank 1).
        """
        if not candidates:
            return []

        if self.strategy == "llm":
            reranked = self._rerank_with_llm(query, candidates, top_k)
        else:
            for item in candidates:
                item["rerank_score"] = round(self._cross_attention_heuristic(query, item["text"]), 4)
            reranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)[:top_k]

        results: List[Dict[str, Any]] = []
        for new_rank, item in enumerate(reranked, start=1):
            cloned = item.copy()
            old_rank = cloned.get("rank", new_rank)
            cloned["initial_rank"] = old_rank
            cloned["final_rank"] = new_rank
            cloned["rank_delta"] = old_rank - new_rank  # Positive means moved up, negative means moved down
            cloned["stage"] = "reranked"
            results.append(cloned)

        return results

