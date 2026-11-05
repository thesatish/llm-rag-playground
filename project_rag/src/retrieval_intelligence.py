import re
from typing import List, Dict, Any, Optional, Tuple
from openai import OpenAI
from config import config
from .embeddings import EmbeddingEngine

class RetrievalIntelligence:
    """
    Topic 13: Building Retrieval Intelligence with Advanced RAG.
    Implements intelligent pre-retrieval and post-retrieval capabilities:
    1. Query Intent Routing
    2. Multi-Query Expansion
    3. HyDE (Hypothetical Document Embeddings)
    4. CRAG (Corrective RAG) Context Relevance Grader
    5. Semantic Query Cache
    """
    def __init__(self, embedding_engine: EmbeddingEngine):
        self.embedding_engine = embedding_engine
        self.gemini_key = config.GEMINI_API_KEY
        self.gemini_url = config.GEMINI_BASE_URL
        self.ollama_url = config.OLLAMA_BASE_URL
        self.model = config.GEMINI_MODEL if config.GEMINI_API_KEY else config.OLLAMA_MODEL

        # Semantic Cache: list of (query_text, query_emb, answer_payload)
        self.semantic_cache: List[Tuple[str, List[float], Dict[str, Any]]] = []
        self.cache_threshold = 0.95

    def _call_llm(self, prompt: str, system: str = "", max_tokens: int = 256) -> str:
        """Helper to invoke configured LLM with fallbacks."""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        # Try Gemini first if key available
        if self.gemini_key:
            try:
                client = OpenAI(base_url=self.gemini_url, api_key=self.gemini_key)
                res = client.chat.completions.create(
                    model=config.GEMINI_MODEL,
                    messages=messages,
                    temperature=0.2,
                    max_tokens=max_tokens
                )
                return res.choices[0].message.content or ""
            except Exception:
                pass

        # Try local Ollama
        try:
            client = OpenAI(base_url=self.ollama_url, api_key="ollama")
            res = client.chat.completions.create(
                model=config.OLLAMA_MODEL,
                messages=messages,
                temperature=0.2,
                max_tokens=max_tokens
            )
            return res.choices[0].message.content or ""
        except Exception:
            return ""

    def route_query(self, query: str) -> Dict[str, Any]:
        """
        Classifies query intent into:
        - 'DIRECT_LLM': conversational chitchat (hi, hello there, how are you, who are you)
        - 'HYBRID_KEYWORD_HEAVY': exact symbols, IDs, error codes
        - 'HYBRID_RAG': default for knowledge-intensive queries
        """
        lower = query.strip().lower()
        chitchat_pattern = r"^(hi|hello|hey|good morning|good evening|who are you|how are you|thanks|thank you)(\s+(there|friend|bot|assistant|everyone))?[\.!\?]*$"
        if re.match(chitchat_pattern, lower) or lower in ["hi", "hello", "hey", "hello there", "hey there", "how are you?"]:
            return {
                "route": "DIRECT_LLM",
                "reason": "Conversational chitchat detected; bypassing vector retrieval."
            }

        # Check for error codes, specific alphanumeric symbols, or exact quotes
        if re.search(r'("[^"]+")|(\b[A-Z0-9_\-\.]{8,}\b)|(error\s+[0-9]+)', query):
            return {
                "route": "HYBRID_KEYWORD_HEAVY",
                "reason": "Exact codes or quoted identifiers detected; boosting lexical BM25 retrieval."
            }

        return {
            "route": "HYBRID_RAG",
            "reason": "Knowledge-intensive semantic query; invoking full hybrid pipeline."
        }

    def expand_query_multiquery(self, query: str, num_queries: int = 3) -> List[str]:
        """
        Generates alternative formulations of user query to search from multiple perspectives.
        """
        prompt = f"""Generate {num_queries} distinct search query variations for the following user question.
Ensure they capture synonyms, technical terminology, and broader context.
Question: "{query}"

Output ONLY numbered lines like:
1. variation one
2. variation two
3. variation three
"""
        response = self._call_llm(prompt, max_tokens=150)
        variations = [query]
        for line in response.splitlines():
            clean = re.sub(r"^\d+[\.\)]\s*", "", line).strip().strip('"')
            if clean and clean.lower() != query.lower():
                variations.append(clean)

        return variations[:num_queries + 1]

    def generate_hyde_document(self, query: str) -> str:
        """
        Hypothetical Document Embeddings (HyDE):
        Generates a synthetic hypothetical answer document to bridge question-document embedding mismatch.
        """
        prompt = f"""Write a concise, authoritative one-paragraph technical explanation that directly answers this question:
Question: "{query}"

Hypothetical Answer Passage:"""
        hypothetical = self._call_llm(prompt, max_tokens=200)
        if not hypothetical.strip():
            return query
        return hypothetical.strip()

    def grade_documents_crag(self, query: str, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Corrective RAG (CRAG) Document Grader:
        Evaluates retrieved chunks and filters out completely irrelevant passages.
        """
        filtered = []
        for c in chunks:
            # Quick heuristic check first
            text = c.get("text", "")
            q_words = set(re.findall(r"\b\w{3,}\b", query.lower()))
            t_words = set(re.findall(r"\b\w{3,}\b", text.lower()))

            overlap = len(q_words.intersection(t_words))
            # If re-rank score is above 0.3 or word overlap is high, keep chunk
            score = c.get("rerank_score", c.get("score", 0.5))
            if score >= 0.25 or overlap >= 1:
                c["crag_grade"] = "RELEVANT"
                filtered.append(c)
            else:
                c["crag_grade"] = "FILTERED_NOISE"

        return filtered if filtered else chunks[:2]

    def check_semantic_cache(self, query: str) -> Optional[Dict[str, Any]]:
        """Checks if a semantically equivalent query was recently answered."""
        q_emb = self.embedding_engine.embed_query(query)
        for cached_query, cached_emb, cached_payload in self.semantic_cache:
            sim = self.embedding_engine.cosine_similarity(q_emb, cached_emb)
            if sim >= self.cache_threshold:
                result = cached_payload.copy()
                result["cache_hit"] = True
                result["matched_query"] = cached_query
                result["cache_similarity"] = round(sim, 4)
                return result
        return None

    def store_in_semantic_cache(self, query: str, payload: Dict[str, Any]):
        """Caches query embedding and response."""
        q_emb = self.embedding_engine.embed_query(query)
        # Limit cache size to 100 entries (FIFO)
        if len(self.semantic_cache) >= 100:
            self.semantic_cache.pop(0)
        self.semantic_cache.append((query, q_emb, payload))
