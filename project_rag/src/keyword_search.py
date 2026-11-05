import re
import math
from typing import List, Dict, Any, Optional
from collections import Counter, defaultdict
from .chunking import Chunk

class BM25SearchEngine:
    """
    Topic 5: BM25 (Best Matching 25) Sparse Keyword Retrieval Engine.
    Implements the Okapi BM25 algorithm for lexical search with inverted indexing,
    IDF calculation, and document length normalization.
    """
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus: List[Dict[str, Any]] = []
        self.doc_len: List[int] = []
        self.avg_doc_len: float = 0.0
        self.doc_count: int = 0
        self.inverted_index: Dict[str, Dict[int, int]] = defaultdict(dict)
        self.idf: Dict[str, float] = {}

    @staticmethod
    def tokenize(text: str) -> List[str]:
        """Lowercases and extracts alphanumeric word tokens."""
        return re.findall(r"\b[a-zA-Z0-9_\-\.]+\b", text.lower())

    def index_documents(self, chunks: List[Chunk]):
        """Builds inverted index and precomputes IDF for all terms."""
        self.corpus = [c.to_dict() for c in chunks]
        self.doc_count = len(chunks)
        if self.doc_count == 0:
            return

        self.doc_len = []
        self.inverted_index = defaultdict(dict)
        term_doc_frequencies: Dict[str, int] = defaultdict(int)

        for doc_idx, chunk in enumerate(chunks):
            tokens = self.tokenize(chunk.text)
            self.doc_len.append(len(tokens))
            term_counts = Counter(tokens)

            for term, count in term_counts.items():
                self.inverted_index[term][doc_idx] = count
                term_doc_frequencies[term] += 1

        self.avg_doc_len = sum(self.doc_len) / float(self.doc_count) if self.doc_count > 0 else 0.0

        # Calculate BM25 IDF: ln((N - n + 0.5) / (n + 0.5) + 1)
        self.idf = {}
        for term, df in term_doc_frequencies.items():
            idf_val = math.log((self.doc_count - df + 0.5) / (df + 0.5) + 1.0)
            self.idf[term] = max(0.0001, idf_val)

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Calculates BM25 score for all candidate documents matching query terms.
        """
        if self.doc_count == 0 or not query.strip():
            return []

        query_tokens = self.tokenize(query)
        if not query_tokens:
            return []

        scores: Dict[int, float] = defaultdict(float)

        for token in query_tokens:
            if token not in self.inverted_index:
                continue

            idf = self.idf.get(token, 0.0)
            postings = self.inverted_index[token]

            for doc_idx, tf in postings.items():
                dl = self.doc_len[doc_idx]
                numerator = tf * (self.k1 + 1.0)
                denominator = tf + self.k1 * (1.0 - self.b + self.b * (dl / self.avg_doc_len if self.avg_doc_len > 0 else 1.0))
                scores[doc_idx] += idf * (numerator / denominator)

        if not scores:
            return []

        # Sort documents by BM25 score descending
        sorted_indices = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_k]

        results: List[Dict[str, Any]] = []
        for rank, (idx, score) in enumerate(sorted_indices, start=1):
            doc = self.corpus[idx]
            results.append({
                "chunk_id": doc["chunk_id"],
                "text": doc["text"],
                "metadata": doc["metadata"],
                "score": float(score),
                "rank": rank,
                "retrieval_type": "keyword_bm25"
            })

        return results

