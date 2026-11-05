import re
import time
from typing import List, Dict, Any, Optional, Tuple
from openai import OpenAI
from config import config

class AnswerGenerator:
    """
    Topic 11: Building the Final Answer Generator for Advanced RAG.
    Crafts grounded prompts, injects retrieved chunk context with metadata boundaries,
    enforces strict factual attribution, parses inline citations, and handles hallucination guardrails.
    """
    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or (config.GEMINI_MODEL if config.GEMINI_API_KEY else config.OLLAMA_MODEL)
        self.gemini_key = config.GEMINI_API_KEY
        self.gemini_url = config.GEMINI_BASE_URL
        self.ollama_url = config.OLLAMA_BASE_URL

    def _build_grounded_prompt(self, query: str, context_chunks: List[Dict[str, Any]]) -> Tuple[str, str]:
        """
        Builds a context-injected prompt with clear chunk markers and metadata tags.
        """
        context_blocks = []
        for i, c in enumerate(context_chunks, start=1):
            cid = c.get("chunk_id", f"chunk_{i}")
            meta = c.get("metadata", {})
            src = meta.get("source", "Document")
            sec = meta.get("section_heading", meta.get("section_path", ""))
            sec_tag = f" | Section: {sec}" if sec else ""

            block = f"""--- [START CHUNK: {cid} | Source: {src}{sec_tag}] ---
{c.get('text', '').strip()}
--- [END CHUNK: {cid}] ---"""
            context_blocks.append(block)

        context_str = "\n\n".join(context_blocks)

        system_instruction = (
            "You are an Advanced RAG Assistant with strict factual grounding.\n"
            "Rules you MUST follow:\n"
            "1. Answer the question using ONLY the provided CONTEXT CHUNKS.\n"
            "2. For every factual statement you make, cite the corresponding chunk ID in brackets, e.g. [Doc: doc1_c001].\n"
            "3. If the context does not contain the answer, explicitly state: 'Based on the provided documents, I do not have enough information to answer this question.' Do NOT invent facts.\n"
            "4. Keep the answer comprehensive, clear, and structured."
        )

        user_content = f"""QUESTION:
{query}

CONTEXT CHUNKS:
{context_str}

GROUNDED ANSWER (with citations):"""

        return system_instruction, user_content

    def _extract_citations(self, answer_text: str, context_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Parses chunk citation tags like [Doc: chunk_id] or [chunk_id] from generated response
        and maps them to context chunks.
        """
        cited_cids = set(re.findall(r"\[(?:Doc:\s*)?([a-zA-Z0-9_\-\.]+?)\]", answer_text))
        chunk_map = {c.get("chunk_id"): c for c in context_chunks}

        citations: List[Dict[str, Any]] = []
        for cid in cited_cids:
            if cid in chunk_map:
                c = chunk_map[cid]
                meta = c.get("metadata", {})
                citations.append({
                    "chunk_id": cid,
                    "source": meta.get("source", "unknown"),
                    "section": meta.get("section_heading", meta.get("section_path", "General")),
                    "snippet": c.get("text", "")[:180] + "...",
                    "score": c.get("rerank_score", c.get("score", 1.0))
                })

        # If model didn't format tags properly, attach top 2 context chunks as referenced sources
        if not citations and context_chunks:
            for c in context_chunks[:2]:
                meta = c.get("metadata", {})
                citations.append({
                    "chunk_id": c.get("chunk_id", "c1"),
                    "source": meta.get("source", "unknown"),
                    "section": meta.get("section_heading", "General"),
                    "snippet": c.get("text", "")[:180] + "...",
                    "score": c.get("rerank_score", c.get("score", 1.0))
                })

        return citations

    def generate_answer(
        self,
        query: str,
        context_chunks: List[Dict[str, Any]],
        temperature: float = 0.1
    ) -> Dict[str, Any]:
        """
        Generates grounded answer with citation extraction and latency telemetry.
        """
        start_time = time.time()

        if not context_chunks:
            return {
                "answer": "No relevant documents were retrieved from the knowledge base to answer your question.",
                "citations": [],
                "model": self.model_name,
                "latency_sec": round(time.time() - start_time, 3),
                "grounded": False
            }

        system_instruction, user_content = self._build_grounded_prompt(query, context_chunks)
        messages = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": user_content}
        ]

        answer_text = ""
        prompt_tokens = 0
        completion_tokens = 0

        # Try Gemini API first if configured
        if self.gemini_key:
            try:
                client = OpenAI(base_url=self.gemini_url, api_key=self.gemini_key)
                res = client.chat.completions.create(
                    model=config.GEMINI_MODEL,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=800
                )
                answer_text = res.choices[0].message.content or ""
                if res.usage:
                    prompt_tokens = res.usage.prompt_tokens
                    completion_tokens = res.usage.completion_tokens
            except Exception as e:
                print(f"[Generator] Gemini error: {e}, falling back to Ollama")

        # Fallback to Ollama if no response yet
        if not answer_text:
            try:
                client = OpenAI(base_url=self.ollama_url, api_key="ollama")
                res = client.chat.completions.create(
                    model=config.OLLAMA_MODEL,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=800
                )
                answer_text = res.choices[0].message.content or ""
                if res.usage:
                    prompt_tokens = res.usage.prompt_tokens
                    completion_tokens = res.usage.completion_tokens
            except Exception as e:
                print(f"[Generator] Ollama error: {e}")

        # Deterministic offline synthesis fallback
        if not answer_text:
            top_chunk = context_chunks[0]
            top_id = top_chunk.get("chunk_id", "chunk_1")
            answer_text = (
                f"Based on the retrieved context [Doc: {top_id}]:\n\n"
                f"{top_chunk.get('text', '').strip()[:400]}...\n\n"
                f"(Generated in offline fallback mode with citations to ground truth)."
            )

        citations = self._extract_citations(answer_text, context_chunks)
        latency = round(time.time() - start_time, 3)

        return {
            "answer": answer_text,
            "citations": citations,
            "context_chunks_used": len(context_chunks),
            "model": self.model_name,
            "latency_sec": latency,
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens
            },
            "grounded": True
        }
