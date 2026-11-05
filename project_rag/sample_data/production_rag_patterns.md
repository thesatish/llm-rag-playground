# Production RAG Patterns and Retrieval Intelligence

## Intelligent Pre-Retrieval Strategies
Modern enterprise RAG systems do not blindly send raw user queries to vector stores. Instead, they apply intelligent pre-retrieval routing:
1. Intent Classification and Routing: Determining whether an incoming prompt is conversational chitchat (which can bypass retrieval entirely to save cost and latency), an exact identifier lookup, or an in-depth semantic research query.
2. Multi-Query Expansion: Prompting an LLM to generate multiple synonymous query formulations. This hedges against query phrasing bias and retrieves candidate chunks across diverse angles in vector space.
3. Hypothetical Document Embeddings (HyDE): In situations where the user query is brief or phrased as a question, query embeddings often reside in a different vector distribution than declarative answer passages. HyDE generates a synthetic hypothetical answer, embeds that answer, and performs doc-to-doc semantic retrieval.

## Corrective RAG (CRAG) and Self-Correction
Retrieved candidates can still contain distractor documents or irrelevant noise that degrades generator focus. Corrective RAG (CRAG) introduces an internal evaluation step:
- Chunks evaluated as confidently relevant are retained.
- Ambiguous or noise chunks are pruned.
- If all retrieved chunks are judged irrelevant, the system triggers a fallback web search or emits a calibrated refusal rather than hallucinating an ungrounded answer.

## Semantic Caching for Zero-Latency Retrieval
Enterprise applications often experience repeated queries or paraphrased questions with identical semantic intent. By caching query embeddings and associated generation payloads, subsequent queries with cosine similarity above a threshold (typically >= 0.95) are served instantly from cache, saving hundreds of milliseconds of compute time and token costs.

