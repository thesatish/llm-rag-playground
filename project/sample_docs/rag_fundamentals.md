# Retrieval-Augmented Generation (RAG) Overview

Retrieval-Augmented Generation (RAG) is an architectural pattern that combines external knowledge retrieval with large language models (LLMs). Rather than relying purely on static parameters learned during pretraining, RAG dynamically retrieves relevant context passages from an external vector index or knowledge store and supplies them to the prompt.

## Key Components of RAG:
1. **Document Ingestion**: Parsing text from PDFs, Markdown, and TXT files.
2. **Chunking**: Splitting raw content into manageable semantic segments with overlapping boundaries.
3. **Embedding Generation**: Converting textual chunks into dense vector representations using models such as qwen3-embedding or text-embedding-004.
4. **Vector Database**: Storing embeddings and metadata in vector indices like ChromaDB, Milvus, or FAISS.
5. **Similarity Search**: Comparing query embeddings with indexed document embeddings using cosine similarity or euclidean distance.
6. **Grounded Generation**: Augmenting the user query with the retrieved top-k chunks so the LLM produces accurate, hallucination-free responses.

