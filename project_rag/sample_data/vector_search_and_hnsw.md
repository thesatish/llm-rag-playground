# Vector Databases and Approximate Nearest Neighbor Search

## Vector Space and Distance Metrics
Vector embeddings map unstructured text into dense mathematical representations in continuous vector spaces. When vectors are L2-normalized such that their Euclidean norm equals 1.0, the inner product (dot product) becomes strictly identical to cosine similarity:
Cosine_Similarity(u, v) = (u . v) / (||u|| * ||v||).
For normalized vectors, cosine distance is related to Euclidean distance by:
Euclidean_Distance^2 = 2 * (1 - Cosine_Similarity).
ChromaDB supports multiple distance metrics via its hnsw:space metadata key: cosine, l2, and ip (inner product).

## Hierarchical Navigable Small World (HNSW) Graphs
Exact k-nearest neighbor (k-NN) search requires computing vector distances against every item in the corpus, resulting in O(N) complexity which becomes prohibitive at scale.

Hierarchical Navigable Small World (HNSW) is a state-of-the-art graph-based algorithm for Approximate Nearest Neighbor (ANN) search. It organizes vectors into multi-layer hierarchical graphs reminiscent of a skip list:
- Top layers contain sparse graphs with long-range edges for high-speed logarithmic navigation across broad semantic regions.
- Bottom layers contain dense graphs with short-range edges for fine-grained local clustering and clustering precision.
- Key tuning parameters:
  - M: The maximum number of bidirectional connection links per node. Higher M increases recall and graph quality at the expense of memory.
  - efConstruction: The size of the dynamic candidate list during graph construction. Higher values produce higher recall graphs.
  - efSearch: The size of the dynamic candidate list evaluated during query execution. Allows dynamic trade-offs between latency and recall at query time.

## ChromaDB Storage Architecture
ChromaDB organizes data into collections with persistent SQLite storage for metadata and inverted properties, coupled with an embedded HNSW vector index built with hnswlib. ChromaDB supports rich metadata filtering queries using operators such as $eq, $ne, $gt, $in, $and, and $or, enabling hybrid filtering where queries are restricted to specific dates, categories, or user tenancy IDs before or during vector ranking.

