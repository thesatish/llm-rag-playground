import os
from dotenv import load_dotenv
from openai import OpenAI
from similarity import cosine_similarity

# Load configuration from .env
load_dotenv()

# Create AI client
client = OpenAI(
    base_url=os.getenv("BASE_URL"),
    api_key=os.getenv("API_KEY", "ollama")
)

# Embedding Model
EMODEL = os.getenv("EMODEL", "qwen3-embedding:0.6b")

# 1. High similarity
text1 = "I want to book a hotel in Paris."
text2 = "I am looking for accommodation in Paris."

# 2. Medium similarity
text3 = "I want to visit Paris during my vacation."
text4 = "What are the best tourist attractions in France?"

# 3. Low similarity
text5 = "I want to book a hotel in Paris."
text6 = "How does a computer processor work?"

# 4. Same meaning, different wording
text7 = "What is the best time to visit Japan?"
text8 = "When should I plan my trip to Japan?"

# 5. Same words, different context
text10 = "I booked a flight to London."
text11 = "The flight of birds looks beautiful."

# 6. RAG-style similarity
text12 = "What documents are required to travel internationally?"
text13 = "International travelers generally need a valid passport and, depending on the destination, an appropriate visa."

# Group pairs with their category description
pairs = [
    ("1. High similarity", text1, text2),
    ("2. Medium similarity", text3, text4),
    ("3. Low similarity", text5, text6),
    ("4. Same meaning, different wording", text7, text8),
    ("5. Same words, different context", text10, text11),
    ("6. RAG-style similarity", text12, text13),
]

# Optimize: embed all texts in a single batch request instead of multiple round-trips
all_texts = [text for _, t1, t2 in pairs for text in (t1, t2)]
print("all_texts:::", len(all_texts))

response = client.embeddings.create(
    model=EMODEL,
    input=all_texts
)

embeddings = [item.embedding for item in response.data]

print("=" * 60)
print(f"Embedding Model: {EMODEL}")
print("=" * 60)

# Calculate and display cosine similarity score for all pairs
for i, (category, t1, t2) in enumerate(pairs):
    emb1 = embeddings[2 * i]
    emb2 = embeddings[2 * i + 1]
    score = cosine_similarity(emb1, emb2)

    print(f"\n{category}")
    print(f"  Text 1: \"{t1}\"")
    print(f"  Text 2: \"{t2}\"")
    print(f"  Score : {score:.4f}")


# How the formula works in the loop:
# When looping through for i, ... in enumerate(pairs):

# For Pair 0 (i = 0):
# emb1 = embeddings[2 * 0]     -> embeddings[0]
# emb2 = embeddings[2 * 0 + 1] -> embeddings[1]

# For Pair 1 (i = 1):
# emb1 = embeddings[2 * 1]     -> embeddings[2]
# emb2 = embeddings[2 * 1 + 1] -> embeddings[3]

# For Pair 2 (i = 2):
# emb1 = embeddings[2 * 2]     -> embeddings[4]
# emb2 = embeddings[2 * 2 + 1] -> embeddings[5]