from openai import OpenAI
from similarity import cosine_similarity
from dotenv import load_dotenv
import os
EMODEL = os.getenv("EMODEL", "qwen3-embedding:0.6b")

load_dotenv()

client = OpenAI(
    base_url=os.getenv("BASE_URL"),
    api_key=os.getenv("API_KEY", "ollama")
)

def create_embedding(content):
    embedding1 = client.embeddings.create(
    model=EMODEL,
    input=content
    ).data[0].embedding
    return embedding1





