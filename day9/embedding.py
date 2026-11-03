from openai import OpenAI
from dotenv import load_dotenv
import os

# Load configuration from .env
load_dotenv()

# Create AI client
client = OpenAI(
    base_url=os.getenv("BASE_URL"),
    api_key=os.getenv("API_KEY")
)

#Embedding Model
EMODEL = os.getenv("EMODEL", "qwen3-embedding:0.6b")

text1 = "Python is a programming language."

text2 = "Python is used for software development."

embedding1 = client.embeddings.create(
    model=EMODEL,
    input=text1
).data[0].embedding


embedding2 = client.embeddings.create(
    model=EMODEL,
    input=text2
).data[0].embedding

print(embedding1)
print(len(embedding2))
