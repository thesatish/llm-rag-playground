from openai import OpenAI
from similarity import cosine_similarity

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)

text1 = "Python is a programming language."

text2 = "Python is used for software development."

embedding1 = client.embeddings.create(
    model="nomic-embed-text",
    input=text1
).data[0].embedding

embedding2 = client.embeddings.create(
    model="nomic-embed-text",
    input=text2
).data[0].embedding



print(len(embedding1))
print(len(embedding2))

score = cosine_similarity(
    embedding1,
    embedding2
)

print(score)
