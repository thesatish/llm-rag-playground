import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client(api_key=os.getenv("API_KEY_G"))

response = client.models.generate_content(
    model="gemini-3.8-flash",
    contents="What is the capital of France?"
)

print(response.text)