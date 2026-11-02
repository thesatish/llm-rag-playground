import os
import requests
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("API_KEY_G")
url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3.8-flash:generateContent?key={api_key}"

payload = {
    "contents": [{"parts": [{"text": "What is the capital of France?"}]}]
}

response = requests.post(url, json=payload).json()
print(response["candidates"][0]["content"]["parts"][0]["text"])