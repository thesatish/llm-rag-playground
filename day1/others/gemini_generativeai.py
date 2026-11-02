import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

genai.configure(api_key=os.getenv("API_KEY_G"))
model = genai.GenerativeModel("gemini-3.8-flash")

response = model.generate_content("What is the capital of France?")
print(response.text)