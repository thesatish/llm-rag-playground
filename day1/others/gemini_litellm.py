import os
from dotenv import load_dotenv
from litellm import completion

load_dotenv()
os.environ["GEMINI_API_KEY"] = os.getenv("API_KEY_G")

response = completion(
    model="gemini/gemini-3.8-flash",
    messages=[{"role": "user", "content": "What is the capital of France?"}]
)

print(response.choices[0].message.content)