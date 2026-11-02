from openai import OpenAI
from dotenv import load_dotenv
import os

# Load configuration from the .env file
load_dotenv()

# Create a client that communicates with Gemini (OpenAI compatibility layer)
client = OpenAI(
    base_url=os.getenv("GEMINI_BASE_URL"),
    api_key=os.getenv("API_KEY_G")
)

# Send a question to the AI model
response = client.chat.completions.create(
    model=os.getenv("GEMINI_MODEL"),
    messages=[
        {
            "role": "user",
            "content": "What are the things I should be aware of while learning RAG?"
        }
    ]
)

# print("response::::::::", response)

# Display the response
print(response.choices[0].message.content)