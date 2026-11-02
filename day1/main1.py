import os
from dotenv import load_dotenv
from openai import OpenAI

# Load configuration from the .env file
load_dotenv()

# Create a client that communicates with Ollama
client = OpenAI(
    base_url=os.getenv("BASE_URL"),
    api_key=os.getenv("API_KEY")
)

# Send a question to the AI model with streaming enabled
stream = client.chat.completions.create(
    model=os.getenv("MODEL"),
    messages=[
        {
            "role": "user",
            "content": "Do you know who is satish?"
        }
    ],
    stream=True  # Enables token-by-token response
)

# Display tokens as they arrive
for chunk in stream:
    token = chunk.choices[0].delta.content or ""
    print(token, end="", flush=True)

print()