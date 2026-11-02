import os
import sys
from dotenv import load_dotenv
from openai import OpenAI, OpenAIError

# Load environment variables
load_dotenv()

BASE_URL = os.getenv("GEMINI_BASE_URL")
API_KEY = os.getenv("API_KEY_G")
# MODEL = os.getenv("API_KEY_G")
MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")

if not API_KEY or not MODEL:
    print("Error: Ensure 'API_KEY' and 'MODEL' are set in your .env file.")
    sys.exit(1)

# Initialize client
client = OpenAI(
    base_url=BASE_URL,
    api_key=API_KEY
)

# Maintain conversation context
messages = [
    {
        "role": "system",
        "content": "You are a helpful, concise, and friendly AI assistant."
    }
]

print("=" * 45)
print("              My AI Assistant")
print(" Type 'quit', 'exit', or press Ctrl+C to exit")
print("=" * 45)

while True:
    try:
        user_input = input("\nYou: ").strip()

        if not user_input:
            continue

        if user_input.lower() in {"quit", "exit"}:
            print("\nAI : Goodbye! Have a great day.\n")
            break

        # Append user turn to conversation history
        messages.append({"role": "user", "content": user_input})

        print("\nAI : ", end="", flush=True)

        print(" List before sending to model:", messages)

        # Stream response chunks directly to console
        stream = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            stream=True
        )

        full_response = []
        for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            print(delta, end="", flush=True)
            full_response.append(delta)

        print()  # Add newline after stream completes

        # Store complete assistant reply for subsequent context
        messages.append({"role": "assistant", "content": "".join(full_response)})

    except OpenAIError as err:
        print(f"\n[API Error]: {err}")
        # Remove failed user message so context remains valid
        if messages and messages[-1]["role"] == "user":
            messages.pop()
    except (KeyboardInterrupt, EOFError):
        print("\n\nSession terminated. Goodbye!\n")
        break