import os
from openai import OpenAI
from dotenv import load_dotenv
from tools import read_text_file

# Load configuration from .env
load_dotenv()

# Initialize AI client
client = OpenAI(
    base_url=os.getenv("GEMINI_BASE_URL", os.getenv("BASE_URL")),
    api_key=os.getenv("API_KEY_G", os.getenv("API_KEY"))
)

model = os.getenv("GEMINI_MODEL", os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite"))


def find_and_read_file(filename):
    """
    Find and read file from given path, data/ directory, or script's data directory.
    Returns (content, resolved_path) or (None, None) if not found.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        filename,
        os.path.join("data", filename),
        os.path.join(script_dir, filename),
        os.path.join(script_dir, "data", filename)
    ]

    for path in candidates:
        if os.path.isfile(path):
            content = read_text_file(path)
            if not content.startswith("Error:"):
                return content, path

    return None, None


def main():
    print("x" * 50)
    print("       Document QA Assistant (File Reader)")
    print("x" * 50)
    
    # Show available sample files if data directory exists
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "data")
   
    if os.path.exists(data_dir):
        files = [f for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f))]
        if files:
            print("\nAvailable files in data/: " + " | ".join(files))

    while True:
        file_input = input("\nEnter file name to load (or 'quit' to exit): ").strip()

        if file_input.lower() in ("quit", "exit"):
            print("\nAI : Goodbye! Have a great day.")
            break

        if not file_input:
            continue

        file_content, resolved_path = find_and_read_file(file_input)

        if not file_content:
            print(f"\nError: File '{file_input}' not found. Please try again.")
            continue

        print(f"\n[+] Successfully loaded '{resolved_path}'")
        print("You can now ask questions about this document.")
        print("(Type 'change file' to load a different file, or 'quit' to exit)")

        while True:
            user_question = input("\nAsk question: ").strip()

            if not user_question:
                continue

            if user_question.lower() in ("quit", "exit"):
                print("\nAI : Goodbye! Have a great day.")
                return

            if user_question.lower() in ("change file", "switch file", "change", "back"):
                break

            prompt = f"""You are given a document.

Document:
\"\"\"
{file_content}
\"\"\"

Question:
{user_question}

Instructions:
1. Answer the question using ONLY the information present in the document.
2. If the answer is not found in the document, reply strictly with:
   'Answer could not be found'
3. Do not assume or add outside information.
"""

            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a strict document QA assistant. Answer only using the document provided. "
                            "If the answer is not present in the document, reply strictly with: 'Answer could not be found'."
                        )
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.0
            )

            answer = response.choices[0].message.content.strip()
            print("\nAI :", answer)


if __name__ == "__main__":
    main()
    