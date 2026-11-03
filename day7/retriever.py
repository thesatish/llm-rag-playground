from pathlib import Path

def load_documents():

    documents = {}

    folder = Path("knowledge")

    for file in folder.glob("*.txt"):

        documents[file.name] = file.read_text(
            encoding="utf-8"
        )

    return documents

