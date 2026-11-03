from retriever import load_documents

documents = load_documents()

for name, text in documents.items():

    print(name)

    print(text)

    print("-" * 40)
