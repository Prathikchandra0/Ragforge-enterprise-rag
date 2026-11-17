from pathlib import Path
import chromadb

from ingest import load_and_chunk

DB_DIR = Path(__file__).resolve().parent.parent / "chroma_db"
COLLECTION_NAME ="company_docs"


def get_client():
    """connect to a chromadb database stored on a disk"""

    return chromadb.PersistentClient(path=str(DB_DIR))

def build_index():
    """Load the chunks and (re)build the collection from scratch."""
    client = get_client()

    # Delete the old collection so we never keep outdated chunks
    if COLLECTION_NAME in [c.name for c in client.list_collections()]:
        client.delete_collection(COLLECTION_NAME)
    collection = client.create_collection(COLLECTION_NAME)

    chunks = load_and_chunk()
    collection.add(
        ids=[f"chunk_{i}" for i in range(len(chunks))],
        documents=[chunk["text"] for chunk in chunks],
        metadatas=[{"source": chunk["source"]} for chunk in chunks],
    )
    return collection.count()


def retrieve(question, n_results=5):
    """Return the n most relevant chunks for a question."""
    collection = get_client().get_collection(COLLECTION_NAME)
    results = collection.query(query_texts=[question], n_results=n_results)

    return [
        {"text": text, "source": meta["source"]}
        for text, meta in zip(results["documents"][0], results["metadatas"][0])
    ]


if __name__ == "__main__":
    count = build_index()
    print(f"Indexed {count} chunks")

    question = "What is the work from home policy?"
    print(f"\nQuestion: {question}")
    for i, chunk in enumerate(retrieve(question), start=1):
        print("-" * 40)
        print(f"#{i}  Source: {chunk['source']}")
        print(chunk["text"][:200])