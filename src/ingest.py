from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent/"data"

MIN_CHUNK_LENGTH = 50 

def load_documents(data_dir=DATA_DIR):
    """Read every .txt file in the data folder """
    documents=[]
    for path in sorted(data_dir.glob("*.txt")):
        text = path.read_text(encoding ="utf-8")
        documents.append({"text":text,"source":path.stem})
    return documents

def chunk_documents(text,source):
    """Split the document into paragraph chunks."""
    chunks =[]
    for para in text.strip().split("\n\n"):
        para = para.strip()
        if len(para) < MIN_CHUNK_LENGTH or para.startswith("====="):
            continue
        chunks.append({"text":para,"source":source})
    return chunks

def load_and_chunk(data_dir=DATA_DIR):
    """Load all documents and return one list of chunks."""
    all_chunks = []
    for doc in load_documents(data_dir):
        all_chunks.extend(chunk_documents(doc["text"], doc["source"]))
    return all_chunks


if __name__ == "__main__":
    chunks = load_and_chunk()
    print(f"Total chunks: {len(chunks)}")
    for chunk in chunks[:3]:
        print("-" * 40)
        print("Source:", chunk["source"])
        print(chunk["text"][:200])
