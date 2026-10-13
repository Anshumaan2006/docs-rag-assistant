import sys
import chromadb
from sentence_transformers import SentenceTransformer
from ingest import load_pdfs, chunk_pages

model = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_or_create_collection("sklearn_docs")


def build_index():
    """Turn every chunk into an embedding and store it in the database."""
    pages = load_pdfs()
    chunks = chunk_pages(pages)
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=True).tolist()
    ids = [f"chunk-{i}" for i in range(len(chunks))]
    metadatas = [{"source": c["source"], "page": c["page"]} for c in chunks]
    collection.upsert(ids=ids, documents=texts, embeddings=embeddings, metadatas=metadatas)
    print(f"Stored {len(chunks)} chunks in the database")


def search(question, k=5):
    """Find the k chunks most similar in meaning to the question."""
    query_embedding = model.encode([question]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=k)
    return list(zip(results["documents"][0], results["metadatas"][0]))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "build":
        build_index()
    elif len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
        for text, meta in search(question):
            print("-" * 60)
            print(f"{meta['source']} (page {meta['page']})")
            print(text)
    else:
        print('Usage: python src\\retrieve.py build   OR   python src\\retrieve.py "your question"')