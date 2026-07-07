"""
Step 4: Embed every chunk and store it in the local ChromaDB vector store.

What "embedding" means here: the model turns each chunk of text into a
list of 768 numbers (a vector) that captures its meaning. Chunks about
similar topics end up as vectors that are close together in that
768-dimensional space. To answer a question, we'll embed the question the
same way and ask ChromaDB for the stored chunks whose vectors are closest
to it -- that's the entire "search" step of retrieval-augmented
generation, and it runs locally with no API calls.

This script clears and rebuilds the collection from scratch each time it
runs, so it's safe to re-run after re-chunking or changing the embedding
model.

Run it with:
    venv\\Scripts\\python.exe scripts\\build_index.py
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from embeddings import embed_documents
from vector_store import get_collection

BATCH_SIZE = 256  # how many chunks to embed/insert per batch


def load_chunks() -> list[dict]:
    chunks_path = config.PROCESSED_DIR / "chunks.jsonl"
    with chunks_path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def main():
    chunks = load_chunks()
    print(f"Loaded {len(chunks)} chunks from {config.PROCESSED_DIR / 'chunks.jsonl'}")

    collection = get_collection()
    existing = collection.count()
    if existing:
        print(f"Clearing {existing} existing entries from the collection...")
        collection.delete(ids=collection.get()["ids"])

    t0 = time.time()
    for start in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[start:start + BATCH_SIZE]
        texts = [c["text"] for c in batch]
        embeddings = embed_documents(texts)

        collection.add(
            ids=[c["chunk_id"] for c in batch],
            embeddings=embeddings.tolist(),
            documents=texts,
            metadatas=[
                {
                    "doc_id": c["doc_id"],
                    "title": c["title"],
                    "section": c["section"] or "",
                    "product": c["product"],
                    "topic": c["topic"] or "",
                    "source_url": c["source_url"],
                }
                for c in batch
            ],
        )
        done = min(start + BATCH_SIZE, len(chunks))
        print(f"  indexed {done}/{len(chunks)}")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s. Collection now has {collection.count()} chunks.")
    print(f"Stored at {config.CHROMA_DB_DIR}")


if __name__ == "__main__":
    main()
