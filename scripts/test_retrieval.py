"""
Step 4b (checkpoint): manually sanity-check retrieval quality.

This does NOT call Claude -- it's pure similarity search against the
ChromaDB collection we just built. The point is to confirm the embedding
+ chunking pipeline actually surfaces the right passages *before* we spend
any API budget on generation. If retrieval is bad, no amount of clever
prompting in the generation step will fix it.

Run it with:
    venv\\Scripts\\python.exe scripts\\test_retrieval.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from embeddings import embed_query
from vector_store import get_collection

# A handful of realistic IT-helpdesk-style questions spanning all three
# products in the corpus, so we can eyeball whether retrieval is on-topic
# for each.
TEST_QUERIES = [
    "How do I reset a user's MFA method in Entra ID?",
    "A user's device won't enroll in Intune, what do I check?",
    "How do I edit the Windows registry from a script?",
    "Windows laptop won't wake up from sleep or hibernation",
    "How do I block a specific app on managed Android devices?",
    "Printer troubleshooting steps on Windows",
]


def run_query(collection, question: str, top_k: int):
    query_embedding = embed_query(question)
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=top_k,
    )

    print(f"\nQ: {question}")
    for rank, (doc, meta, distance) in enumerate(
        zip(results["documents"][0], results["metadatas"][0], results["distances"][0]), start=1
    ):
        preview = doc.replace("\n", " ")[:120]
        print(f"  [{rank}] (dist={distance:.3f}, product={meta['product']}) {meta['title']}")
        print(f"       section: {meta['section'] or '(top of doc)'}")
        print(f"       {preview}...")


def main():
    collection = get_collection()
    count = collection.count()
    if count == 0:
        print("Collection is empty -- run scripts/build_index.py first.")
        return
    print(f"Querying collection with {count} chunks (top_k={config.RETRIEVAL_TOP_K})")

    for question in TEST_QUERIES:
        run_query(collection, question, config.RETRIEVAL_TOP_K)


if __name__ == "__main__":
    main()
