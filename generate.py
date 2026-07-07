"""
Step 5: generate a grounded answer using retrieved chunks + Claude.

This is the only step in the whole pipeline that costs money or calls a
network API (other than the one-time docs/model downloads). Everything
before this -- chunking, embedding, retrieval -- runs locally for free.

The flow for one question:
  1. Embed the question and retrieve the top-K most similar chunks from
     ChromaDB (see vector_store.py / embeddings.py). This is the
     "Retrieval" half of Retrieval-Augmented Generation.
  2. Build a prompt that gives Claude those chunks as numbered context and
     instructs it to answer using ONLY that context, then name which
     numbered sources it actually relied on.
  3. Call the Claude API (model is configurable in config.py -- Haiku by
     default) and log the real cost of the call using the token counts
     the API returns with every response.
"""

import anthropic

import config
from cost_logger import log_call
from embeddings import embed_query
from vector_store import get_collection

SYSTEM_PROMPT = """You are an IT helpdesk knowledge assistant. You answer IT support questions \
using ONLY the numbered context passages provided below -- they come from official Microsoft \
documentation and troubleshooting articles.

Rules:
- Base your answer strictly on the provided passages. Do not use outside knowledge, and do not \
guess at menu names, settings, or steps that aren't in the passages.
- Never reference, invent, or suggest a URL, KB article number, or documentation link that isn't \
explicitly present in the passages below. If further reading would help but isn't in the \
passages, say that plainly instead of naming or linking a specific page.
- If the passages don't contain enough information to answer, say so plainly instead of guessing.
- Write a clear, direct answer a tier-1 IT support agent could act on immediately.
- End your response with a line starting "Sources:" listing the passage numbers you actually \
relied on, e.g. "Sources: [1], [3]". Only list passages you used.
"""


def retrieve(question: str, top_k: int | None = None) -> list[dict]:
    """Embed the question and fetch the most similar chunks from the vector store."""
    top_k = top_k or config.RETRIEVAL_TOP_K
    collection = get_collection()
    query_embedding = embed_query(question)
    results = collection.query(query_embeddings=[query_embedding.tolist()], n_results=top_k)

    chunks = []
    for doc, meta, distance in zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0]
    ):
        chunks.append({
            "text": doc,
            "title": meta["title"],
            "section": meta["section"],
            "product": meta["product"],
            "topic": meta.get("topic", ""),
            "source_url": meta["source_url"],
            "distance": distance,
        })
    return chunks


def build_context_block(chunks: list[dict]) -> str:
    """Number each chunk so Claude (and our citation legend) can refer to it as [1], [2], ..."""
    parts = []
    for i, chunk in enumerate(chunks, start=1):
        header = f"[{i}] {chunk['title']}"
        if chunk["section"]:
            header += f" - {chunk['section']}"
        parts.append(f"{header}\n{chunk['text']}")
    return "\n\n---\n\n".join(parts)


def ask(question: str, top_k: int | None = None) -> dict:
    """Run the full retrieve -> generate flow for one question.

    Returns a dict with the answer text, the retrieved chunks (so the
    caller can print a citation legend), and real token usage / cost.
    """
    chunks = retrieve(question, top_k)
    context_block = build_context_block(chunks)

    client = anthropic.Anthropic()
    response = client.messages.create(
        model=config.CLAUDE_MODEL,
        max_tokens=config.CLAUDE_MAX_TOKENS,
        system=SYSTEM_PROMPT,
        messages=[{
            "role": "user",
            "content": f"Context passages:\n\n{context_block}\n\nQuestion: {question}",
        }],
    )

    answer_text = next((block.text for block in response.content if block.type == "text"), "")

    log_entry = log_call(
        model=config.CLAUDE_MODEL,
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
        question=question,
    )

    return {
        "answer": answer_text,
        "chunks": chunks,
        "usage": {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        },
        "cost_usd": log_entry["cost_usd"],
    }
