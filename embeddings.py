"""
Thin wrapper around the local embedding model.

Both the indexing script (embeds document chunks) and the query/CLI code
(embeds the user's question) need the *same* model, loaded the same way --
so that logic lives here once instead of being copy-pasted.

Important detail about BAAI/bge-base-en-v1.5 specifically: it was trained
so that at query time you prepend an instruction ("Represent this sentence
for searching relevant passages: ") to the search query, but NOT to the
documents you're searching over. Getting this asymmetric on purpose is
part of how the model was trained -- it measurably improves retrieval
quality. See config.BGE_QUERY_INSTRUCTION.
"""

from functools import lru_cache

from sentence_transformers import SentenceTransformer

import config


@lru_cache(maxsize=1)
def get_model() -> SentenceTransformer:
    """Load the embedding model once and reuse it (loading takes a few seconds)."""
    return SentenceTransformer(config.EMBEDDING_MODEL_NAME)


def embed_documents(texts: list[str]):
    """Embed document/chunk text for storage in the vector database."""
    model = get_model()
    return model.encode(texts, show_progress_bar=True, normalize_embeddings=True)


def embed_query(text: str):
    """Embed a user's question for similarity search against stored chunks."""
    model = get_model()
    prefixed = config.BGE_QUERY_INSTRUCTION + text
    return model.encode([prefixed], normalize_embeddings=True)[0]
