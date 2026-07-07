"""
Thin wrapper around the local ChromaDB vector store.

ChromaDB stores everything as plain files under config.CHROMA_DB_DIR --
no server to run, no account to create. get_collection() below is used by
both the indexing script (writes chunks in) and the query/CLI code
(reads similar chunks back out), so the connection setup lives in one
place.
"""

from functools import lru_cache

import chromadb

import config


@lru_cache(maxsize=1)
def get_collection():
    """Open (or create) the local persistent Chroma collection.

    Cached so an interactive session (the CLI asking many questions in a
    row) reuses one client/collection instead of reopening the on-disk
    database every time.
    """
    client = chromadb.PersistentClient(path=str(config.CHROMA_DB_DIR))
    return client.get_or_create_collection(
        name=config.CHROMA_COLLECTION_NAME,
        # Our embeddings are normalized (see embeddings.py), so cosine
        # similarity is the correct distance function -- ChromaDB
        # defaults to squared L2, which would still rank results in the
        # same order for normalized vectors, but "cosine" is the explicit,
        # standard choice for text embeddings and makes the distance
        # scores easier to reason about (0 = identical, 2 = opposite).
        metadata={"hnsw:space": "cosine"},
    )
