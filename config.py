"""
Central configuration for the IT Helpdesk Knowledge Assistant.

Everything you're likely to want to tweak lives here: which Claude
model answers questions, where data/embeddings are stored, and how documents
get split into chunks before indexing. Nothing in this file talks to the
network or costs money by itself -- it's just settings.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load variables from a local .env file (API keys, etc.) into the environment.
# See .env.example for the expected format. This file is gitignored so your
# real key never gets committed.
load_dotenv()

# --- Paths -------------------------------------------------------------
# PROJECT_ROOT is the folder this config.py file lives in, so paths below
# work no matter what directory you run a script from.
PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_MS_DOCS_DIR = DATA_DIR / "raw" / "ms_docs"
RAW_TICKETS_DIR = DATA_DIR / "raw" / "tickets"
PROCESSED_DIR = DATA_DIR / "processed"

CHROMA_DB_DIR = PROJECT_ROOT / "chroma_db"
CHROMA_COLLECTION_NAME = "it_helpdesk_knowledge"

LOG_DIR = PROJECT_ROOT / "logs"
COST_LOG_PATH = LOG_DIR / "cost_log.jsonl"

# --- Embedding model (local, free, CPU-only) ----------------------------
# BAAI/bge-base-en-v1.5 is a strong general-purpose English embedding model
# that runs comfortably on CPU for a few hundred chunks. It downloads once
# (~440MB) from Hugging Face the first time you run it, then is cached
# locally -- no per-query cost.
EMBEDDING_MODEL_NAME = "BAAI/bge-base-en-v1.5"

# bge models are trained to expect a special instruction prefix on the QUERY
# side (not the document side) for retrieval tasks. Leaving this off doesn't
# break anything, but including it measurably improves retrieval quality.
BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

# --- Claude API (generation step only) ----------------------------------
# Swap this to try Sonnet for quality comparisons. Haiku is the default
# because it's ~5x cheaper and plenty capable for grounded Q&A once the
# retrieval step has already found the right context.
#   claude-haiku-4-5  -> $1.00 / $5.00 per 1M input/output tokens
#   claude-sonnet-5   -> $3.00 / $15.00 per 1M input/output tokens
#                        (intro pricing $2.00 / $10.00 through 2026-08-31)
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-haiku-4-5")

# Per-1M-token prices, used by the cost logger to estimate $ spent.
# Keep this in sync with config.CLAUDE_MODEL if you add more models.
MODEL_PRICING_PER_MILLION_TOKENS = {
    "claude-haiku-4-5": {"input": 1.00, "output": 5.00},
    "claude-sonnet-5": {"input": 2.00, "output": 10.00},  # intro pricing
}

CLAUDE_MAX_TOKENS = 1024

# Soft budget ceiling for the project brief's $5-10 target, used only to
# render a spend-vs-budget meter in the web UI -- purely informational,
# nothing stops working if you go over it.
BUDGET_USD_LIMIT = 10.0

# --- Retrieval -----------------------------------------------------------
# How many chunks to retrieve from the vector store per question.
RETRIEVAL_TOP_K = 4

# --- Chunking --------------------------------------------------------
# Chunk size and overlap are measured in characters -- simple and
# predictable, and good enough at this corpus size.
#
# Microsoft Learn docs are long, structured Markdown pages -- we chunk by
# heading section first, then split any section that's still too long.
# 1200 chars (~250-300 tokens) keeps each chunk focused on one sub-topic
# while staying well within the embedding model's context window (512
# tokens) and giving Claude a coherent, self-contained passage to cite.
# 150 chars of overlap avoids losing meaning at a chunk boundary (e.g. a
# sentence that explains a setting right after the heading that names it).
MS_DOCS_CHUNK_SIZE = 1200
MS_DOCS_CHUNK_OVERLAP = 150

# IT tickets are short and self-contained (a subject + description + maybe
# a resolution). We treat each ticket as a single chunk rather than
# splitting it -- splitting would separate the problem from its solution,
# which is exactly the part we want retrievable together. If a ticket is
# unusually long, it still gets capped here to keep embeddings meaningful.
TICKET_MAX_CHUNK_SIZE = 1500

# --- Microsoft Learn doc sources -----------------------------------------
# Public, official Microsoft documentation repos on GitHub. We sparse-clone
# (no full checkout, no blobs until needed) just the folders below instead
# of the whole repo -- each of these repos has thousands of pages; we only
# want a representative, IT-helpdesk-relevant slice.
#
# "product" becomes the sub-folder name under data/raw/ms_docs/ and is
# stored in each chunk's metadata so retrieval results can say which
# product they came from.
MS_DOCS_SOURCES = [
    {
        "repo": "https://github.com/MicrosoftDocs/entra-docs.git",
        "product": "entra-id",  # Azure AD / Microsoft Entra ID
        "folders": [
            "docs/identity/authentication",
            "docs/identity/conditional-access",
            "docs/identity/devices",
            "docs/fundamentals",
        ],
    },
    {
        "repo": "https://github.com/MicrosoftDocs/memdocs.git",
        "product": "intune",
        "folders": [
            "intune/user-help",
            "intune/device-enrollment",
            "intune/device-security",
            "intune/app-management",
            "intune/fundamentals",
        ],
    },
    {
        "repo": "https://github.com/MicrosoftDocs/SupportArticles-docs.git",
        "product": "windows-troubleshooting",
        "folders": [
            "support/windows-client/networking",
            "support/windows-client/setup-upgrade-and-drivers",
            "support/windows-client/performance",
            "support/windows-client/group-policy",
            "support/windows-client/printing",
            "support/windows-client/application-management",
            "support/windows-hardware/drivers",
        ],
    },
]

# Cap per folder so the corpus stays "rich" without turning this into an
# hours-long clone/embed job. With the folders above this yields
# roughly 400-500 source pages -> several hundred chunks after splitting.
MS_DOCS_MAX_FILES_PER_FOLDER = 35

# Sub-paths that exist inside doc folders but aren't content pages
# (images, shared includes, nav breadcrumbs) -- always skipped.
MS_DOCS_SKIP_DIR_NAMES = {"media", "includes", "breadcrumb"}


# --- Small helper ---------------------------------------------------------
def ensure_dirs():
    """Create every directory this project writes to, if missing."""
    for d in (RAW_MS_DOCS_DIR, RAW_TICKETS_DIR, PROCESSED_DIR, CHROMA_DB_DIR, LOG_DIR):
        d.mkdir(parents=True, exist_ok=True)
