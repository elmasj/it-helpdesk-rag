"""
Step 3: Split cleaned Microsoft Learn pages into retrieval-sized chunks.

Why chunk at all? Embedding models and retrieval work best over short,
focused passages -- if we embedded a whole 5,000-word doc as one vector,
a question about one specific setting buried in it would get diluted by
everything else on the page. Splitting into smaller pieces lets the
vector search find the *specific* paragraph that answers a question.

Chunking strategy for Markdown docs (see config.py for the size/overlap
numbers and the reasoning behind them):

  1. Split each page into sections by Markdown heading ("# ", "## ", ...).
     Headings are natural topic boundaries -- a section under
     "## Reset a user's password" is usually self-contained, so this is a
     better first cut than blindly slicing every N characters.
  2. Drop navigation-only sections ("Related articles", "See also", etc.)
     -- they're just link lists that would otherwise show up as
     near-duplicate boilerplate across hundreds of chunks and dilute
     retrieval.
  3. Merge sections that are too short to stand alone (e.g. a heading
     immediately followed by another heading) into the next section.
  4. If a section is still longer than MS_DOCS_CHUNK_SIZE, split it with a
     sliding window that tries to break on a paragraph or sentence
     boundary rather than mid-word, carrying MS_DOCS_CHUNK_OVERLAP
     characters of overlap into the next chunk so we don't lose meaning
     right at a cut point.

Output: data/processed/chunks.jsonl -- one JSON object per line, each a
chunk of text plus the metadata (title, product, source URL, section)
needed to embed it and later cite it back to the user.

Run it with:
    venv\\Scripts\\python.exe scripts\\chunk_documents.py
"""

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
MD_LINK_RE = re.compile(r"\[([^\]]*)\]\([^)]*\)")

# Boilerplate section headings that are just navigation link lists, not
# content -- skipping them avoids near-duplicate "Related articles" noise
# showing up as retrieval results.
BOILERPLATE_HEADINGS = {
    "related articles", "related content", "see also", "next steps",
    "related links", "additional resources",
}

# A section shorter than this (after stripping its own heading line) is
# almost certainly just a heading with no real body -- merge it into
# whatever comes next rather than emit a near-empty chunk.
MIN_SECTION_BODY_CHARS = 40


def split_into_sections(body: str) -> list[dict]:
    """Split a Markdown doc into sections at heading lines.

    Returns a list of {"heading": str | None, "text": str} dicts. Text
    includes the heading line itself, since that's useful context for
    both embedding and for the chunk shown to Claude later.
    """
    lines = body.splitlines()
    sections = []
    current_heading = None
    current_lines: list[str] = []

    for line in lines:
        match = HEADING_RE.match(line)
        if match:
            if current_lines:
                sections.append({"heading": current_heading, "text": "\n".join(current_lines).strip()})
            current_heading = match.group(2).strip()
            current_lines = [line]
        else:
            current_lines.append(line)

    if current_lines:
        sections.append({"heading": current_heading, "text": "\n".join(current_lines).strip()})

    return sections


def is_boilerplate(section: dict) -> bool:
    heading = (section["heading"] or "").strip().lower()
    return heading in BOILERPLATE_HEADINGS


def body_length_excluding_heading(section: dict) -> int:
    text = section["text"]
    if section["heading"]:
        # Drop just the first line (the heading itself) before measuring.
        text = "\n".join(text.splitlines()[1:])
    return len(text.strip())


def merge_short_sections(sections: list[dict]) -> list[dict]:
    """Fold sections with little/no body into the following section."""
    merged: list[dict] = []
    carry = ""

    for section in sections:
        text = carry + ("\n\n" if carry else "") + section["text"]
        if body_length_excluding_heading(section) < MIN_SECTION_BODY_CHARS:
            carry = text
            continue
        merged.append({"heading": section["heading"], "text": text})
        carry = ""

    if carry:
        # Trailing short section with nothing after it -- attach to the
        # last real section rather than drop it.
        if merged:
            merged[-1]["text"] += "\n\n" + carry
        else:
            merged.append({"heading": None, "text": carry})

    return merged


def split_long_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Sliding-window split that prefers breaking on paragraph/sentence boundaries."""
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        if end >= len(text):
            chunks.append(text[start:].strip())
            break

        # Prefer a paragraph break, then a sentence end, before falling
        # back to a hard cut at `end`. Only accept a "natural" break if
        # it's at least half a chunk_size past `start` -- otherwise, on
        # text with no real sentence structure (JSON blobs, config
        # samples), rfind() can lock onto a stray period right near the
        # start of the window and produce a runaway series of tiny
        # chunks instead of properly-sized ones.
        min_break = start + (chunk_size // 2)
        break_point = text.rfind("\n\n", start, end)
        if break_point < min_break:
            candidate = text.rfind(". ", start, end)
            break_point = candidate + 1 if candidate >= min_break else -1
        if break_point < min_break:
            break_point = end

        chunk = text[start:break_point].strip()
        if chunk:
            chunks.append(chunk)

        # Step forward, backing up by `overlap` characters so the next
        # chunk repeats a little context from the end of this one.
        start = max(break_point - overlap, start + 1)

    return [c for c in chunks if c]


def topic_for_doc(doc_meta: dict) -> str | None:
    """Work out which configured source folder a doc came from (e.g. "networking",
    "drivers", "authentication") from its GitHub source URL -- this becomes the
    chunk's "topic", one level more specific than "product". No re-fetch needed:
    everything required is already in config.MS_DOCS_SOURCES and the manifest.
    """
    marker = "/blob/main/"
    if marker not in doc_meta["source_url"]:
        return None
    path_in_repo = doc_meta["source_url"].split(marker, 1)[1]

    for source in config.MS_DOCS_SOURCES:
        if source["product"] != doc_meta["product"]:
            continue
        for folder in source["folders"]:
            if path_in_repo.startswith(folder + "/"):
                return folder.rsplit("/", 1)[-1]
    return None


def chunk_document(doc_meta: dict) -> list[dict]:
    file_path = config.PROJECT_ROOT / doc_meta["local_path"]
    body = file_path.read_text(encoding="utf-8")
    topic = topic_for_doc(doc_meta)

    sections = split_into_sections(body)
    sections = [s for s in sections if not is_boilerplate(s)]
    sections = merge_short_sections(sections)

    chunks = []
    for section_index, section in enumerate(sections):
        pieces = split_long_text(section["text"], config.MS_DOCS_CHUNK_SIZE, config.MS_DOCS_CHUNK_OVERLAP)
        for piece_index, piece in enumerate(pieces):
            chunks.append({
                "chunk_id": f"{doc_meta['id']}__s{section_index}_p{piece_index}",
                "doc_id": doc_meta["id"],
                "source_type": "ms_doc",
                "product": doc_meta["product"],
                "topic": topic,
                "title": doc_meta["title"],
                "section": section["heading"],
                "source_url": doc_meta["source_url"],
                "text": piece,
                "char_count": len(piece),
            })

    return chunks


def main():
    manifest_path = config.RAW_MS_DOCS_DIR / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    all_chunks = []
    for doc_meta in manifest:
        all_chunks.extend(chunk_document(doc_meta))

    config.ensure_dirs()
    out_path = config.PROCESSED_DIR / "chunks.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk) + "\n")

    char_counts = [c["char_count"] for c in all_chunks]
    print(f"{len(manifest)} source documents -> {len(all_chunks)} chunks")
    print(f"chunk length: min={min(char_counts)}, max={max(char_counts)}, "
          f"avg={sum(char_counts)/len(char_counts):.0f} characters")
    print(f"Written to {out_path}")


if __name__ == "__main__":
    main()
