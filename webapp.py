"""
Step 7 (GUI): a small Flask web app for the IT Helpdesk Knowledge Assistant.

This is a second front-end for the exact same pipeline ask.py (the CLI)
uses -- retrieve -> generate, all in generate.py. Nothing about the RAG
logic lives in this file; it's purely: serve a page, accept a question
over JSON, run the existing pipeline, return the result as JSON.

Run it with:
    venv\\Scripts\\python.exe webapp.py
Then open http://127.0.0.1:5000 in a browser.

This is a local, single-user dev tool (see README "Out of scope") -- it
runs Flask's built-in server and isn't hardened for public/multi-user
deployment.
"""

from functools import lru_cache

import anthropic
import bleach
import markdown as md
from flask import Flask, jsonify, render_template, request

import config
from cost_logger import total_spend_usd
from generate import ask
from vector_store import get_collection

app = Flask(__name__)

# Friendly display names for the three products in the corpus. Topics
# (the finer-grained folders under each product -- "networking",
# "drivers", "authentication", ...) don't need a lookup table: their
# slugs are already readable words, so we just title-case them.
PRODUCT_LABELS = {
    "entra-id": "Microsoft Entra ID",
    "intune": "Microsoft Intune",
    "windows-troubleshooting": "Windows Troubleshooting",
}

# Words that shouldn't be capitalized when we title-case a topic slug
# (e.g. "setup-upgrade-and-drivers" -> "Setup, Upgrade and Drivers").
_LOWERCASE_WORDS = {"and", "or", "of", "the", "in", "for"}


def humanize_slug(slug: str) -> str:
    words = slug.replace("-", " ").split()
    return " ".join(
        w if w in _LOWERCASE_WORDS else w.capitalize()
        for w in words
    )


@lru_cache(maxsize=1)
def get_topic_index() -> list[dict]:
    """Build the sidebar's product -> topic -> chunk-count breakdown.

    Reads every chunk's metadata once from ChromaDB and aggregates counts
    in Python. Cached because the collection doesn't change while the app
    is running -- rebuilding the index (scripts/build_index.py) requires
    restarting the app anyway, so a fresh process picks up fresh counts.
    """
    metadatas = get_collection().get(include=["metadatas"])["metadatas"]

    counts: dict[str, dict[str, int]] = {}
    for meta in metadatas:
        product = meta.get("product") or "other"
        topic = meta.get("topic") or "general"
        counts.setdefault(product, {})
        counts[product][topic] = counts[product].get(topic, 0) + 1

    # Fixed product order (matches the order they were added to the
    # corpus) rather than alphabetical, so it reads as a deliberate
    # narrative rather than a shuffled list.
    product_order = ["entra-id", "intune", "windows-troubleshooting"]
    ordered_products = [p for p in product_order if p in counts]
    ordered_products += [p for p in counts if p not in product_order]

    index = []
    for product in ordered_products:
        topics = counts[product]
        topic_list = [
            {"slug": topic, "label": humanize_slug(topic), "count": count}
            for topic, count in sorted(topics.items(), key=lambda kv: -kv[1])
        ]
        index.append({
            "slug": product,
            "label": PRODUCT_LABELS.get(product, humanize_slug(product)),
            "total": sum(topics.values()),
            "topics": topic_list,
        })
    return index

# Claude's answer is Markdown, which we render to HTML server-side so the
# browser doesn't need any external JS library (keeps the whole app
# self-contained). We still run the result through bleach afterwards,
# allowing only a small set of formatting tags -- defense in depth in case
# a retrieved passage or answer ever contained something like a raw
# <script> tag, so the browser never executes anything Claude wrote.
ALLOWED_TAGS = [
    "p", "br", "strong", "em", "ul", "ol", "li", "code", "pre",
    "h1", "h2", "h3", "h4", "blockquote", "a", "hr",
]
ALLOWED_ATTRS = {"a": ["href", "title"]}


def render_answer_html(answer_text: str) -> str:
    raw_html = md.markdown(answer_text, extensions=["fenced_code"])
    return bleach.clean(raw_html, tags=ALLOWED_TAGS, attributes=ALLOWED_ATTRS)


@app.route("/")
def index():
    return render_template(
        "index.html",
        model=config.CLAUDE_MODEL,
        budget_limit=config.BUDGET_USD_LIMIT,
        total_spend=total_spend_usd(),
        topic_index=get_topic_index(),
    )


@app.route("/api/ask", methods=["POST"])
def api_ask():
    data = request.get_json(force=True, silent=True) or {}
    question = data.get("question", "").strip()
    if not question:
        return jsonify({"error": "Please enter a question."}), 400

    if get_collection().count() == 0:
        return jsonify({
            "error": "The vector store is empty. Run scripts/fetch_ms_docs.py, "
                     "scripts/chunk_documents.py, then scripts/build_index.py first.",
        }), 503

    try:
        result = ask(question)
    except anthropic.AuthenticationError:
        return jsonify({
            "error": "Claude API rejected the request: invalid or missing API key. "
                     "Set a real key in .env (ANTHROPIC_API_KEY=...) and restart the app.",
        }), 401
    except anthropic.APIConnectionError:
        return jsonify({"error": "Couldn't reach the Claude API -- check your internet connection."}), 502
    except anthropic.APIStatusError as e:
        return jsonify({"error": f"Claude API error ({e.status_code}): {e.message}"}), 502

    citations = [
        {
            "index": i,
            "title": c["title"],
            "section": c["section"],
            "product": c["product"],
            "source_url": c["source_url"],
        }
        for i, c in enumerate(result["chunks"], start=1)
    ]

    spend = total_spend_usd()
    return jsonify({
        "answer_html": render_answer_html(result["answer"]),
        "citations": citations,
        "usage": result["usage"],
        "cost_usd": result["cost_usd"],
        "total_spend_usd": spend,
        "budget_limit_usd": config.BUDGET_USD_LIMIT,
        "budget_pct_used": min(100, round(100 * spend / config.BUDGET_USD_LIMIT, 1)),
    })


if __name__ == "__main__":
    app.run(debug=True)
