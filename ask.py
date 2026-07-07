"""
Step 6: interactive CLI -- ask the IT Helpdesk Knowledge Assistant a question.

This is the main entry point for actually using the assistant once the
data pipeline (fetch -> chunk -> build_index, all under scripts/) has been
run once. Each question you type:
  1. Is embedded and matched against the local ChromaDB index (free, local)
  2. Gets sent to Claude along with the matched passages as context (costs
     a fraction of a cent -- see the per-query cost printed after each answer)
  3. Comes back as a grounded answer plus a legend of exactly which
     documents it drew from, so you can verify or dig deeper

Run it with:
    venv\\Scripts\\python.exe ask.py
"""

import sys

import anthropic

import config
from cost_logger import total_spend_usd
from generate import ask
from vector_store import get_collection

BANNER = """
IT Helpdesk Knowledge Assistant
================================
Model: {model}
Ask an IT support question, or type 'quit' / 'exit' to stop.
"""


def print_citation_legend(chunks: list[dict]) -> None:
    print("\nRetrieved passages (Claude's answer cites these by number):")
    for i, chunk in enumerate(chunks, start=1):
        section = f" - {chunk['section']}" if chunk["section"] else ""
        print(f"  [{i}] {chunk['title']}{section}")
        print(f"      product: {chunk['product']} | {chunk['source_url']}")


def main():
    collection = get_collection()
    if collection.count() == 0:
        print("The vector store is empty. Run these first:")
        print("  venv\\Scripts\\python.exe scripts\\fetch_ms_docs.py")
        print("  venv\\Scripts\\python.exe scripts\\chunk_documents.py")
        print("  venv\\Scripts\\python.exe scripts\\build_index.py")
        sys.exit(1)

    print(BANNER.format(model=config.CLAUDE_MODEL))

    while True:
        try:
            question = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not question:
            continue
        if question.lower() in {"quit", "exit"}:
            break

        try:
            result = ask(question)
        except anthropic.AuthenticationError:
            print("Claude API rejected the request: invalid or missing API key.")
            print("Set a real key in .env (ANTHROPIC_API_KEY=...) and try again.")
            continue
        except anthropic.APIConnectionError:
            print("Couldn't reach the Claude API -- check your internet connection.")
            continue
        except anthropic.APIStatusError as e:
            print(f"Claude API error ({e.status_code}): {e.message}")
            continue

        print(f"\n{result['answer']}")
        print_citation_legend(result["chunks"])
        usage = result["usage"]
        print(
            f"\n[tokens: {usage['input_tokens']} in / {usage['output_tokens']} out "
            f"| this query: ${result['cost_usd']:.4f} "
            f"| session total so far: ${total_spend_usd():.4f}]"
        )

    print(f"\nTotal spend this session and all prior runs: ${total_spend_usd():.4f}")
    print("Goodbye.")


if __name__ == "__main__":
    main()
