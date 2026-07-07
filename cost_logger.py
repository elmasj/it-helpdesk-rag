"""
Track $ spent on Claude API calls against your budget.

Every response from the Messages API includes a `usage` object with the
real input/output token counts for that exact call -- there's no need to
separately estimate or pre-count tokens for cost tracking; we just price
out the numbers the API already gives us for free.

Each call appends one line to logs/cost_log.jsonl (so history survives
across runs) and prints a running total for the session.
"""

import json
import time

import config


def _price_for(model: str) -> dict:
    try:
        return config.MODEL_PRICING_PER_MILLION_TOKENS[model]
    except KeyError:
        raise ValueError(
            f"No pricing entry for model '{model}' in "
            f"config.MODEL_PRICING_PER_MILLION_TOKENS -- add one before using it."
        )


def compute_cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    price = _price_for(model)
    return (input_tokens * price["input"] + output_tokens * price["output"]) / 1_000_000


def log_call(model: str, input_tokens: int, output_tokens: int, question: str) -> dict:
    """Price out and append one API call to the cost log. Returns the log entry."""
    cost_usd = compute_cost_usd(model, input_tokens, output_tokens)
    entry = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model": model,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_usd": round(cost_usd, 6),
        "question": question,
    }

    config.ensure_dirs()
    with config.COST_LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")

    return entry


def total_spend_usd() -> float:
    """Sum every logged call so far (across all past runs, not just this session)."""
    if not config.COST_LOG_PATH.exists():
        return 0.0
    total = 0.0
    with config.COST_LOG_PATH.open(encoding="utf-8") as f:
        for line in f:
            total += json.loads(line)["cost_usd"]
    return total
