"""Optional LLM-as-judge: a strong model grades each reply like a support lead would.

Automatic metrics (metrics.py) are cheap but shallow: ROUGE rewards copying
the reference wording, and the intent classifier only checks the topic. A
judge model reads the customer's message, the reference answer (or the
expected behaviour for out-of-scope prompts) and the bot's reply, and scores
correctness, helpfulness and tone.

Judges have biases of their own (they can favour longer answers, for example),
so we use a fixed rubric, ask for structured scores, and always report the
judge next to the automatic metrics rather than instead of them.

Needs ANTHROPIC_API_KEY (or ANTHROPIC_AUTH_TOKEN). Without it, evaluate.py skips this step.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from statistics import mean
from typing import Any

from pydantic import BaseModel

SCORE_FIELDS = ("correctness", "helpfulness", "tone")
MAX_WORKERS_LIMIT = 32  # hard upper bound on parallel requests
# USD per million tokens (input, output), used only to print an estimate.
PRICES = {
    "claude-opus-5-5": (4.00, 20.00),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-haiku-5-5": (0.10, 0.50),
}

RUBRIC = """You grade replies written by a customer support chatbot for an online store.

Store convention: when the bot does not know a store-specific detail (order number, phone number,
website, opening hours, amounts), it must write a placeholder in double curly braces, for example
{{Order Number}} or {{Customer Support Phone Number}}. Placeholders are correct behaviour. Writing a
concrete made-up value instead (a real-looking phone number, URL, tracking number or price) is an
invented detail.

Score the reply on a 1-5 scale for each of:
- correctness: does it give the right information or steps for this request? Compare with the
  reference answer when one is given (different wording is fine), or with the expected behaviour.
  1 = wrong or harmful, 3 = partly right or missing key steps, 5 = fully right.
- helpfulness: would the customer know what to do next? 1 = not at all, 5 = completely.
- tone: polite, calm, concise and professional. 1 = rude or rambling, 5 = excellent.
Also decide:
- invented_details: true if the reply states concrete facts that are not in the conversation and
  are not placeholders.
- acceptable: true if a support team lead would send this reply to a customer as-is.
Give a one-sentence reason first."""


class JudgeVerdict(BaseModel):
    """The structured answer the judge must return (enforced by structured outputs)."""

    reason: str
    correctness: int
    helpfulness: int
    tone: int
    invented_details: bool
    acceptable: bool


@dataclass
class JudgeItem:
    """One reply to grade."""

    item_id: str
    user_message: str
    reply: str
    reference: str | None = None  # in-domain test example
    expected: str | None = None  # out-of-scope prompt

    def render(self) -> str:
        parts = [f"Customer message:\n{self.user_message}"]
        if self.reference:
            parts.append(f"Reference answer (one good reply, wording may differ):\n{self.reference}")
        if self.expected:
            parts.append(f"Expected behaviour:\n{self.expected}")
        parts.append(f"Chatbot reply to grade:\n{self.reply}")
        return "\n\n".join(parts)


@dataclass
class JudgeResult:
    item_id: str
    verdict: dict[str, Any] | None
    error: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class JudgeSummary:
    n: int
    n_errors: int
    scores: dict[str, float] = field(default_factory=dict)
    est_cost_usd: float | None = None


def validate_verdict(verdict: JudgeVerdict) -> dict[str, Any]:
    """Clamp scores to 1-5; structured outputs guarantee the types but not the range."""
    data = verdict.model_dump()
    for name in SCORE_FIELDS:
        data[name] = max(1, min(5, int(data[name])))
    return data


def _request_kwargs(model: str, effort: str) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"model": model, "max_tokens": 4000, "output_config": {"effort": effort}}
    # Server-side fallback re-runs a request that a safety classifier declines on another
    # model. Haiku has no server-side fallback, so we only ask for it on the other models.
    if not model.startswith("claude-haiku"):
        kwargs["betas"] = ["server-side-fallback-2026-07-01"]
        kwargs["fallbacks"] = "default"
    return kwargs


def judge_one(client: Any, item: JudgeItem, model: str, effort: str) -> JudgeResult:
    """Grade one reply. Errors are recorded, not raised, so one failure cannot stop the run."""
    import anthropic

    try:
        response = client.beta.messages.parse(
            system=RUBRIC,
            messages=[{"role": "user", "content": item.render()}],
            output_format=JudgeVerdict,
            **_request_kwargs(model, effort),
        )
    except (anthropic.APIStatusError, anthropic.APIConnectionError) as exc:
        return JudgeResult(item.item_id, None, error=f"{type(exc).__name__}: {exc}")
    usage = (response.usage.input_tokens, response.usage.output_tokens)
    if response.stop_reason == "refusal" or response.parsed_output is None:
        return JudgeResult(item.item_id, None, f"no verdict (stop_reason={response.stop_reason})", *usage)
    return JudgeResult(item.item_id, validate_verdict(response.parsed_output), None, *usage)


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float | None:
    if model not in PRICES:
        return None
    price_in, price_out = PRICES[model]
    return round(input_tokens / 1e6 * price_in + output_tokens / 1e6 * price_out, 4)


def summarize(results: Sequence[JudgeResult], model: str) -> JudgeSummary:
    """Average the verdicts into one score per field."""
    ok = [r.verdict for r in results if r.verdict is not None]
    scores: dict[str, float] = {}
    if ok:
        for name in SCORE_FIELDS:
            scores[f"judge_{name}"] = round(mean(v[name] for v in ok), 3)
        scores["judge_invented_detail_rate"] = round(mean(float(v["invented_details"]) for v in ok), 4)
        scores["judge_acceptable_rate"] = round(mean(float(v["acceptable"]) for v in ok), 4)
    cost = estimate_cost(model, sum(r.input_tokens for r in results), sum(r.output_tokens for r in results))
    return JudgeSummary(n=len(results), n_errors=len(results) - len(ok), scores=scores, est_cost_usd=cost)


def judge_all(
    items: Sequence[JudgeItem],
    model: str,
    effort: str = "low",
    max_workers: int = 8,
    client: Any = None,
    judge_fn: Callable[[Any, JudgeItem, str, str], JudgeResult] = judge_one,
) -> tuple[list[JudgeResult], JudgeSummary]:
    """Grade all items in parallel (bounded) and summarize."""
    if client is None:
        import anthropic

        client = anthropic.Anthropic(max_retries=5)  # the SDK retries 429/5xx with backoff
    workers = max(1, min(max_workers, MAX_WORKERS_LIMIT, len(items) or 1))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(lambda it: judge_fn(client, it, model, effort), items))
    return results, summarize(results, model)


def judge_available() -> bool:
    """The judge runs only when Anthropic credentials are configured (the SDK reads either variable)."""
    return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))
