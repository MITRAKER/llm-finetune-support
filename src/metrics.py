"""Automatic metrics for support answers (used by evaluate.py).

Everything here is cheap, deterministic and runs on CPU, so it can be unit
tested. The optional LLM judge (judge.py) adds a slower, more human-like score
on top of these.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from statistics import mean
from typing import Any

from common import PLACEHOLDER_RE

# ---------------------------------------------------------------------------
# Output cleanup
# ---------------------------------------------------------------------------
THINK_RE = re.compile(r"<think>.*?</think>\s*", flags=re.DOTALL)


def strip_reasoning(text: str) -> str:
    """Remove <think>...</think> blocks that some reasoning models emit."""
    text = THINK_RE.sub("", text)
    if "<think>" in text:  # unterminated block (generation was cut off)
        text = text.split("<think>")[0]
    return text.strip()


# ---------------------------------------------------------------------------
# Invented details ("hallucinated specifics")
# ---------------------------------------------------------------------------
# A support bot must not make up order numbers, phone numbers, emails, URLs or
# amounts. The dataset teaches it to write placeholders such as {{Order Number}}
# instead. These patterns flag concrete details that did not come from the
# customer's own message.
DETAIL_PATTERNS: dict[str, re.Pattern[str]] = {
    "email": re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b"),
    "url": re.compile(r"\b(?:https?://|www\.)\S+|\b[\w-]+\.(?:com|net|org|io|co|shop|store)\b", re.IGNORECASE),
    # Phone numbers are digit groups split by spaces, dots or dashes:
    # 1-800-555-0199, +44 20 7946 0958, (555) 123-4567.
    "phone": re.compile(r"(?<![\w])(?:\+?\d{1,3}[\s.-]+)?(?:\(?\d{2,4}\)?[\s.-]+){2,}\d{3,4}(?![\w])"),
    # Order / tracking numbers: 5+ digits in a row, optionally after "#".
    "long_number": re.compile(r"(?<![\w])#?\d{5,}(?![\w])"),
    "money": re.compile(
        r"[$€£]\s?\d[\d,]*(?:\.\d+)?|\b\d[\d,]*(?:\.\d+)?\s?(?:USD|EUR|GBP|dollars|euros)\b", re.IGNORECASE
    ),
}


def _without_placeholders(text: str) -> str:
    return PLACEHOLDER_RE.sub(" ", text)


def _canon(item: str) -> str:
    return re.sub(r"[^a-z0-9@.]", "", item.lower())


def invented_details(response: str, user_message: str = "") -> dict[str, list[str]]:
    """Concrete details in the response that the customer did not provide."""
    response_text = _without_placeholders(response)
    given = _canon(user_message)
    found: dict[str, list[str]] = {}
    for kind, pattern in DETAIL_PATTERNS.items():
        hits = [m.group(0).strip() for m in pattern.finditer(response_text)]
        hits = [h for h in hits if _canon(h) and _canon(h) not in given]
        if hits:
            found[kind] = hits
    return found


def uses_placeholder(text: str) -> bool:
    return bool(PLACEHOLDER_RE.search(text))


# ---------------------------------------------------------------------------
# Overlap with the reference answer
# ---------------------------------------------------------------------------
def rouge_l(predictions: Sequence[str], references: Sequence[str]) -> list[float]:
    """ROUGE-L F1: longest-common-subsequence overlap with the reference answer.

    A rough signal only: two good answers can be worded very differently.
    """
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    return [scorer.score(ref, pred)["rougeL"].fmeasure for pred, ref in zip(predictions, references, strict=True)]


# ---------------------------------------------------------------------------
# Intent accuracy: does the answer address what the customer asked?
# ---------------------------------------------------------------------------
class IntentClassifier:
    """TF-IDF + logistic regression trained on reference ANSWERS -> intent.

    We train it on the training split's answers, then ask it which intent a
    generated answer belongs to. If the model answers a refund question with a
    shipping answer, the classifier says "shipping" and that example counts as
    wrong. It never sees the test split during training, so there is no leakage.
    """

    def __init__(self) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline

        self.pipeline = make_pipeline(
            TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
            LogisticRegression(max_iter=2000),
        )

    def fit(self, texts: Sequence[str], intents: Sequence[str]) -> IntentClassifier:
        self.pipeline.fit(list(texts), list(intents))
        return self

    def predict(self, texts: Sequence[str]) -> list[str]:
        return list(self.pipeline.predict(list(texts)))

    def accuracy(self, texts: Sequence[str], intents: Sequence[str]) -> float:
        preds = self.predict(texts)
        return sum(p == t for p, t in zip(preds, intents, strict=True)) / max(len(intents), 1)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
@dataclass
class Summary:
    n: int
    intent_accuracy: float
    rouge_l: float
    invented_detail_rate: float
    placeholder_rate: float
    avg_words: float

    def as_dict(self) -> dict[str, float]:
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in self.__dict__.items()}


def summarize(
    predictions: Sequence[str],
    references: Sequence[str],
    user_messages: Sequence[str],
    true_intents: Sequence[str],
    classifier: IntentClassifier,
) -> tuple[Summary, list[dict]]:
    """Score every prediction and return the summary plus per-example rows."""
    rouge = rouge_l(predictions, references)
    pred_intents = classifier.predict(predictions)
    rows: list[dict[str, Any]] = []
    for i, pred in enumerate(predictions):
        details = invented_details(pred, user_messages[i])
        rows.append(
            {
                "rouge_l": round(rouge[i], 4),
                "predicted_intent": pred_intents[i],
                "intent_correct": pred_intents[i] == true_intents[i],
                "invented_details": details,
                "uses_placeholder": uses_placeholder(pred),
                "words": len(pred.split()),
            }
        )
    summary = Summary(
        n=len(predictions),
        intent_accuracy=_avg(r["intent_correct"] for r in rows),
        rouge_l=_avg(r["rouge_l"] for r in rows),
        invented_detail_rate=_avg(bool(r["invented_details"]) for r in rows),
        placeholder_rate=_avg(r["uses_placeholder"] for r in rows),
        avg_words=_avg(r["words"] for r in rows),
    )
    return summary, rows


def _avg(values: Iterable[float]) -> float:
    values = list(values)
    return float(mean(values)) if values else 0.0
