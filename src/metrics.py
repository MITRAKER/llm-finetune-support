"""Automatic metrics for support answers (used by evaluate.py).

Everything here is cheap, deterministic and runs on CPU, so it can be unit
tested. The optional LLM judge (judge.py) adds a slower, more human-like score
on top of these.
"""

from __future__ import annotations

import bisect
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
# A support bot must not make up order numbers, phone numbers, emails, URLs,
# amounts or codes. The dataset teaches it to write placeholders such as
# {{Order Number}} instead. These patterns flag concrete details in a reply;
# details the customer wrote in their own message may be repeated.
DETAIL_PATTERNS: dict[str, re.Pattern[str]] = {
    "email": re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b"),
    # A URL ends before brackets, quotes and Markdown bold, so a link written as
    # [track it](https://shop.example.com/track), **https://...** or “https://...” keeps none of them.
    "url": re.compile(
        r"\b(?:https?://|www\.)[^\s<>()\[\]{}\"'`*“”‘’«»…]+"
        r"|\b[\w-]+\.(?:com|net|org|io|co|shop|store|info|biz|us|uk|ca|au|de|fr|eu|app)\b",
        re.IGNORECASE,
    ),
    # Phone numbers: 1-800-555-0199, +44 20 7946 0958, (555) 123-4567, (555)123-4567 and
    # +1 555 0100. Short numbers such as 555-0199 are found by _phone_numbers() below.
    # Dates are masked first, so 12.05.2024 is not a phone number.
    "phone": re.compile(
        r"(?<![\w])(?:\+?\d{1,3}[\s.-]+)?(?:\(?\d{2,4}\)?[\s.-]+){2,}\d{3,4}(?![\w])"
        r"|(?<![\w])\+\d{1,3}(?:[\s.-]\d{2,4}){2,}(?![\w])"
        r"|\(\d{3}\)\d{3}[\s.-]?\d{4}(?![\w])"
    ),
    # Order / tracking numbers: 5+ digits in a row, optionally after "#".
    "long_number": re.compile(r"(?<![\w])#?\d{5,}(?![\w])"),
    # Codes that mix capital letters and digits, such as the discount code SAVE100 or the
    # tracking number 1Z999AA10123456784. Case-sensitive on purpose: ordinary words never match.
    "code": re.compile(r"\b(?=[A-Z0-9]*\d)(?=[A-Z0-9]*[A-Z])[A-Z0-9]{6,}\b"),
    "money": re.compile(
        r"[$€£]\s?\d[\d,]*(?:\.\d+)?|\b\d[\d,]*(?:\.\d+)?\s?(?:USD|EUR|GBP|dollars|euros)\b", re.IGNORECASE
    ),
}
# A short number such as 555-0199 or 555 0199 is a phone number only when a word such as
# "call", "phone" or "contact" comes before it in the same sentence: on its own it is as often
# a range ("orders of 100-1000"). Even then, a range of round hundreds ("our lines are open
# 900-1700") or a number followed by a unit on the same line ("250-5000 units") is not one.
PHONE_CUE_RE = re.compile(
    r"\b(?:call\w*|phon\w*|tel|telephone|dial\w*|text\w*|sms|fax|hotlines?|helplines?|lines?|numbers?"
    r"|contact\w*|reach\w*|mobile|cell|whatsapp|ring)\b",
    re.IGNORECASE,
)
SHORT_PHONE_RE = re.compile(
    r"(?<![\w.$€£-])(?!\d00[\s.-]\d\d00(?!\d))\d{3}[\s.-]\d{4}(?![\w-]|[ \t.-]\d)"
    r"(?![ \t]*(?:units?|items?|pieces?|pcs|products?|orders?|people|hours?|hrs|minutes?|mins|days?|weeks?)\b)"
)
# A sentence ends at . ! ? followed by a capital letter ("tel. 555 0199" goes on), or at a line
# break, unless the line ends with a colon: a label such as "**Phone:**" goes on to the next line.
SENTENCE_END_RE = re.compile(r"[.!?]\s+(?=[A-Z])|(?<![:*_\s])[*_ \t\r]*\n\s*")
# Dates such as 12.05.2024, 12/05/24 or 2024-05-12 look like digit groups but are not invented details.
DATE_RE = re.compile(r"(?<![\w.-])(?:\d{1,2}[./-]\d{1,2}[./-]\d{2,4}|\d{4}[./-]\d{1,2}[./-]\d{1,2})(?![\w-]|\.\d)")
NUMBER_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")
# A bare number in the customer's message counts as an amount only next to a money word, as in
# "I was charged 50 twice", "my payment of 1,299" or "50 dollars": "order 100 is missing" and
# "invoice 100 is wrong" do not excuse "$100". (A number with two decimals, such as 49.99, is
# taken as an amount wherever the customer wrote it.)
MONEY_CUE_RE = re.compile(
    r"[$€£]|\b(?:charg(?:e[sd]?|ing)|pa(?:id|ys?|ying|yments?)|refund\w*|pric(?:e[sd]?|ing)|cost(?:s|ing)?"
    r"|fees?|totals?|amounts?|spent|bill\w*|bought|usd|eur|gbp|dollars?|euros?|pounds?)\b",
    re.IGNORECASE,
)
MONEY_CUE_GAP = 12  # at most this many characters between the money word and the number
# A number right after one of these words is an id, even next to a money word: in "will I be
# refunded for order 88231?", 88231 is the order, not an amount.
ID_BEFORE_RE = re.compile(
    r"(?:#|\b(?:order|invoice|ticket|tracking|account|reference|ref|item|number|no\.)\s*#?)\s*$", re.IGNORECASE
)


def _without_placeholders(text: str) -> str:
    """Placeholders become a space, so that one cannot join the text around it into an email or a link."""
    return PLACEHOLDER_RE.sub(" ", text)


def _searchable(text: str) -> str:
    """The text the patterns run on: placeholders and dates removed."""
    return DATE_RE.sub(" ", _without_placeholders(text))


def _any_between(positions: list[int], low: int, high: int) -> bool:
    """Whether a sorted list of positions holds one from low to high."""
    i = bisect.bisect_left(positions, low)
    return i < len(positions) and positions[i] <= high


def _phone_numbers(text: str) -> list[str]:
    """The phone numbers in a text, in order: every match of DETAIL_PATTERNS["phone"], plus the
    short numbers that follow a word such as "call" in their sentence. A short number inside a
    longer match, such as 123-4567 in (555) 123-4567, is not counted again."""
    full = list(DETAIL_PATTERNS["phone"].finditer(text))
    full_starts = [m.start() for m in full]
    cue_ends = [m.end() for m in PHONE_CUE_RE.finditer(text)]
    sentence_starts = [0, *(m.end() for m in SENTENCE_END_RE.finditer(text))]
    short = []
    for number in SHORT_PHONE_RE.finditer(text):
        at = number.start()
        sentence_start = sentence_starts[bisect.bisect_right(sentence_starts, at) - 1]
        before = bisect.bisect_right(full_starts, at) - 1  # the last longer match that starts at or before it
        inside_full = before >= 0 and at < full[before].end()
        if _any_between(cue_ends, sentence_start, at) and not inside_full:
            short.append(number)
    return [m.group(0) for m in sorted(full + short, key=lambda m: m.start())]


def _find_details(text: str) -> dict[str, list[str]]:
    """The details in a text, by kind (placeholders and dates already removed)."""
    return {
        kind: _phone_numbers(text) if kind == "phone" else [m.group(0) for m in pattern.finditer(text)]
        for kind, pattern in DETAIL_PATTERNS.items()
    }


def _canon(kind: str, item: str) -> str:
    """A normal form, so that "$1,000" in a reply equals "1000" in the customer's message."""
    if kind in ("phone", "long_number"):
        return re.sub(r"\D", "", item)
    if kind == "money":
        number = NUMBER_RE.search(item)
        return f"{float(number.group(0).replace(',', '')):.2f}" if number else ""
    if kind == "url":
        return re.sub(r"^(?:https?://)?(?:www\.)?", "", item.lower()).rstrip("/.,;:!?_~")
    return item.lower()


def _same_phone(a: str, b: str) -> bool:
    """Two phone numbers (digits only) that differ at most in the country code or the leading 0:
    +1 555 123 4567 is 555 123 4567, and +44 20 7946 0958 is 020 7946 0958. The shorter one must
    be a whole national number (8 digits or more) and the longer one may add at most 3 digits,
    so neither (212) 555-0199 nor 1-800-123-4567 is the customer's 555-0199 or order 1234567.
    (Numbers that follow +1 or +44 have 9 or 10 digits, so a 7-digit number that gains digits
    is an invented prefix, not a country code.) A known limit: from the digits alone, a
    country code cannot be told from any other prefix of up to 3 digits."""
    shorter, longer = sorted((a.lstrip("0"), b.lstrip("0")), key=len)
    return len(shorter) >= 8 and len(longer) - len(shorter) <= 3 and longer.endswith(shorter)


def _given_amounts(text: str) -> set[str]:
    """Money normal forms of the bare numbers the customer wrote next to a money word (ids excepted)."""
    cues = list(MONEY_CUE_RE.finditer(text))
    cue_ends, cue_starts = [c.end() for c in cues], [c.start() for c in cues]
    return {
        _canon("money", number.group(0))
        for number in NUMBER_RE.finditer(text)
        if (
            _any_between(cue_ends, number.start() - MONEY_CUE_GAP, number.start())
            or _any_between(cue_starts, number.end(), number.end() + MONEY_CUE_GAP)
        )
        and not ID_BEFORE_RE.search(text[max(0, number.start() - 20) : number.start()])
    }


def _given_details(user_message: str) -> set[str]:
    """Normal forms of the details the customer wrote. A detail must match a whole one, never a
    part: the customer's order 88231 does not excuse an invented refund of $88. The one exception
    is a phone number's country code (see _same_phone)."""
    text = _searchable(user_message)
    given = {_canon(kind, hit) for kind, hits in _find_details(text).items() for hit in hits}
    # The customer's own short numbers need no word such as "call": "it's 555-0199 now".
    given.update(_canon("phone", m.group(0)) for m in SHORT_PHONE_RE.finditer(text))
    given.update(_canon("long_number", number) for number in NUMBER_RE.findall(text))
    given.update(_given_amounts(text))
    given.update(re.findall(r"[\w@.+-]+", text.lower()))
    return given


def _was_given(kind: str, canon: str, given: set[str]) -> bool:
    """Whether the customer wrote this detail (a phone number may gain or lose its country code)."""
    if canon in given:
        return True
    return kind == "phone" and any(g.isdigit() and _same_phone(canon, g) for g in given)


def invented_details(response: str, user_message: str = "") -> dict[str, list[str]]:
    """Concrete details in the response that the customer did not provide."""
    given = _given_details(user_message)
    found: dict[str, list[str]] = {}
    for kind, hits in _find_details(_searchable(response)).items():
        invented = [h for h in hits if _canon(kind, h) and not _was_given(kind, _canon(kind, h), given)]
        if invented:
            found[kind] = invented
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
