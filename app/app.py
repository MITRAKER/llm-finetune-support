"""Gradio demo for the fine-tuned support assistant (step 5).

Runs on a free Hugging Face Space (CPU only). The model is the 4-bit GGUF
export from step 5, served by llama.cpp, which is fast enough on a CPU for a
3-4B model.

The model is trained to write placeholders such as {{Customer Support Phone Number}}
instead of inventing details. A real deployment fills them from the store's
own systems; this demo fills a few from STORE_DETAILS below (obviously fake
values) and shows the rest in [brackets].

Environment variables (set them in the Space settings):
    MODEL_REPO   Hub repo with the GGUF file, e.g. "your-name/qwen3-4b-support-GGUF"
    MODEL_FILE   file name or glob pattern inside it (default "*Q4_K_M.gguf")
    N_THREADS    CPU threads for llama.cpp (default 2, the free "CPU basic" Space size)
"""

from __future__ import annotations

import os
import re
from collections.abc import Callable, Iterator
from functools import lru_cache
from typing import Any

# Must stay identical to SYSTEM_PROMPT in src/common.py (tests/test_common.py checks this):
# the model answers best with the exact prompt it was trained with.
SYSTEM_PROMPT = (
    "You are a friendly, concise customer support assistant for an online store. "
    "Help customers with orders, cancellations, refunds, shipping, payments, "
    "invoices and their account. When you need a store-specific detail you do not "
    "know (an order number, phone number, website URL, opening hours and so on), "
    "write a placeholder in double curly braces, such as {{Order Number}} or "
    "{{Customer Support Phone Number}}, instead of making one up."
)
PLACEHOLDER_RE = re.compile(r"\{\{\s*([^{}]+?)\s*\}\}")
STORE_DETAILS = {
    "Customer Support Phone Number": "+1 (555) 010-0199",
    "Customer Support Email": "support@example.com",
    "Website URL": "shop.example.com",
    "Customer Support Hours": "Monday to Friday, 9 AM to 6 PM",
    "Company Name": "Example Store",
}
MODEL_REPO = os.environ.get("MODEL_REPO", "your-hf-username/qwen3-4b-support-GGUF")
MODEL_FILE = os.environ.get("MODEL_FILE", "*Q4_K_M.gguf")
# os.cpu_count() reports the host's cores inside a container, which oversubscribes a 2-vCPU Space.
N_THREADS = int(os.environ.get("N_THREADS", "2"))
MAX_NEW_TOKENS = 400
CONTEXT_TOKENS = 2048
# Bounded context, in two steps. First a cheap cut by characters: at most the last 3
# exchanges and HISTORY_CHAR_BUDGET characters of them. Then fit_to_context() counts real
# tokens with the model's tokenizer and drops the oldest turns until the prompt leaves
# room for a full reply. Characters alone are not enough: English takes about 4
# characters per token, but Chinese or a list of order numbers takes 1 or 2.
MAX_HISTORY_MESSAGES = 6
HISTORY_CHAR_BUDGET = 3000
MAX_MESSAGE_CHARS = 1500
# Tokens the chat template adds around each message (<|im_start|>role ... <|im_end|>)
# and before the reply, counted generously so the estimate never comes out too small.
TEMPLATE_TOKENS_PER_MESSAGE = 8
TEMPLATE_TOKENS_FOR_REPLY = 16
PROMPT_TOKEN_BUDGET = CONTEXT_TOKENS - MAX_NEW_TOKENS - TEMPLATE_TOKENS_FOR_REPLY
TOO_LONG = "Please send a shorter message: this one does not fit in the model's context window."
BRACKETED_RE = re.compile(r"\[([A-Z][^\[\]{}]{0,60})\]")
EXAMPLES = [
    "I want to cancel my order",
    "How do I get a refund for a damaged item?",
    "What's your customer service phone number?",
    "Where is my package? It was supposed to arrive yesterday.",
    "Can you write me a poem about cats?",
]


def fill_placeholders(text: str, details: dict[str, str] = STORE_DETAILS) -> str:
    """Replace known placeholders with store details; show unknown ones as [Name]."""
    return PLACEHOLDER_RE.sub(lambda m: details.get(m.group(1).strip(), f"[{m.group(1).strip()}]"), text)


def restore_placeholders(text: str, details: dict[str, str] = STORE_DETAILS) -> str:
    """Undo fill_placeholders on a reply we showed, so the model sees its own format again."""
    for key, value in details.items():
        text = text.replace(value, "{{" + key + "}}")
    return BRACKETED_RE.sub(lambda m: "{{" + m.group(1) + "}}", text)


def message_text(content: Any) -> str:
    """The text of one chat message.

    Gradio 6 passes earlier messages with content as a list of parts, such as
    [{"type": "text", "text": "..."}]; older versions pass a plain string. Files
    and images have no text and are skipped.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(str(p.get("text", "")) for p in content if isinstance(p, dict) and p.get("type") == "text")
    return ""


def recent_turns(history: list[dict[str, Any]]) -> list[dict[str, str]]:
    """The most recent text turns that fit the history budget, oldest first."""
    turns: list[dict[str, str]] = []
    used = 0
    for m in reversed(history[-MAX_HISTORY_MESSAGES:]):
        text = message_text(m.get("content")).strip()
        if m.get("role") not in ("user", "assistant") or not text:
            continue
        if used + len(text) > HISTORY_CHAR_BUDGET:
            break
        used += len(text)
        role = str(m["role"])
        turns.append({"role": role, "content": restore_placeholders(text) if role == "assistant" else text})
    return turns[::-1]


def build_messages(message: str, history: list[dict[str, Any]]) -> list[dict[str, str]]:
    """System prompt + the most recent text turns + the new user message."""
    if not message.strip():
        raise ValueError("empty message")
    if len(message) > MAX_MESSAGE_CHARS:
        raise ValueError(f"Please keep your message under {MAX_MESSAGE_CHARS:,} characters.")
    return [{"role": "system", "content": SYSTEM_PROMPT}, *recent_turns(history), {"role": "user", "content": message}]


def fit_to_context(
    messages: list[dict[str, str]], count_tokens: Callable[[str], int], budget: int = PROMPT_TOKEN_BUDGET
) -> list[dict[str, str]]:
    """Drop the oldest turns until the prompt fits in budget tokens.

    messages is the output of build_messages(): the system prompt first, the new user
    message last, and the earlier turns in between. count_tokens counts the tokens of one
    text with the model's own tokenizer. Raises ValueError when even the system prompt and
    the new message do not fit.
    """
    system, *turns, latest = messages
    used = sum(count_tokens(m["content"]) + TEMPLATE_TOKENS_PER_MESSAGE for m in (system, latest))
    if used > budget:
        raise ValueError(TOO_LONG)
    kept: list[dict[str, str]] = []
    for turn in reversed(turns):
        used += count_tokens(turn["content"]) + TEMPLATE_TOKENS_PER_MESSAGE
        if used > budget:
            break
        kept.append(turn)
    return [system, *kept[::-1], latest]


@lru_cache(maxsize=1)
def get_model() -> Any:  # pragma: no cover - downloads the GGUF model
    from llama_cpp import Llama

    return Llama.from_pretrained(
        repo_id=MODEL_REPO,
        filename=MODEL_FILE,
        n_ctx=CONTEXT_TOKENS,
        n_threads=N_THREADS,
        n_threads_batch=N_THREADS,  # prompt processing; llama.cpp would otherwise use every host core
        verbose=False,
    )


def respond(message: str, history: list[dict[str, Any]]) -> Iterator[str]:  # pragma: no cover - needs the model
    """Stream the reply token by token, filling placeholders as they complete."""
    try:
        messages = build_messages(message, history)
    except ValueError as exc:
        yield str(exc)
        return
    try:
        model = get_model()
    except (OSError, ValueError, RuntimeError, ImportError) as exc:  # download or load failed: say so, keep the UI up
        yield f"The model could not be loaded ({type(exc).__name__}). Check MODEL_REPO in the Space settings."
        return
    try:
        messages = fit_to_context(messages, lambda text: len(model.tokenize(text.encode("utf-8"), add_bos=False)))
        stream = model.create_chat_completion(
            messages=messages,
            max_tokens=MAX_NEW_TOKENS,
            temperature=0.2,
            stream=True,
        )
        reply = ""
        for chunk in stream:
            reply += chunk["choices"][0]["delta"].get("content") or ""
            yield fill_placeholders(reply)
    except ValueError:  # too long for the context window: llama.cpp raises it while streaming
        yield TOO_LONG


def build_demo() -> Any:  # pragma: no cover - UI wiring
    import gradio as gr

    return gr.ChatInterface(
        fn=respond,
        title="Customer support assistant (fine-tuned Qwen3-4B)",
        description=(
            "A small open model fine-tuned with QLoRA on the Bitext customer-support dataset. "
            "It is trained to write placeholders for store details such as phone numbers, and the app "
            "fills them in. It can still make a detail up, so do not rely on any number it gives. "
            "Demo only: no real orders or accounts."
        ),
        examples=EXAMPLES,
        # Gradio caches examples at startup on Spaces by default: that would download the
        # model and generate every example before the page can load.
        cache_examples=False,
    )


if __name__ == "__main__":  # pragma: no cover
    build_demo().launch()
