"""Gradio demo for the fine-tuned support assistant (step 5).

Runs on a free Hugging Face Space (CPU only). The model is the 4-bit GGUF
export from step 5, served by llama.cpp, which is fast enough on a CPU for a
3-4B model.

The model writes placeholders such as {{Customer Support Phone Number}}
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
from collections.abc import Iterator
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
MAX_HISTORY_MESSAGES = 6  # bounded context: only the last 3 exchanges are sent
CONTEXT_TOKENS = 2048
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


def build_messages(message: str, history: list[dict[str, Any]]) -> list[dict[str, str]]:
    """System prompt + the most recent text turns + the new user message."""
    if not message.strip():
        raise ValueError("empty message")
    turns = [
        {"role": str(m["role"]), "content": str(m["content"])}
        for m in history[-MAX_HISTORY_MESSAGES:]
        if m.get("role") in ("user", "assistant") and isinstance(m.get("content"), str)
    ]
    return [{"role": "system", "content": SYSTEM_PROMPT}, *turns, {"role": "user", "content": message}]


@lru_cache(maxsize=1)
def get_model() -> Any:  # pragma: no cover - downloads the GGUF model
    from llama_cpp import Llama

    return Llama.from_pretrained(
        repo_id=MODEL_REPO,
        filename=MODEL_FILE,
        n_ctx=CONTEXT_TOKENS,
        n_threads=N_THREADS,
        verbose=False,
    )


def respond(message: str, history: list[dict[str, Any]]) -> Iterator[str]:  # pragma: no cover - needs the model
    """Stream the reply token by token, filling placeholders as they complete."""
    stream = get_model().create_chat_completion(
        messages=build_messages(message, history),
        max_tokens=MAX_NEW_TOKENS,
        temperature=0.2,
        stream=True,
    )
    reply = ""
    for chunk in stream:
        reply += chunk["choices"][0]["delta"].get("content") or ""
        yield fill_placeholders(reply)


def build_demo() -> Any:  # pragma: no cover - UI wiring
    import gradio as gr

    return gr.ChatInterface(
        fn=respond,
        title="Customer support assistant (fine-tuned Qwen3-4B)",
        description=(
            "A small open model fine-tuned with QLoRA on the Bitext customer-support dataset. "
            "Details such as phone numbers are placeholders filled by the app, not invented by the model. "
            "Demo only: no real orders or accounts."
        ),
        examples=EXAMPLES,
    )


if __name__ == "__main__":  # pragma: no cover
    build_demo().launch()
