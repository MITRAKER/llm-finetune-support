---
title: Customer Support Assistant
emoji: 💬
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 6.32.0
python_version: "3.12"
app_file: app.py
pinned: false
license: apache-2.0
short_description: A small open LLM fine-tuned for customer support
---

# Customer support assistant

A demo of a small open model (Qwen3-4B-Instruct-2507) fine-tuned with QLoRA to
answer customer-support questions for an online store. It runs on the free CPU
hardware of Hugging Face Spaces through llama.cpp, using the 4-bit GGUF export.

The model writes placeholders such as `{{Order Number}}` instead of inventing
store details. The app fills a few of them with obviously fake demo values and
shows the rest in [brackets]. It has no access to real orders or accounts.

Code, training pipeline, evaluation report and lessons:
https://github.com/MITRAKER/llm-finetune-support

## Settings

| Variable | Meaning | Default |
|---|---|---|
| `MODEL_REPO` | Hub repo that holds the GGUF file | `your-hf-username/qwen3-4b-support-GGUF` |
| `MODEL_FILE` | File name or glob pattern inside that repo | `*Q4_K_M.gguf` |
| `N_THREADS` | CPU threads for llama.cpp | `2` |

The deploy workflow in the GitHub repository sets `MODEL_REPO` for you.
