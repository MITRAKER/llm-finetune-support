---
base_model: {{base_model}}
datasets:
  - {{dataset}}
language:
  - en
license: apache-2.0
pipeline_tag: text-generation
tags:
  - customer-support
  - qlora
  - lora
  - unsloth
  - trl
  - gguf
---

# Customer support assistant ({{hub_model_id}})

A small open model fine-tuned to answer customer-support questions for an online
store: orders, cancellations, refunds, shipping, payments, invoices and accounts.
It was trained with QLoRA on one free GPU, evaluated against the base model with
automatic metrics and quality gates, and exported to GGUF so it runs on a CPU.

Code, data pipeline and evaluation: https://github.com/MITRAKER/llm-finetune-support

## How it answers: placeholders instead of made-up details

The model does not know your store's phone number, website or order numbers, and
it must not invent them. Where a store-specific detail belongs, it writes a
placeholder in double curly braces, for example:

> I'm sorry about the delay with order {{Order Number}}. You can reach our team at
> {{Customer Support Phone Number}} during {{Customer Support Hours}}.

Your application fills these placeholders from its own systems before showing the
reply. The demo app in the repository shows how (`fill_placeholders` in `app/app.py`).

## Training

| Setting | Value |
|---|---|
| Base model | `{{base_model}}` |
| Dataset | `{{dataset}}` |
| Training examples | {{train_examples}} |
| Method | QLoRA (4-bit NF4 base, LoRA adapters on all attention and MLP projections) |
| LoRA rank / alpha | {{lora_r}} / {{lora_alpha}} |
| Learning rate | {{learning_rate}} |
| Epochs | {{epochs}} |
| Loss | on the assistant's answer only (completion-only loss) |
| Export | GGUF, `{{quantization}}` |

The model was trained with this system prompt. Use exactly the same prompt at
inference time: the model answers best, and writes placeholders most reliably,
with the prompt it was trained with.

```text
{{system_prompt}}
```

## Evaluation

These numbers were measured on the fine-tuned model in its training form (the
4-bit NF4 base model plus the LoRA adapter), on held-out test questions and 24
out-of-scope prompts (off-topic requests, prompt injection, requests for other
customers' data). The GGUF file is the same adapter merged into the 16-bit base
model and quantized to `{{quantization}}`. That conversion usually changes the
quality only slightly, but it was not measured separately.

{{evaluation_report}}

## How to use

With llama.cpp (CPU, from the GGUF repository):

```python
from llama_cpp import Llama

SYSTEM_PROMPT = """{{system_prompt}}"""

llm = Llama.from_pretrained(repo_id="{{hub_model_id}}-GGUF", filename="{{gguf_file}}", n_ctx=2048)
reply = llm.create_chat_completion(
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "I want to cancel my order"},
    ],
    temperature=0.3,
)
print(reply["choices"][0]["message"]["content"])
```

With transformers and peft (GPU, from the adapter repository `{{hub_model_id}}-lora`):
load the base model, then `PeftModel.from_pretrained(base_model, "{{hub_model_id}}-lora")`.

## Intended use and limitations

- Intended for demos, prototypes and research on fine-tuning. It is not a
  production support agent: it has no access to real orders or accounts.
- It learned from synthetic English support conversations, so it is weaker on
  other languages, unusual requests and multi-step problems.
- Like any language model it can be wrong. Keep a human in the loop for refunds,
  payments and anything that affects a customer's money or data.
- It was not trained on off-topic or adversarial requests. The evaluation measures
  how it handles 24 such prompts and checks that it is not worse than the base
  model (the out-of-scope rows above), but nothing guarantees that it declines
  them. Put your own guardrails in front of it.

## License

The fine-tuned weights are released under Apache 2.0, the license of the base
model. Check the dataset's own license before reusing the training data.
