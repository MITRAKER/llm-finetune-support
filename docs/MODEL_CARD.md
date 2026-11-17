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

The system prompt used in training is in `src/common.py`. Use the same prompt at
inference time: the model answers best with the exact prompt it was trained with.

## Evaluation

{{evaluation_report}}

## How to use

With llama.cpp (CPU, from the GGUF repository):

```python
from llama_cpp import Llama

llm = Llama.from_pretrained(repo_id="{{hub_model_id}}-GGUF", filename="{{gguf_file}}", n_ctx=2048)
reply = llm.create_chat_completion(
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},  # copy it from src/common.py
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
- It is trained to decline off-topic requests and prompt-injection attempts, but
  that behaviour is not guaranteed. The repository's evaluation includes a set of
  out-of-scope prompts so you can measure it.

## License

The fine-tuned weights are released under Apache 2.0, the license of the base
model. Check the dataset's own license before reusing the training data.
