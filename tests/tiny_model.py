"""Build a tiny random Qwen3-style model and ChatML tokenizer on disk, offline.

The smoke test uses it to run the real training and evaluation code on a CPU
in seconds. The model is random, so its answers are gibberish: the test checks
that the pipeline runs and produces the right files, not answer quality.
"""

from __future__ import annotations

from pathlib import Path

# The same shape as Qwen's ChatML template: <|im_start|>role\ncontent<|im_end|>\n
CHATML_TEMPLATE = (
    "{%- for message in messages %}"
    "{{ '<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n' }}"
    "{%- endfor %}"
    "{%- if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{%- endif %}"
)
SPECIAL_TOKENS = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]


def build_tiny_model(out_dir: str | Path, corpus: list[str], vocab_size: int = 400) -> Path:
    """Train a byte-level BPE tokenizer on `corpus` and save a 2-layer random model next to it."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast, Qwen3Config

    out_dir = Path(out_dir)
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tok.train_from_iterator(corpus, trainer)

    hf_tok = PreTrainedTokenizerFast(
        tokenizer_object=tok,
        eos_token="<|im_end|>",
        pad_token="<|endoftext|>",
        additional_special_tokens=["<|im_start|>", "<|im_end|>"],
    )
    hf_tok.chat_template = CHATML_TEMPLATE
    hf_tok.save_pretrained(out_dir)

    config = Qwen3Config(
        vocab_size=len(hf_tok),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=1024,
        tie_word_embeddings=True,
        pad_token_id=hf_tok.pad_token_id,
        eos_token_id=hf_tok.eos_token_id,
    )
    model = AutoModelForCausalLM.from_config(config)
    model.generation_config.eos_token_id = hf_tok.eos_token_id
    model.generation_config.pad_token_id = hf_tok.pad_token_id
    model.save_pretrained(out_dir)
    return out_dir
