# HuggingFace-compatible wrapper for AGPT models

Wrap our `.model` checkpoint format in a `PreTrainedModel` subclass so we can use the standard LM evaluation ecosystem (`lm-eval-harness`, `evaluate`, etc.) and get apples-to-apples comparisons with published char-LM numbers (Transformer-XL, GPT-2-byte, etc.).

## Why

Currently to evaluate AGPT we have to either:
- Use our own evaluators (`bin/agpt_sliding_window_perplexity`, `src/tools/agpt_ppl.py`) — fine for inner-loop iteration but invisible to the wider LM-eval ecosystem.
- Hand-port any standard benchmark dataset to our format and write the eval loop ourselves — high friction, easy to introduce subtle protocol mismatches.

With an HF wrapper:
- `lm-eval-harness` runs out of the box on any of its built-in benchmarks (enwik8, text8, wikitext, lambada, etc.)
- HF's `evaluate.load("perplexity")` gives one-call PPL on any HF dataset
- Side-by-side comparison with GPT-2, LLaMA, etc., uses the same eval code as the published numbers
- Anyone else can `pip install`-style try our model

## Cost: ~half-day

We already have ~80% of this. `src/tools/agpt_ppl.py` already loads our `.model` into a plain PyTorch transformer with the right architecture (RoPE, ReLU FFN, etc.). The remaining work is packaging.

## Scope

1. **`AGPTConfig(PretrainedConfig)`** — fields: `vocab_size`, `d_model`, `n_heads`, `n_layers`, `d_ff`, `seq_len` (= our d). Maps to `.model` file's u32 header.
2. **`AGPTModel(PreTrainedModel)`** — `__init__` builds the PyTorch modules. `forward(input_ids, attention_mask=None, ...)` returns `CausalLMOutput(logits=...)`. Refactor `agpt_ppl.py`'s forward into this method.
3. **`from_pretrained(path)`** — reads our `.model` binary, populates the PyTorch state_dict (token embeddings, per-layer Q/K/V/O/LN/FFN matrices, final LN, output projection). The mapping is already in `agpt_ppl.py`'s `load_model()`.
4. **`save_pretrained(path)`** — round-trip to HF format (config.json + safetensors or pytorch_model.bin). Not strictly required for eval; nice for sharing.
5. **Tokenizer wrapper** — a minimal `PreTrainedTokenizer` subclass that does char-level tokenization with our 65-char vocab. `encode(text) → list[int]`, `decode(ids) → str`. Trivial since char = id.
6. **Test:** load a checkpoint, evaluate it with `lm-eval-harness --task wikitext --model_args path=<our_model.pt>`. Should run end-to-end.

Files: probably `src/tools/agpt_hf.py` (config + model + tokenizer in one file is fine, total ~300 lines).

## When

Defer until current position-distribution experiment lands. This is research-velocity-neutral until we need to run a standardized benchmark — then it's the unblock for the entire LM-eval ecosystem. Natural timing: before the "grand finale" run on enwik8/text8.

## Related

- `src/tools/agpt_ppl.py` — has the PyTorch architecture code that becomes `AGPTModel.forward`.
- `notes/agpt/per-node-position-distributions.md` — current research direction; this work is unrelated and can wait.
- `feedback_evaluator_consistency.md` — confirms our existing PyTorch evaluator (`agpt_ppl.py --mode fixed`) gives the canonical PPL number; the HF wrapper inherits that correctness.
