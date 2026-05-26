#!/usr/bin/env python3
"""HuggingFace-compatible wrapper for AGPT models.

Wraps the existing PyTorch implementation in `agpt_ppl.py` as
`AGPTForCausalLM(PreTrainedModel)` + `AGPTTokenizer(PreTrainedTokenizer)`
so the model is loadable by `lm-evaluation-harness`, `evaluate`, and the
broader HF ecosystem.

This is the canonical AGPT evaluator once verified. `agpt_ppl.py` and
`bin/agpt_sliding_window_perplexity` remain for legacy comparison only.

Usage:
    # Round-trip: load .model, save as HF format
    python3 src/tools/agpt_hf.py convert \\
        --model PATH/in.model --vocab-file PATH/vocab.txt \\
        --out PATH/hf_model_dir

    # Forward parity check vs agpt_ppl.py
    python3 src/tools/agpt_hf.py parity-check \\
        --model PATH/in.model --vocab-file PATH/vocab.txt

    # Standalone PPL (calls lm-evaluation-harness)
    python3 src/tools/agpt_hf.py ppl \\
        --hf-dir PATH/hf_model_dir --file PATH/corpus.txt \\
        --eval-tail-frac 0.05 --max-positions 10000
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    PretrainedConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from transformers.modeling_outputs import CausalLMOutput

# Reuse the proven PyTorch implementation. agpt_ppl.py lives next to us.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
from agpt_ppl import (  # noqa: E402
    AGPTModel as _AGPTBackbone,
    build_rope_cache,
    build_vocab,
    load_model,
)


class AGPTConfig(PretrainedConfig):
    """Config matching the .model file header.

    Fields mirror struct in src/cuda/agpt_train.cu's save_model_weights.
    """

    model_type = "agpt"

    def __init__(
        self,
        vocab_size: int = 65,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 8,
        d_ff: int = 512,
        seq_len: int = 16,
        rope_base: float = 10000.0,
        layer_norm_eps: float = 1e-5,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_ff = d_ff
        self.seq_len = seq_len
        self.rope_base = rope_base
        self.layer_norm_eps = layer_norm_eps
        # HF convention: these aliases let downstream tools that expect
        # GPT-style names still introspect this config.
        kwargs.setdefault("hidden_size", d_model)
        kwargs.setdefault("num_attention_heads", n_heads)
        kwargs.setdefault("num_hidden_layers", n_layers)
        kwargs.setdefault("max_position_embeddings", seq_len)
        super().__init__(**kwargs)

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads


class AGPTForCausalLM(PreTrainedModel):
    """HF-compatible causal LM head over the AGPT backbone."""

    config_class = AGPTConfig
    base_model_prefix = "agpt"
    supports_gradient_checkpointing = False

    def __init__(self, config: AGPTConfig):
        super().__init__(config)
        # Build empty state_dict structure first; weights filled by
        # from_pretrained_agpt() or HF's standard load mechanisms.
        cfg = dict(
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_layers=config.n_layers,
            d_ff=config.d_ff,
            vocab_size=config.vocab_size,
            seq_len=config.seq_len,
            head_dim=config.head_dim,
        )
        sd = self._zero_state_dict(cfg)
        self.backbone = _AGPTBackbone(cfg, sd, device="cpu")
        # Post-init for HF hooks; weights remain zero until loaded.
        self.post_init()

    @staticmethod
    def _zero_state_dict(cfg):
        """Build a zero-initialized state_dict in the shape AGPTBackbone expects."""
        D, F_dim, V, L = (
            cfg["d_model"],
            cfg["d_ff"],
            cfg["vocab_size"],
            cfg["n_layers"],
        )
        sd = {"token_emb": torch.zeros(V, D)}
        for l in range(L):
            sd[f"l{l}.wq_w"] = torch.zeros(D, D)
            sd[f"l{l}.wq_b"] = torch.zeros(1, D)
            sd[f"l{l}.wk_w"] = torch.zeros(D, D)
            sd[f"l{l}.wk_b"] = torch.zeros(1, D)
            sd[f"l{l}.wv_w"] = torch.zeros(D, D)
            sd[f"l{l}.wv_b"] = torch.zeros(1, D)
            sd[f"l{l}.wo_w"] = torch.zeros(D, D)
            sd[f"l{l}.wo_b"] = torch.zeros(1, D)
            sd[f"l{l}.ln1_g"] = torch.ones(1, D)
            sd[f"l{l}.ln1_b"] = torch.zeros(1, D)
            sd[f"l{l}.l1_w"] = torch.zeros(D, F_dim)
            sd[f"l{l}.l1_b"] = torch.zeros(1, F_dim)
            sd[f"l{l}.l2_w"] = torch.zeros(F_dim, D)
            sd[f"l{l}.l2_b"] = torch.zeros(1, D)
            sd[f"l{l}.ln2_g"] = torch.ones(1, D)
            sd[f"l{l}.ln2_b"] = torch.zeros(1, D)
        sd["final_g"] = torch.ones(1, D)
        sd["final_b"] = torch.zeros(1, D)
        sd["out_w"] = torch.zeros(D, V)
        sd["out_b"] = torch.zeros(1, V)
        return sd

    @classmethod
    def from_agpt_checkpoint(cls, model_path: str | os.PathLike) -> "AGPTForCausalLM":
        """Build an HF model directly from a native .model file."""
        cfg_dict, sd = load_model(str(model_path))
        config = AGPTConfig(
            vocab_size=cfg_dict["vocab_size"],
            d_model=cfg_dict["d_model"],
            n_heads=cfg_dict["n_heads"],
            n_layers=cfg_dict["n_layers"],
            d_ff=cfg_dict["d_ff"],
            seq_len=cfg_dict["seq_len"],
        )
        # Construct with zeros, then overwrite with loaded weights via the
        # backbone's existing init path. Reuse _AGPTBackbone so we don't
        # duplicate the row/col transpose logic.
        model = cls(config)
        model.backbone = _AGPTBackbone(cfg_dict, sd, device="cpu")
        return model

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **_unused,
    ) -> CausalLMOutput:
        # AGPT is causal-only; attention_mask is ignored (the backbone applies
        # its own causal mask).
        logits = self.backbone(input_ids)
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
        return CausalLMOutput(loss=loss, logits=logits)

    # HF wants explicit input-embedding accessors for generation.
    def get_input_embeddings(self) -> nn.Module:
        return self.backbone.tok_emb

    def set_input_embeddings(self, value: nn.Module) -> None:
        self.backbone.tok_emb = value


class AGPTTokenizer(PreTrainedTokenizer):
    """Char-level tokenizer. char_id is the index in the sorted-unique-chars
    vocab built from the training corpus.

    Compatible with `from_pretrained` if a `vocab.json` is present
    (list[str] of chars in vocab-id order).
    """

    model_input_names = ["input_ids", "attention_mask"]
    vocab_files_names = {"vocab_file": "vocab.json"}

    def __init__(
        self,
        vocab_file: Optional[str] = None,
        chars: Optional[list[str]] = None,
        unk_token: Optional[str] = None,
        **kwargs,
    ):
        if chars is not None:
            self._chars = list(chars)
        elif vocab_file is not None:
            with open(vocab_file, "r", encoding="utf-8") as f:
                payload = json.load(f)
            self._chars = list(payload)
        else:
            raise ValueError("AGPTTokenizer needs either vocab_file or chars=")
        self._char_to_id = {c: i for i, c in enumerate(self._chars)}
        self._id_to_char = {i: c for i, c in enumerate(self._chars)}
        # unk_token must be one of the vocab chars (or we add it).
        if unk_token not in self._char_to_id:
            unk_token = self._chars[0]
        # BOS/EOS default to newline (a natural char-level document
        # separator); else first vocab entry. lm-evaluation-harness needs
        # a non-None prefix_token_id for loglikelihood_rolling.
        nl = chr(0x0A)
        bos = nl if nl in self._char_to_id else self._chars[0]
        kwargs.setdefault("bos_token", bos)
        kwargs.setdefault("eos_token", bos)
        super().__init__(unk_token=unk_token, **kwargs)

    @classmethod
    def from_corpus(cls, corpus_path: str | os.PathLike) -> "AGPTTokenizer":
        char_to_id, _ = build_vocab(str(corpus_path))
        chars = sorted(char_to_id.keys(), key=lambda c: char_to_id[c])
        return cls(chars=chars)

    @property
    def vocab_size(self) -> int:
        return len(self._chars)

    def get_vocab(self) -> dict:
        return dict(self._char_to_id)

    def _tokenize(self, text: str, **_kwargs) -> list[str]:
        return list(text)

    def _convert_token_to_id(self, token: str) -> int:
        return self._char_to_id.get(token, self._char_to_id[self.unk_token])

    def _convert_id_to_token(self, index: int) -> str:
        return self._id_to_char.get(index, self.unk_token)

    def convert_tokens_to_string(self, tokens: list[str]) -> str:
        return "".join(tokens)

    def save_vocabulary(
        self, save_directory: str, filename_prefix: Optional[str] = None
    ) -> tuple[str]:
        path = Path(save_directory)
        path.mkdir(parents=True, exist_ok=True)
        prefix = f"{filename_prefix}-" if filename_prefix else ""
        out = path / f"{prefix}vocab.json"
        with out.open("w", encoding="utf-8") as f:
            json.dump(self._chars, f, ensure_ascii=False)
        return (str(out),)


def _convert_command(args: argparse.Namespace) -> None:
    """Convert a native .model + vocab into an HF directory."""
    model = AGPTForCausalLM.from_agpt_checkpoint(args.model)
    tokenizer = AGPTTokenizer.from_corpus(args.vocab_file)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"wrote HF model + tokenizer to {out_dir}")


def _parity_command(args: argparse.Namespace) -> None:
    """Compare HF wrapper forward to agpt_ppl's backbone on identical input."""
    cfg_dict, sd = load_model(args.model)
    char_to_id, _ = build_vocab(args.vocab_file)

    # Reference (agpt_ppl path): construct backbone directly.
    ref = _AGPTBackbone(cfg_dict, sd, device="cpu").eval()
    # HF wrapper:
    wrapped = AGPTForCausalLM.from_agpt_checkpoint(args.model).eval()

    torch.manual_seed(0)
    T = min(cfg_dict["seq_len"], 16)
    V = cfg_dict["vocab_size"]
    ids = torch.randint(0, V, (2, T))

    with torch.no_grad():
        ref_logits = ref(ids)
        out = wrapped(ids)
        hf_logits = out.logits

    max_abs = (ref_logits - hf_logits).abs().max().item()
    print(
        f"parity: max abs diff = {max_abs:.3e} "
        f"(shape {tuple(hf_logits.shape)}, vocab {V})"
    )
    if max_abs > 1e-5:
        print("FAIL: parity exceeds 1e-5 threshold", file=sys.stderr)
        sys.exit(2)
    print("PASS")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    conv = sub.add_parser(
        "convert", help="convert .model + vocab → HF directory"
    )
    conv.add_argument("--model", required=True)
    conv.add_argument("--vocab-file", required=True)
    conv.add_argument("--out", required=True)
    conv.set_defaults(func=_convert_command)

    par = sub.add_parser(
        "parity-check", help="verify HF wrapper forward matches agpt_ppl"
    )
    par.add_argument("--model", required=True)
    par.add_argument("--vocab-file", required=True)
    par.set_defaults(func=_parity_command)

    return p


def main() -> None:
    args = _build_parser().parse_args()
    args.func(args)


def _register_with_hf():
    """Register the AGPT model/tokenizer with HF's auto classes.

    Called at import time so any tool that does `import agpt_hf` (or runs
    a script in this directory) gets the registration. Idempotent.
    """
    try:
        AutoConfig.register("agpt", AGPTConfig)
        AutoModelForCausalLM.register(AGPTConfig, AGPTForCausalLM)
        AutoTokenizer.register(AGPTConfig, AGPTTokenizer)
    except ValueError:
        # Already registered (e.g. re-import in same process).
        pass


_register_with_hf()


if __name__ == "__main__":
    main()
