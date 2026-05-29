#!/usr/bin/env python3
"""Migrate legacy AGPT YAML configs to the new schema (docs/yaml-schema.md).

For each legacy .yml passed in, writes a sibling .new-schema.yml that
mirrors the legacy intent under the new field shape. Flags anything that
needs manual review (extra_args, train_frac with non-canonical carve,
growth_frontiers, etc).

Usage:
    tools/migrate_config.py <legacy.yml> [<legacy.yml> ...]
    tools/migrate_config.py --in-place <legacy.yml> ...   # overwrites the .yml

Canonical Shakespeare tail-95/5 carved files live at:
    data/.splits/2b7ded401e96b610/{train,heldout}_corpus.txt

so legacy configs with `corpus.path: data/input.txt` and
`corpus.train_frac: 0.95` map directly onto that cache.
"""

import sys
from pathlib import Path
from collections import OrderedDict

import yaml

CANONICAL_TAIL_95_5_SOURCE = "data/input.txt"
CANONICAL_TAIL_95_5_CACHE  = "data/.splits/2b7ded401e96b610"


def _od(*pairs):
    """OrderedDict from positional (key, value) pairs."""
    return OrderedDict(pairs)


def migrate(old: dict, src_name: str) -> tuple[OrderedDict, str | None, list[str]]:
    warnings: list[str] = []
    new: OrderedDict = OrderedDict()

    # ---- top-level identity ----
    meta = old.get("meta", {})
    desc = meta.get("description", "")
    if "hypothesis_ref" in meta:
        desc = (desc + " ").strip() + f"(ref: {meta['hypothesis_ref']})"
    new["description"] = desc
    new["experiment"]  = meta.get("experiment", "")
    if "run_slug" in meta:
        new["run_slug"] = meta["run_slug"]

    # ---- corpus ----
    old_corpus = old.get("corpus", {})
    new_corpus: OrderedDict = OrderedDict()
    train_frac = old_corpus.get("train_frac", 1.0)
    corpus_path = old_corpus.get("path", "")
    vocab_source = old_corpus.get("vocab_source")

    if train_frac < 1.0:
        if train_frac == 0.95 and corpus_path == CANONICAL_TAIL_95_5_SOURCE:
            new_corpus["path"]    = f"{CANONICAL_TAIL_95_5_CACHE}/train_corpus.txt"
            new_corpus["heldout"] = f"{CANONICAL_TAIL_95_5_CACHE}/heldout_corpus.txt"
            if vocab_source:
                new_corpus["vocab_source"] = vocab_source
            new_corpus["carve"] = _od(
                ("source", CANONICAL_TAIL_95_5_SOURCE),
                ("mode",   "tail"),
                ("ratio",  0.05),
            )
        else:
            # Non-canonical: emit a carve block; user must pre-run agpt_carve
            new_corpus["path"]    = corpus_path
            if vocab_source:
                new_corpus["vocab_source"] = vocab_source
            new_corpus["carve"] = _od(
                ("source", corpus_path),
                ("mode",   "tail"),
                ("ratio",  round(1.0 - train_frac, 6)),
            )
            warnings.append(
                f"non-canonical carve (path={corpus_path!r}, train_frac={train_frac}); "
                f"pre-run bin/agpt_carve to populate corpus.path before orchestrating"
            )
    else:
        new_corpus["path"] = corpus_path
        if vocab_source:
            new_corpus["vocab_source"] = vocab_source
        if old.get("eval", {}).get("split") == "tail-heldout":
            warnings.append(
                "train_frac=1.0 but eval.split=tail-heldout in legacy — no heldout was "
                "ever produced; new orchestrator will error. Add a carve block or set eval differently."
            )
    new["corpus"] = new_corpus

    # ---- model ----
    old_model = old.get("model", {})
    new_model: OrderedDict = OrderedDict()
    old_train = old.get("train", {})

    if "init_from" in old_model:
        new_model["init_file"] = old_model["init_from"]
    if "init_seed" in old_model:
        new_model["init_seed"] = old_model["init_seed"]

    # Microgpt configs put architecture fields under train; lift them.
    # Omit when init_file is present (header carries the shape; redundant fields
    # would just need to match it).
    init_file_present = "init_file" in new_model
    for f in ("d_model", "n_layers", "n_heads", "d_ff", "head_dim"):
        if f in old_train and not init_file_present:
            new_model[f] = old_train[f]

    if new_model:
        new["model"] = new_model

    # ---- train ----
    new_train: OrderedDict = OrderedDict()

    # Budget
    if "epochs" in old_train:
        new_train["budget"] = _od(("unit", "epochs"), ("value", int(old_train["epochs"])))
    elif "steps" in old_train:
        new_train["budget"] = _od(("unit", "steps"), ("value", int(old_train["steps"])))
    else:
        warnings.append("no train.epochs or train.steps in legacy; budget left unset")

    if "seed" in old_train:
        new_train["seed"] = int(old_train["seed"])
    if old_train.get("quiet"):
        new_train["quiet"] = True

    # Optimizer
    opt: OrderedDict = OrderedDict()
    opt["name"] = old_train.get("optimizer", "adam")
    for src, dst in [
        ("lr",             "lr"),
        ("rmsprop_beta",   "beta"),
        ("momentum_beta",  "momentum_beta"),
        ("weight_decay",   "weight_decay"),
        ("grad_clip_norm", "grad_clip_norm"),
    ]:
        if src in old_train:
            opt[dst] = old_train[src]
    new_train["optimizer"] = opt

    # LR schedule
    if "lr_schedule" in old_train or "warmup_epochs" in old_train:
        sched: OrderedDict = OrderedDict()
        sched["name"] = old_train.get("lr_schedule", "constant")
        sched["warmup_epochs"] = int(old_train.get("warmup_epochs", 0))
        new_train["lr_schedule"] = sched
    if "warmup_steps" in old_train:
        warnings.append(
            f"warmup_steps={old_train['warmup_steps']} dropped (schema only has lr_schedule.warmup_epochs)"
        )

    # Context window
    if "growth_max_depth" in old_train:
        new_train["max_depth"] = int(old_train["growth_max_depth"])
    if "seq_len" in old_train:
        new_train["seq_len"] = int(old_train["seq_len"])

    # AGPT knobs (same names)
    for f in ("partition_depth", "chunk_queries", "mass_weight",
              "fire_norm", "entropy_lambda", "ce_only"):
        if f in old_train:
            new_train[f] = old_train[f]

    # anc_grad / ablate_anc_grad (the legacy ablate flag flips anc_grad to false)
    if old_train.get("ablate_anc_grad"):
        new_train["anc_grad"] = False
    elif "anc_grad" in old_train:
        new_train["anc_grad"] = bool(old_train["anc_grad"])

    # Growth — emit a `train.growth` block whenever the legacy config has any
    # of the three real growth knobs (divisions / min_epochs / epoch_ramp).
    # Legacy divisions=1 selects train-growth + incremental-radix materializer,
    # which is a distinct code path from train-epoch + prebuilt static trie,
    # so we preserve it. `growth_max_depth` alone is NOT enough — that field
    # is the context window and migrates to train.max_depth above; without
    # the real growth knobs the legacy config was static-only.
    growth_keys = ("growth_divisions", "growth_min_epochs", "growth_epoch_ramp")
    if any(k in old_train for k in growth_keys):
        growth: OrderedDict = OrderedDict()
        if "growth_divisions" in old_train:
            growth["divisions"] = int(old_train["growth_divisions"])
        if "growth_min_epochs" in old_train:
            growth["min_epochs"] = int(old_train["growth_min_epochs"])
        if "growth_epoch_ramp" in old_train:
            growth["epoch_ramp"] = old_train["growth_epoch_ramp"]
        new_train["growth"] = growth
    if "growth_frontiers" in old_train:
        warnings.append(
            f"growth_frontiers={old_train['growth_frontiers']!r} not in new schema; dropped"
        )

    # Microgpt-only knobs
    for f in ("backend", "heads", "lookahead"):
        if f in old_train:
            new_train[f] = old_train[f]

    # Hard-rejects / advisory
    if "extra_args" in old_train:
        warnings.append(
            f"extra_args={old_train['extra_args']!r} dropped — schema has no escape hatch. "
            f"Audit each flag and either add it as a real field or remove."
        )
    if "accumulate" in old_train:
        warnings.append(
            f"accumulate={old_train['accumulate']} dropped — express as partition_depth=0 if you want one fire over the whole trie"
        )

    new["train"] = new_train

    # ---- eval ----
    old_eval = old.get("eval", {})
    new_eval: OrderedDict = OrderedDict()
    split = old_eval.get("split", "")
    if split == "train":
        new_eval["train_sanity"] = True
    elif split == "external-heldout":
        if "external_file" in old_eval:
            new_eval["external_file"] = old_eval["external_file"]
    elif split == "benchmark":
        if "benchmark" in old_eval:
            new_eval["benchmark"] = old_eval["benchmark"]
    # tail-heldout / unset → schema default
    for f in ("batch_size", "device", "limit"):
        if f in old_eval:
            new_eval[f] = old_eval[f]
    if new_eval:
        new["eval"] = new_eval

    # ---- trainer hint (comment only; --trainer is a CLI flag, not a YAML field) ----
    trainer = None
    kind = old_train.get("kind")
    tool = old_train.get("tool", "") or ""
    if kind == "microgpt-sgd" or "microgpt" in tool:
        trainer = "microgpt"
    elif kind == "cudax" or "agpt_train_v2" in tool:
        trainer = "v2"
    elif tool.endswith("/agpt_train") or tool == "bin/agpt_train":
        trainer = "v1"

    return new, trainer, warnings


def write_yaml(out_path: Path, new: OrderedDict, trainer: str | None, src: Path) -> None:
    # PyYAML emits OrderedDict with !!omap by default; register a representer so
    # we get a plain mapping in insertion order.
    yaml.add_representer(
        OrderedDict,
        lambda dumper, data: dumper.represent_mapping("tag:yaml.org,2002:map", data.items()),
        Dumper=yaml.SafeDumper,
    )

    with open(out_path, "w") as f:
        f.write(f"# Migrated from {src.name} via tools/migrate_config.py.\n")
        if trainer:
            f.write(f"# Run with: bin/agpt_experiment --config <this> --trainer {trainer}\n")
        f.write("\n")
        yaml.safe_dump(new, f, sort_keys=False, default_flow_style=False)


def main(argv: list[str]) -> int:
    in_place = False
    if argv and argv[0] == "--in-place":
        in_place = True
        argv = argv[1:]
    if not argv:
        print(__doc__, file=sys.stderr)
        return 2

    failed = 0
    for path_str in argv:
        src = Path(path_str)
        if not src.exists():
            print(f"error: not found: {src}", file=sys.stderr)
            failed += 1
            continue
        with open(src) as f:
            old = yaml.safe_load(f)
        if not isinstance(old, dict):
            print(f"error: {src} is not a YAML mapping", file=sys.stderr)
            failed += 1
            continue

        new, trainer, warnings = migrate(old, src.name)
        dst = src if in_place else src.with_name(src.stem + ".new-schema.yml")
        write_yaml(dst, new, trainer, src)
        print(f"{src} → {dst}  (trainer hint: {trainer or '?'})")
        for w in warnings:
            print(f"  WARN: {w}")

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
