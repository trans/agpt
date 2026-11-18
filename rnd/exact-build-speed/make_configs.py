#!/usr/bin/env python3
"""Write the exact-build-speed grid's configs into configs/.

The question (the user, 2026-11-18): how much slower is v2 in full precision
than in its normal build? Two of rnd/one-pass-gap's runs, unchanged except for
their names, under four precision settings, twice each:

  setting    K/V cache  matmuls  trainer (bin/agpt_experiment --trainer)  env
  bf16-tf32  bf16       TF32     v2 (the normal build)                    -
  bf16-fp32  bf16       fp32     v2                                       NVIDIA_TF32_OVERRIDE=0
  fp32-tf32  fp32       TF32     v2-kvfp32                                -
  fp32-fp32  fp32       fp32     v2-kvfp32 (full precision)               NVIDIA_TF32_OVERRIDE=0

  pd1: one-pass-gap configs/agpt-e4-lr0.012.yml (AGPT pd=1, 4 passes, ~27 s)
  pd3: one-pass-gap configs/pd3cs-e1-lr0.0015.yml (pd=3 + crown units,
       shuffled, 1 pass, ~53 s; the 7.81 of rnd/one-pass-gap)

The config cannot name the build or the env var, so the run_slug does and
run.sh maps it: <case>-<setting>-r<repeat>. Both bases use anc_grad_exact
(the exact ancestor gradient) and seed 42.
"""
from pathlib import Path

D = Path(__file__).resolve().parent
SRC = D.parent / "one-pass-gap" / "configs"
BASES = {"pd1": "agpt-e4-lr0.012.yml", "pd3": "pd3cs-e1-lr0.0015.yml"}
SETTINGS = {
    "bf16-tf32": "bf16 K/V cache, TF32 matmuls (the normal build)",
    "bf16-fp32": "bf16 K/V cache, fp32 matmuls (NVIDIA_TF32_OVERRIDE=0)",
    "fp32-tf32": "fp32 K/V cache (agpt_train_v2_kvfp32), TF32 matmuls",
    "fp32-fp32": "fp32 K/V cache, fp32 matmuls (full precision)",
}

out = D / "configs"
out.mkdir(exist_ok=True)
for case, base in BASES.items():
    lines = (SRC / base).read_text().splitlines()
    for r in (1, 2):
        for s, what in SETTINGS.items():
            slug = f"{case}-{s}-r{r}"
            new = []
            for ln in lines:
                if ln.startswith("description:"):
                    old = ln.split(":", 1)[1].strip().strip('"')
                    ln = f'description: "exact-build speed, {what}, repeat {r}: {old}"'
                elif ln.startswith("experiment:"):
                    ln = "experiment: exact-build-speed"
                elif ln.startswith("run_slug:"):
                    ln = f"run_slug: {slug}"
                new.append(ln)
            (out / f"{slug}.yml").write_text("\n".join(new) + "\n")
            print(out / f"{slug}.yml")
