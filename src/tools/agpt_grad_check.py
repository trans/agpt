#!/usr/bin/env python3
"""Directional-derivative check of the v2 trainer's aggregated gradient.

For a checkpoint theta:
  1. frozen pd=1 dump (AGPT_GRAD_DUMP_DIR) -> event-weighted mean gradient g
     and trie loss f(theta) (event-weighted mean of unit mean losses)
  2. write theta +/- eps * d  (d = g/|g|) as .model files
  3. frozen dumps at theta +/- eps*d -> f(theta +/- eps*d)
  4. compare central difference (f+ - f-)/(2 eps) with g.d = |g|

An exact gradient agrees to O(eps^2). Run with anc_grad on and off (the
forward, and therefore f, does not depend on it; the backward does).

Usage:
  python3 src/tools/agpt_grad_check.py --ckpt CKPT.model --base-config CFG.yml --out DIR [--eps 1e-2 3e-3]
"""
import argparse
import csv
import json
import os
import re
import struct
import subprocess
import sys

import numpy as np


def read_model_flat(path):
    data = open(path, "rb").read()
    off = 4 + 24
    blocks = []  # (header_offset, n_floats)
    while off + 8 <= len(data):
        rows, cols = struct.unpack_from("<2i", data, off)
        n = rows * cols
        if rows <= 0 or cols <= 0 or off + 8 + 4 * n > len(data):
            break
        blocks.append((off, n))
        off += 8 + 4 * n
        if len(blocks) >= 1 + 16 * 64 + 4:  # safety
            break
    return data, blocks


def write_perturbed(src, dst, delta, n_params):
    data, blocks = read_model_flat(src)
    buf = bytearray(data)
    pos = 0
    for off, n in blocks:
        if pos >= n_params:
            break
        vals = np.frombuffer(data, dtype="<f4", count=n, offset=off + 8).astype(np.float64)
        vals = vals + delta[pos:pos + n]
        buf[off + 8:off + 8 + 4 * n] = vals.astype("<f4").tobytes()
        pos += n
    assert pos == n_params, f"walked {pos} floats, expected {n_params}"
    open(dst, "wb").write(bytes(buf))


def run_dump(cfg_text, init, anc, dump_dir, work, trainer="bin/agpt_train_v2"):
    os.makedirs(dump_dir, exist_ok=True)
    y = re.sub(r"init_file: .*", f"init_file: {init}", cfg_text)
    y = re.sub(r"anc_grad: .*", f"anc_grad: {'true' if anc else 'false'}", y)
    y = re.sub(r"value: \d+", "value: 1", y)
    y = re.sub(r"partition_depth: \d+", "partition_depth: 1", y)
    y = re.sub(r"\n  save_file: .*", "", y)
    y = re.sub(r"\n  checkpoint_epochs:(\n *- *\d+)+", "", y)
    y = re.sub(r"name: lbfgs", "name: sgd", y)
    cfgp = os.path.join(work, "cfg.yml")
    open(cfgp, "w").write(y)
    env = dict(os.environ, AGPT_GRAD_DUMP_DIR=dump_dir)
    r = subprocess.run([trainer, "--config", cfgp], env=env, capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(r.stdout[-2000:] + r.stderr[-2000:])
    lay = json.load(open(os.path.join(dump_dir, "layout.json")))
    rows = list(csv.DictReader(open(os.path.join(dump_dir, "units.tsv")), delimiter="\t"))
    ev = np.array([float(r["trained_events"]) for r in rows])
    ml = np.array([float(r["mean_loss"]) for r in rows])
    f = float((ev * ml).sum() / ev.sum())
    X = np.fromfile(os.path.join(dump_dir, "grads.f32"), dtype=np.float32).reshape(len(rows), lay["total_floats"])
    g = (X.astype(np.float64) * ev[:, None]).sum(0) / ev.sum()
    return f, g, lay


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--base-config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--eps", type=float, nargs="+", default=[1e-2, 3e-3])
    ap.add_argument("--trainer", default="bin/agpt_train_v2")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    cfg_text = open(args.base_config).read()

    f0, g_anc, lay = run_dump(cfg_text, args.ckpt, True, os.path.join(args.out, "d0_anc"), args.out, args.trainer)
    _, g_no, _ = run_dump(cfg_text, args.ckpt, False, os.path.join(args.out, "d0_noanc"), args.out, args.trainer)
    P = lay["total_floats"]
    na, nn = np.linalg.norm(g_anc), np.linalg.norm(g_no)
    cos = float(g_anc @ g_no / (na * nn))
    print(f"ckpt {args.ckpt}  trainer {args.trainer}")
    print(f"  f(theta) = {f0:.7f}   |g_anc| = {na:.5f}   |g_noanc| = {nn:.5f}   cos(g_anc, g_noanc) = {cos:.4f}")
    res = {"ckpt": args.ckpt, "f0": f0, "g_anc_norm": na, "g_noanc_norm": nn, "cos_anc_noanc": cos, "checks": []}
    for which, g in (("anc", g_anc), ("noanc", g_no)):
        d = g / np.linalg.norm(g)
        for eps in args.eps:
            fs = {}
            for sign in (+1, -1):
                mp = os.path.join(args.out, f"theta_{which}_{eps:g}_{'p' if sign > 0 else 'm'}.model")
                write_perturbed(args.ckpt, mp, sign * eps * d, P)
                f, _, _ = run_dump(cfg_text, mp, True, os.path.join(args.out, f"d_{which}_{eps:g}_{sign}"), args.out, args.trainer)
                fs[sign] = f
                os.remove(mp)
            fd = (fs[+1] - fs[-1]) / (2 * eps)
            pred_anc = float(g_anc @ d)
            pred_no = float(g_no @ d)
            curv = (fs[+1] + fs[-1] - 2 * f0) / eps ** 2
            print(f"  dir=g_{which:5s} eps={eps:<6g} finite-diff d f/d eps = {fd:+.6f} | g_anc.d = {pred_anc:+.6f} (ratio {fd / pred_anc:.4f}) "
                  f"| g_noanc.d = {pred_no:+.6f} (ratio {fd / pred_no:.4f}) | curvature d2f = {curv:.4f}")
            res["checks"].append({"dir": which, "eps": eps, "fd": fd, "g_anc_dot_d": pred_anc, "g_noanc_dot_d": pred_no,
                                  "f_plus": fs[+1], "f_minus": fs[-1], "curvature": curv})
    json.dump(res, open(os.path.join(args.out, "result.json"), "w"), indent=1)
    # clean the large dumps
    for sub in os.listdir(args.out):
        p = os.path.join(args.out, sub, "grads.f32")
        if os.path.exists(p):
            os.remove(p)


if __name__ == "__main__":
    main()
