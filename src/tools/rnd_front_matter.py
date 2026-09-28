#!/usr/bin/env python3
"""Parse and validate experiment front matter (notes/operations/experiment-front-matter.md).

Usage:
  python3 src/tools/rnd_front_matter.py validate [rnd/<exp> ...]   # default: every rnd/*/README.md
  python3 src/tools/rnd_front_matter.py index [--json]              # one line (or JSON record) per experiment

Library use (site generator):
  from rnd_front_matter import load_all, split_front_matter
"""
import datetime
import glob
import json
import os
import re
import sys

import yaml

KINDS = {"experiment", "diagnostic", "baseline", "design", "infrastructure"}
STATUSES = {"planned", "active", "concluded"}
OUTCOMES = {"positive", "negative", "mixed", "inconclusive", "n/a"}
EVALS = {"canonical", "legacy", "none"}
FAMILIES = {"attention", "recurrent", "count-prior", "hybrid", "n/a"}
REQUIRED = ["title", "kind", "status", "outcome", "question", "answer", "opened", "updated", "code", "eval", "family"]
OPTIONAL = ["headline", "tags", "related", "superseded_by"]

FM_RE = re.compile(r"\A---[ \t]*\n(.*?\n)---[ \t]*\n", re.S)


def split_front_matter(text):
    """Return (front_matter_dict_or_None, body_text)."""
    m = FM_RE.match(text)
    if not m:
        return None, text
    data = yaml.safe_load(m.group(1)) or {}
    return data, text[m.end():]


def load_all(root="rnd"):
    out = {}
    for readme in sorted(glob.glob(os.path.join(root, "*", "README.md"))):
        d = os.path.dirname(readme)
        fm, _ = split_front_matter(open(readme, encoding="utf-8").read())
        out[d] = fm
    return out


def _date_ok(v):
    if isinstance(v, datetime.date):
        return True
    return isinstance(v, str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}", v) is not None


TEXT_EXTS = (".md", ".json", ".txt", ".csv", ".tsv", ".log", ".yml", ".yaml")
MAX_SCAN_BYTES = 5_000_000


def _number_in_sources(d, value):
    """True if the headline value appears in any text file of the experiment directory
    (README, notes, result/eval json, csv/tsv tables, logs)."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return False
    texts = []
    for f in glob.glob(os.path.join(d, "**", "*"), recursive=True):
        if not f.endswith(TEXT_EXTS) or not os.path.isfile(f):
            continue
        try:
            if os.path.getsize(f) > MAX_SCAN_BYTES:
                continue
            texts.append(open(f, encoding="utf-8", errors="replace").read())
        except OSError:
            pass
    blob = "\n".join(texts)
    # accept any rendering that rounds to the stated value at its stated precision
    s = repr(value) if not isinstance(value, str) else value
    decimals = len(s.split(".")[1]) if "." in s else 0
    for tok in re.findall(r"-?\d+\.\d+|-?\d+", blob):
        try:
            if round(float(tok), decimals) == round(v, decimals):
                return True
        except ValueError:
            continue
    return False


def validate_entry(d, fm, check_numbers=True):
    errs, warns = [], []
    if fm is None:
        return ["no front matter"], warns
    for k in REQUIRED:
        if k not in fm:
            errs.append(f"missing {k}")
    for k in fm:
        if k not in REQUIRED and k not in OPTIONAL:
            warns.append(f"unknown field {k}")
    if fm.get("kind") not in KINDS:
        errs.append(f"kind {fm.get('kind')!r} not in {sorted(KINDS)}")
    if fm.get("status") not in STATUSES:
        errs.append(f"status {fm.get('status')!r} not in {sorted(STATUSES)}")
    if fm.get("outcome") not in OUTCOMES:
        errs.append(f"outcome {fm.get('outcome')!r} not in {sorted(OUTCOMES)}")
    if fm.get("eval") not in EVALS:
        errs.append(f"eval {fm.get('eval')!r} not in {sorted(EVALS)}")
    if fm.get("family") not in FAMILIES:
        errs.append(f"family {fm.get('family')!r} not in {sorted(FAMILIES)}")
    for k in ("opened", "updated"):
        if k in fm and fm[k] not in ("", None) and not _date_ok(fm[k]):
            errs.append(f"{k} {fm[k]!r} is not YYYY-MM-DD")
    if fm.get("status") == "concluded" and not (fm.get("answer") or "").strip():
        errs.append("concluded but answer is empty")
    if fm.get("status") in ("planned", "active") and fm.get("outcome") not in ("n/a", "inconclusive", "mixed"):
        warns.append(f"status {fm.get('status')} with outcome {fm.get('outcome')}")
    if fm.get("kind") in ("design", "infrastructure") and fm.get("outcome") not in ("n/a", None):
        warns.append(f"{fm.get('kind')} with outcome {fm.get('outcome')}")
    code = fm.get("code")
    if not (code == "main" or (isinstance(code, dict) and "branch" in code)):
        errs.append(f"code must be 'main' or {{branch, tag}}, got {code!r}")
    hl = fm.get("headline") or []
    if not isinstance(hl, list):
        errs.append("headline must be a list")
        hl = []
    if len(hl) > 3:
        warns.append(f"{len(hl)} headline numbers (max 3)")
    for h in hl:
        if not isinstance(h, dict) or not {"label", "metric", "value"} <= set(h):
            errs.append(f"headline item needs label/metric/value: {h!r}")
            continue
        if check_numbers and not _number_in_sources(d, h["value"]):
            warns.append(f"headline value {h['value']} not found in the directory's text files")
    for k in ("related", "superseded_by"):
        for r in fm.get(k) or []:
            if not os.path.isdir(os.path.join("rnd", str(r))):
                warns.append(f"{k} {r!r} is not an rnd/ directory")
    return errs, warns


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ("validate", "index"):
        print(__doc__)
        sys.exit(2)
    cmd = sys.argv[1]
    args = [a for a in sys.argv[2:] if not a.startswith("--")]
    dirs = [a.rstrip("/") for a in args] or sorted(os.path.dirname(p) for p in glob.glob("rnd/*/README.md"))
    all_dirs = sorted(d.rstrip("/") for d in glob.glob("rnd/*/"))
    if cmd == "validate":
        n_err = 0
        missing = [d for d in all_dirs if not os.path.exists(os.path.join(d, "README.md"))] if not args else []
        for d in missing:
            print(f"ERROR {d}: no README.md")
            n_err += 1
        for d in dirs:
            fm, _ = split_front_matter(open(os.path.join(d, "README.md"), encoding="utf-8").read())
            errs, warns = validate_entry(d, fm)
            for e in errs:
                print(f"ERROR {d}: {e}")
            for w in warns:
                print(f"warn  {d}: {w}")
            n_err += len(errs)
        print(f"{len(dirs)} READMEs checked, {n_err} errors")
        sys.exit(1 if n_err else 0)
    else:
        recs = []
        for d in dirs:
            fm, _ = split_front_matter(open(os.path.join(d, "README.md"), encoding="utf-8").read())
            if fm is None:
                continue
            rec = {"dir": d, **{k: (str(v) if isinstance(v, datetime.date) else v) for k, v in fm.items()}}
            recs.append(rec)
        if "--json" in sys.argv:
            print(json.dumps(recs, indent=1))
        else:
            for r in recs:
                print(f"{r['dir']:40} {r.get('kind',''):14} {r.get('status',''):10} {r.get('outcome',''):12} {r.get('title','')}")


if __name__ == "__main__":
    main()
