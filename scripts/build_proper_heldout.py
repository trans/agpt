#!/usr/bin/env python3
"""Build a proper disjoint held-out from war_peace.txt.

data/gutenberg_5m.txt is the first 5M chars of preprocessed combined_raw.txt
(per data/gutenberg/README.md). Concatenation order:
pride → moby → tale2cities → great_expectations → war_peace

Lengths (raw chars): 748k + 1.24M + 777k + 1.01M + 3.23M = 7.0M raw.
Preprocessing (Unicode→ASCII, vocab filter) trims to 7.27M (matches README)
and the 5M truncation falls inside war_peace, leaving ~2M chars of
preprocessed war_peace OUTSIDE training.

This script preprocesses war_peace.txt using the same pipeline and slices
the LAST 200K chars — definitely disjoint from training.
"""

import sys
from pathlib import Path


SHAKE_VOCAB = set(open('data/input.txt').read())


def preprocess(text):
    """Match data/gutenberg/README.md's described pipeline."""
    # Unicode → ASCII
    text = text.replace('“', '"').replace('”', '"')
    text = text.replace('‘', "'").replace('’', "'")
    text = text.replace('—', '-').replace('–', '-')
    text = text.replace('…', '...')
    text = text.replace(' ', ' ')
    # Drop chars not in Shakespeare vocab
    return ''.join(c for c in text if c in SHAKE_VOCAB)


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else '/tmp/gut_holdout_proper.txt'
    cap = int(sys.argv[2]) if len(sys.argv) > 2 else 200000

    raw = Path('data/gutenberg/war_peace.txt').read_text(encoding='utf-8', errors='replace')
    processed = preprocess(raw)
    print(f"war_peace raw: {len(raw)} chars, preprocessed: {len(processed)} chars", file=sys.stderr)

    # Take last `cap` chars — definitely past the 5M training cut point.
    heldout = processed[-cap:]
    print(f"Held-out: last {cap} chars of preprocessed war_peace", file=sys.stderr)
    print(f"  starts at war_peace position {len(processed) - cap} of {len(processed)} preprocessed", file=sys.stderr)

    # Sanity check: confirm not in the 5M training corpus
    train = Path('data/gutenberg_5m.txt').read_text(encoding='utf-8', errors='replace')
    sample = heldout[:1000]
    if sample in train:
        print(f"ERROR: held-out first 1000 chars FOUND in training corpus — overlap detected!", file=sys.stderr)
        sys.exit(1)
    sample_end = heldout[-1000:]
    if sample_end in train:
        print(f"ERROR: held-out last 1000 chars FOUND in training corpus — overlap detected!", file=sys.stderr)
        sys.exit(1)
    print(f"✓ Disjoint from data/gutenberg_5m.txt", file=sys.stderr)

    Path(out_path).write_text(heldout, encoding='utf-8')
    print(f"Wrote {out_path} ({len(heldout)} chars)", file=sys.stderr)


if __name__ == '__main__':
    main()
