#!/usr/bin/env bash
# Reproduce the d* (branching depth) analysis across three corpora.
# Confirms the framing's first descriptive prediction:
#   d* ≈ log₂(N) / per-char-entropy
#
# Output: per-corpus mean d*, median d*, and a depth histogram.
# Expected (predicted vs observed):
#   Shakespeare 100k: 8.31 vs 7.94
#   Shakespeare 1M:   10.04 vs 9.71
#   Gutenberg 5M:     11.15 vs 11.23
set -eu -o pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$PWD}"
cd "$PROJECT_ROOT" || { echo "Cannot cd to PROJECT_ROOT=$PROJECT_ROOT" >&2; exit 1; }
[ -f Justfile ] || { echo "Run from agpt project root" >&2; exit 1; }

PROTO=rnd/trie-attention-framing/proto/branching_depth.py

# Build the tries we need (idempotent — skip if already present)
build_trie () {
    local corpus="$1" depth="$2" out="$3"
    if [ -d "$out" ]; then
        echo "[skip] trie already exists: $out"
        return 0
    fi
    echo "[build] $out"
    bin/agpt_build_radix_corpus --corpus "$corpus" --max-depth "$depth" --out "$out"
}

# Shakespeare 100k subset
if [ ! -f /tmp/shakespeare_100k.txt ]; then
    head -c 100000 data/input.txt > /tmp/shakespeare_100k.txt
fi

build_trie data/input.txt 32 /home/trans/agpt-tries/shakespeare_d32_radix_corpus
build_trie /tmp/shakespeare_100k.txt 32 /home/trans/agpt-tries/shakespeare_100k_d32_radix_corpus
# Gutenberg 5M trie is pre-existing in this repo.

echo
echo "=== Shakespeare 100k d=32 ==="
python3 "$PROTO" /home/trans/agpt-tries/shakespeare_100k_d32_radix_corpus

echo
echo "=== Shakespeare 1M d=32 ==="
python3 "$PROTO" /home/trans/agpt-tries/shakespeare_d32_radix_corpus

echo
echo "=== Gutenberg 5M d=32 ==="
python3 "$PROTO" /home/trans/agpt-tries/gutenberg_5m_d32_radix_corpus
