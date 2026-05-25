#!/usr/bin/env bash
# KenLM modified-KN baseline at char level.
# Uses --discount_fallback because char-level vocabs (~65 types) are too small
# for KenLM's standard three-discount estimation.
#
# PPL numbers will differ slightly from nltk's KneserNeyInterpolated due to
# discount-strategy differences; use within one tool for cross-order comparison.
#
# Usage:
#   src/tools/kenlm_baseline.sh <train.txt> <heldout.txt> <order> [chunk_size]

set -e
TRAIN=${1:?train file required}
HELDOUT=${2:?heldout file required}
ORDER=${3:?order required}
CHUNK=${4:-1000}  # chars per heldout chunk to avoid huge single-line eval

KENLM=$(dirname "$0")/../../tools/kenlm/build/bin
WORK=$(mktemp -d -t kenlm_XXXXXX)
trap "rm -rf $WORK" EXIT

# Char-tokenize train: one big "sentence", space-separated chars.
# Real spaces in text replaced with '_' sentinel (KenLM treats whitespace
# as delimiter and would otherwise drop them entirely).
python3 -c "
with open('$TRAIN') as f: t = f.read().replace(' ', '_').replace('\n', ' ')
print(' '.join(c for c in t if c != ' ' or True))
" > "$WORK/train.txt"

# Heldout: same substitution, then chunk into $CHUNK-char lines so kenlm
# scores each as a separate sentence (boundary effect ~0.2% with chunk=1000).
python3 -c "
with open('$HELDOUT') as f: t = f.read().replace(' ', '_').replace('\n', ' ')
cs = [c for c in t if c != ' ']
C = $CHUNK
print('\n'.join(' '.join(cs[i:i+C]) for i in range(0, len(cs), C)))
" > "$WORK/heldout.txt"

# Train
"$KENLM/lmplz" -o "$ORDER" --discount_fallback < "$WORK/train.txt" > "$WORK/m.arpa" 2>/dev/null
"$KENLM/build_binary" -s "$WORK/m.arpa" "$WORK/m.bin" >/dev/null 2>&1

# Eval — extract PPL line
"$KENLM/query" -v summary "$WORK/m.bin" < "$WORK/heldout.txt" 2>&1 \
    | grep -E "Perplexity excluding|Tokens" \
    | head -2
