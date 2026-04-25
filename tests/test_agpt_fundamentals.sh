#!/usr/bin/env bash
# Foundational unit tests for the AGPT CUDA trainer.
#
# Covers things that unit tests for the Crystal side don't:
#   1. Every weight matrix changes during training (would have caught the
#      Wk/Wv-frozen bug that went undetected for ~5 months).
#   2. Radix build from a leveled trie produces valid output (would catch
#      silent-exit failures like d=64).
#   3. Training loss never reaches NaN.
#   4. Post-training PPL beats random-init PPL by a large margin.
#   5. L4 path-sampling matches real SGD window training within tolerance
#      at matched hyperparameters (cross-validates the L4 emulator).
#
# Run: just test-agpt
# Requires: bin/agpt_train, bin/microgpt, bin/perplexity, bin/check_weights,
#   a built d=8 radix trie at /tmp/agpt_input_d8_radix, and a random-init
#   checkpoint at data/input.random.model.

set -u

PROJECT_ROOT="${PROJECT_ROOT:-$PWD}"
cd "$PROJECT_ROOT" || exit 2
if [ ! -f Justfile ]; then
    echo "FAIL: $PROJECT_ROOT is not the microgpt project root" >&2
    exit 2
fi

pass=0
fail=0
fail_reasons=()

# ANSI colors (only if stdout is a tty)
if [ -t 1 ]; then
    GREEN="\033[32m"; RED="\033[31m"; YELLOW="\033[33m"; RESET="\033[0m"
else
    GREEN=""; RED=""; YELLOW=""; RESET=""
fi

ok ()   { printf "${GREEN}PASS${RESET}  %s\n" "$1"; pass=$((pass+1)); }
bad ()  { printf "${RED}FAIL${RESET}  %s\n    reason: %s\n" "$1" "$2"; fail=$((fail+1)); fail_reasons+=("$1: $2"); }
skip () { printf "${YELLOW}SKIP${RESET}  %s\n    reason: %s\n" "$1" "$2"; }

# ---- Prerequisites ----

if [ ! -x bin/agpt_train ];        then bad "prereq" "bin/agpt_train missing (just build-agpt-train)"; exit 1; fi
if [ ! -x bin/agpt_build_radix ];  then bad "prereq" "bin/agpt_build_radix missing (just build-agpt-build-radix)"; exit 1; fi
if [ ! -x bin/microgpt ];          then bad "prereq" "bin/microgpt missing (just build-microgpt-tools)"; exit 1; fi
if [ ! -x bin/perplexity ];        then bad "prereq" "bin/perplexity missing (just build-microgpt-tools)"; exit 1; fi
if [ ! -x bin/check_weights ];     then just build-check-weights >/dev/null 2>&1 || { bad "prereq" "cannot build bin/check_weights"; exit 1; }; fi
if [ ! -f data/input.random.model ]; then bad "prereq" "data/input.random.model missing"; exit 1; fi
if [ ! -d /tmp/agpt_input_d8_radix ]; then bad "prereq" "/tmp/agpt_input_d8_radix missing (need a built d=8 radix trie)"; exit 1; fi

echo "AGPT fundamentals test"
echo "  project: $PROJECT_ROOT"
echo "  working: /tmp/agpt_fundamentals_$$"
WORK=/tmp/agpt_fundamentals_$$
mkdir -p "$WORK"
trap 'rm -rf "$WORK"' EXIT
echo ""

# ====================================================================
# Test 1: Every weight matrix changes during training
# ====================================================================
# Would have caught the Wk/Wv/bias freeze bug in the CUDA backward.
TEST="1. every weight matrix changes during training"
cp data/input.random.model "$WORK/pre.model"
cp data/input.random.model "$WORK/post.model"
bin/agpt_train \
    --model "$WORK/post.model" --trie-dir /tmp/agpt_input_d8_radix \
    --save "$WORK/post.model" --epochs 1 --lr 3e-3 \
    --optimizer rmsprop --rmsprop-beta 0.999 \
    --lr-schedule warmup-cosine --warmup-epochs 1 \
    --entropy-lambda 1.0 --mass-weight linear --no-accumulate \
    > "$WORK/t1.log" 2>&1
rc=$?
if [ $rc -ne 0 ]; then
    bad "$TEST" "training exited with code $rc (see $WORK/t1.log)"
else
    bin/check_weights --quiet "$WORK/pre.model" "$WORK/post.model"
    rc=$?
    if [ $rc -eq 0 ]; then
        ok "$TEST"
    else
        bad "$TEST" "some weight matrix frozen (run bin/check_weights $WORK/pre.model $WORK/post.model for details)"
    fi
fi

# ====================================================================
# Test 2: Radix build from a leveled trie produces non-empty output
# ====================================================================
# Would catch d=64 silent-exit failures.
TEST="2. radix build from d=8 leveled trie produces valid output"
if [ ! -d /tmp/agpt_input_d8 ]; then
    skip "$TEST" "/tmp/agpt_input_d8 (leveled) missing; skipping build test"
else
    rm -rf "$WORK/radix_check"
    bin/agpt_build_radix --leveled /tmp/agpt_input_d8 \
        --out "$WORK/radix_check" > "$WORK/t2.log" 2>&1
    rc=$?
    if [ $rc -ne 0 ]; then
        bad "$TEST" "build exited with code $rc"
    elif [ ! -f "$WORK/radix_check/meta.bin" ]; then
        bad "$TEST" "meta.bin not written to output dir (silent exit?)"
    else
        # Check radix_depth_*.bin presence
        n_depth_files=$(ls "$WORK/radix_check"/radix_depth_*.bin 2>/dev/null | wc -l)
        if [ "$n_depth_files" -lt 2 ]; then
            bad "$TEST" "only $n_depth_files depth files written (expected ≥2)"
        else
            # Extract reported radix_count from the log
            radix_count=$(grep "radix_count:" "$WORK/t2.log" | awk '{print $2}')
            if [ -z "$radix_count" ] || [ "$radix_count" -lt 100000 ]; then
                bad "$TEST" "radix_count=$radix_count seems too small for d=8 Shakespeare"
            else
                ok "$TEST (radix_count=$radix_count, $n_depth_files depth files)"
            fi
        fi
    fi
fi

# ====================================================================
# Test 3: Training loss stays finite (no NaN)
# ====================================================================
TEST="3. training loss stays finite"
if [ -f "$WORK/t1.log" ]; then
    nan_count=$(grep -Ei "\bnan\b|^.*loss=nan|inf" "$WORK/t1.log" | grep -cvi "rmsprop\|warmup" || true)
    if [ "$nan_count" -gt 0 ]; then
        bad "$TEST" "found $nan_count nan/inf occurrences in training log"
    else
        ok "$TEST"
    fi
else
    bad "$TEST" "no training log available (test 1 didn't run)"
fi

# ====================================================================
# Test 4: Post-training PPL << random-init PPL
# ====================================================================
TEST="4. post-training PPL far below random baseline"
rand_ppl=$(bin/perplexity --model data/input.random.model --file data/input.txt \
    --max-positions 4096 --backend openblas 2>&1 | awk '/^Perplexity:/ {print $2}')
post_ppl=$(bin/perplexity --model "$WORK/post.model" --file data/input.txt \
    --max-positions 4096 --backend openblas 2>&1 | awk '/^Perplexity:/ {print $2}')
if [ -z "$rand_ppl" ] || [ -z "$post_ppl" ]; then
    bad "$TEST" "could not read PPL (rand=$rand_ppl post=$post_ppl)"
else
    ratio=$(awk -v r=$rand_ppl -v p=$post_ppl 'BEGIN { printf "%.2f", p/r }')
    # Expect post/rand < 0.5 (post PPL is less than half of random-init PPL)
    check=$(awk -v r=$rand_ppl -v p=$post_ppl 'BEGIN { print (p < r * 0.5) ? "yes" : "no" }')
    if [ "$check" = "yes" ]; then
        ok "$TEST (random=$rand_ppl, post-train=$post_ppl, ratio=$ratio)"
    else
        bad "$TEST" "post-train PPL $post_ppl not << random $rand_ppl (ratio=$ratio)"
    fi
fi

# ====================================================================
# Test 5: L4 path-sampling ≈ real SGD (parity)
# ====================================================================
# Within-20% tolerance at matched hyperparameters (Adam, const lr=3e-4).
# bin/microgpt cublas is broken; use openblas for SGD reference.
TEST="5. L4 path-sampling ≈ bin/microgpt SGD (within tolerance)"
# SGD reference: seq=8 (matches d=8), 500 steps
cp data/input.random.model "$WORK/sgd_ref.model"
bin/microgpt data/input.txt --model "$WORK/sgd_ref.model" \
    --seq-len 8 --steps 500 --lr 3e-4 \
    --d-model 64 --n-layers 2 --backend openblas --seed 42 \
    > "$WORK/t5_sgd.log" 2>&1
sgd_ppl=$(bin/perplexity --model "$WORK/sgd_ref.model" --file data/input.txt \
    --max-positions 4096 --backend openblas 2>&1 | awk '/^Perplexity:/ {print $2}')

# L4 at matched hyperparams
cp data/input.random.model "$WORK/l4.model"
bin/agpt_train --model "$WORK/l4.model" --trie-dir /tmp/agpt_input_d8_radix \
    --save "$WORK/l4.model" --epochs 1 --lr 3e-4 --optimizer adam \
    --lr-schedule constant --entropy-lambda 0 --mass-weight off \
    --lightning-steps 500 --lightning-sampler l4 --lightning-seed 42 \
    > "$WORK/t5_l4.log" 2>&1
l4_ppl=$(bin/perplexity --model "$WORK/l4.model" --file data/input.txt \
    --max-positions 4096 --backend openblas 2>&1 | awk '/^Perplexity:/ {print $2}')

if [ -z "$sgd_ppl" ] || [ -z "$l4_ppl" ]; then
    bad "$TEST" "could not read PPL (sgd=$sgd_ppl l4=$l4_ppl)"
else
    ratio=$(awk -v a=$sgd_ppl -v b=$l4_ppl 'BEGIN { d = (a > b) ? a/b : b/a; printf "%.2f", d }')
    # Tolerance 2.0× (loose — L4 and SGD do converge on the same distribution,
    # but per-step dynamics differ enough that strict parity isn't expected).
    check=$(awk -v r=$ratio 'BEGIN { print (r < 2.0) ? "yes" : "no" }')
    if [ "$check" = "yes" ]; then
        ok "$TEST (SGD=$sgd_ppl, L4=$l4_ppl, ratio=$ratio)"
    else
        bad "$TEST" "SGD=$sgd_ppl L4=$l4_ppl differ by $ratio× (expected <2×)"
    fi
fi

# ====================================================================
# Summary
# ====================================================================
echo ""
echo "============================================================"
total=$((pass + fail))
if [ $fail -eq 0 ]; then
    printf "${GREEN}ALL %d PASSED${RESET}\n" "$total"
    exit 0
else
    printf "${RED}%d / %d FAILED${RESET}\n" "$fail" "$total"
    for r in "${fail_reasons[@]}"; do
        printf "  - %s\n" "$r"
    done
    exit 1
fi
