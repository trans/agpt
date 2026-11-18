#!/bin/sh
# Rebuild the partition-depth sweep's binaries (2026-04-30) from the code of
# that day: agpt at a8ddb15 (2026-04-29, the last commit before the sweep; it
# has every flag the sweep used) and the microgpt shard its shard.lock pins
# (bdd8e82), whose CUDA kernels that agpt_train linked. The build lines are
# a8ddb15's own Justfile recipes (build-stubs, build-agpt-build-radix-corpus,
# build-agpt-train). Outputs go to the machine-local trie cache, outside git.
# Usage: sh rnd/partition-depth-rerun/build_old.sh [WORKDIR]
set -eu
AGPT=$(pwd)
MGPT=${MGPT:-/home/trans/Projects/air/microgpt}
W=${1:-${TMPDIR:-/tmp}/agpt-a8ddb15}
OUT=/home/trans/agpt-tries/old-a8ddb15
[ -d "$W" ] || git -C "$AGPT" worktree add --detach "$W" a8ddb15
[ -d "$W.microgpt" ] || git -C "$MGPT" worktree add --detach "$W.microgpt" bdd8e82
mkdir -p "$W/lib" "$W/bin" "$W/build" "$OUT"
ln -sfn "$W.microgpt" "$W/lib/microgpt"
cd "$W"
nice -n 19 cc -c -O2 lib/microgpt/src/cuda/stubs.c -o build/kernels.o
nice -n 19 crystal build src/tools/build_radix_corpus.cr -o bin/agpt_build_radix_corpus --release \
  --link-flags="$W/build/kernels.o -lstdc++"
nice -n 19 /opt/cuda/bin/nvcc -O2 src/cuda/agpt_train.cu lib/microgpt/src/cuda/kernels.cu -lcublas -o bin/agpt_train
cp bin/agpt_train bin/agpt_build_radix_corpus "$OUT/"
sha256sum "$OUT/agpt_train" "$OUT/agpt_build_radix_corpus"
