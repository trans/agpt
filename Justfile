# AGPT project tasks
#
# AGPT is its own project/repo. CUDA kernels and stubs live in src/cuda/
# (forked from microgpt 2026-05-22). The lib/microgpt shard is still used
# by Crystal-side tools for Mat / MiniGPT / Config / TextDataset primitives
# (~250 call sites across 11 files; not worth forking). The shard is
# treated as frozen — kernel/architecture evolution happens here, not there.

# Resolve absolute path for linker
root := `pwd`

# CUDA stubs (CPU-only). Most Crystal-side tools link a CPU-only build.o
# against these symbols so model code (which references CUDA symbols)
# can compile without pulling in libcudart. src/cuda/stubs.c is agpt's
# own copy (forked from lib/microgpt 2026-05-22 as part of microgpt
# severance).
build-stubs:
    mkdir -p build
    cc -c -O2 src/cuda/stubs.c -o build/kernels.o

# Build the shared GPU kernels object (µGPT shard's kernels.cu) once,
# so both v1 (src/cuda/agpt_train.cu) and v2 (src/cudax/agpt_train_v2.cu)
# link the SAME compiled kernel SASS. Previously each trainer recompiled
# kernels.cu inline, which produced slightly different register-allocation
# and intermediate-rounding decisions per binary — the v1↔v2 LayerNorm
# bit-mismatch traced back to this (resolved 2026-05-22).
#
# --use_fast_math is DROPPED here and on both training binaries below:
# nvcc's fast-math intrinsics (--prec-sqrt=false, --prec-div=false,
# --ftz=true) make individual fp32 ops non-bit-reproducible across
# runs — small drift in LN's 1/sqrt(var+ε), softmax exp/log, etc.
# Verified empirically 2026-05-22: with fast-math, even two consecutive
# runs of the same binary diverge in forward Q/K/V output; without it
# and with this shared kernels object, v1↔v2 forward is bit-identical
# on chunk 1 layer 0 (14/14 dumped tensors match).
#
# Cost: "modest" softmax slowdown vs the prior fast-math build (the
# original justification for --use_fast_math). Determinism by default
# is worth the cost — see notes/agpt/forward-parity.md if/when written.
#
# Distinct from build/kernels.o (CPU stubs for Crystal-side tools, see
# build-stubs); we use build/kernels_gpu.o to avoid filename collision.
build-cuda-kernels:
    mkdir -p build
    # Idempotent: skip rebuild if build/kernels_gpu.o exists and is newer
    # than the source. nvcc embeds non-determinism (timestamps?) so two
    # compilations of the same source produce different .o bytes — which
    # breaks v1↔v2 bit-parity if build-agpt-train and build-agpt-train-v2
    # are invoked in separate `just` runs and each rebuilds kernels.o.
    # Treating it as up-to-date when the source hasn't changed keeps both
    # trainers linked against the same .o across invocations.
    #
    # src/cuda/kernels.cu is agpt's own copy (forked from lib/microgpt
    # 2026-05-22 as part of severing microgpt dependency). Future kernel
    # evolution (deterministic LN reductions, atomicAdd-free backward,
    # etc.) happens here without coordinating with the moth-balled
    # microgpt project.
    if [ ! -e build/kernels_gpu.o ] || [ src/cuda/kernels.cu -nt build/kernels_gpu.o ]; then \
        /opt/cuda/bin/nvcc --allow-unsupported-compiler -std=c++17 -c -O3 \
            -gencode=arch=compute_80,code=sm_80 \
            -gencode=arch=compute_89,code=sm_89 \
            -gencode=arch=compute_90,code=sm_90 \
            src/cuda/kernels.cu -o build/kernels_gpu.o; \
    fi

# Build AGPT CUDA training engine (standalone GPU trainer).
# Links against the shared build/kernels_gpu.o (see build-cuda-kernels).
# No --use_fast_math: see build-cuda-kernels comment.
build-agpt-train: build-cuda-kernels
    mkdir -p bin
    # -gencode for sm_80 (Ampere A100), sm_89 (Ada RTX 40xx), and sm_90
    # (Hopper H100/H200) emits native SASS for all three — no JIT recompile
    # delay on first run on any of them.
    # If you have a 30xx (Ampere sm_86), add -gencode=arch=compute_86,code=sm_86.
    /opt/cuda/bin/nvcc --allow-unsupported-compiler -std=c++17 -O3 \
        -gencode=arch=compute_80,code=sm_80 \
        -gencode=arch=compute_89,code=sm_89 \
        -gencode=arch=compute_90,code=sm_90 \
        src/cuda/agpt_train.cu build/kernels_gpu.o -lcublas -o bin/agpt_train

# Build AGPT CUDA training engine v2 skeleton.
# Shares build/kernels_gpu.o with v1; same flags — load-bearing for
# v1↔v2 forward bit-parity. No --use_fast_math: see build-cuda-kernels.
build-agpt-train-v2: build-cuda-kernels
    mkdir -p bin
    /opt/cuda/bin/nvcc --allow-unsupported-compiler -std=c++17 -O3 \
        -gencode=arch=compute_80,code=sm_80 \
        -gencode=arch=compute_89,code=sm_89 \
        -gencode=arch=compute_90,code=sm_90 \
        src/cudax/agpt_train_v2.cu build/kernels_gpu.o -lcublas -o bin/agpt_train_v2

# Build standalone cudax seed-model generator.
build-agpt-seed:
    mkdir -p bin
    /opt/cuda/bin/nvcc --allow-unsupported-compiler -std=c++17 -O3 \
        src/cudax/agpt_seed.cu -o bin/agpt-seed

# Build the leveled-trie index builder.
build-agpt-build-index: build-stubs
    mkdir -p bin
    timeout 10m crystal build src/tools/build_index.cr -o bin/agpt_build_index --release --link-flags="{{root}}/build/kernels.o -lstdc++"

# Build the radix-trie builder.
build-agpt-build-radix: build-stubs
    mkdir -p bin
    timeout 10m crystal build src/tools/build_radix.cr -o bin/agpt_build_radix --release --link-flags="{{root}}/build/kernels.o -lstdc++"

# Build the corpus → radix builder. Bypasses the leveled-trie intermediate;
# bounded memory per root-character subtree. Use this for large corpora
# (5M+ chars at d=32) that OOM the leveled-then-radix pipeline.
build-agpt-build-radix-corpus: build-stubs
    mkdir -p bin
    timeout 10m crystal build src/tools/build_radix_corpus.cr -o bin/agpt_build_radix_corpus --release --link-flags="{{root}}/build/kernels.o -lstdc++"

# Build wrap-around corpus synthesizer.
build-synth-wrap-corpus: build-stubs
    mkdir -p bin
    timeout 10m crystal build src/tools/synth_wrap_corpus.cr -o bin/synth_wrap_corpus --release --link-flags="{{root}}/build/kernels.o -lstdc++"

# Build radix-trie verify tool.
build-radix-verify: build-stubs
    mkdir -p bin
    timeout 10m crystal build src/tools/radix_verify.cr -o bin/radix-verify --link-flags="{{root}}/build/kernels.o"

# Dump per-radix-node context strings (one per line, root-first chars) for
# downstream tools that compute per-node distributions.
build-dump-trie-contexts: build-stubs
    mkdir -p bin
    timeout 10m crystal build src/tools/dump_trie_contexts.cr -o bin/dump_trie_contexts --link-flags="{{root}}/build/kernels.o"

# Build per-substring position-distribution tables (catalog + radix→substring
# lookup + sparse position counts) for the multi-position encoding work.
build-agpt-build-position-table: build-stubs
    mkdir -p bin
    timeout 10m crystal build src/tools/build_position_table.cr -o bin/agpt_build_position_table --release --link-flags="{{root}}/build/kernels.o -lstdc++"

# Build trie sparsity-profile tool.
build-trie-profile: build-stubs
    mkdir -p bin
    timeout 10m crystal build src/tools/trie_profile.cr -o bin/trie-profile --link-flags="{{root}}/build/kernels.o"

# Build cap-folding side-table builder.
# Walks the prefix trie from root with each cap's tail (longest match,
# w_max..w_min) to produce per-cap fold-target distributions. Output is
# consumed by `agpt_train --fold-table PATH`.
build-agpt-build-fold-table: build-stubs
    mkdir -p bin
    timeout 10m crystal build src/tools/agpt_build_fold_table.cr -o bin/agpt_build_fold_table --release --link-flags="{{root}}/build/kernels.o -lstdc++"

# Build virtual-tree builder for cap-tunnel expansion.
# For each cap with edge length L, emits min(expansion_depth, L) composite
# distributions — one per tunnel position — formed as length-weighted
# mixtures of shifted-prefix walks. Drops the cap-as-SGD pathology by
# substituting non-degenerate targets at the +N tunnel positions while
# leaving the rest of the cap edge as is.
build-agpt-build-virtual-tree: build-stubs
    mkdir -p bin
    timeout 10m crystal build src/tools/agpt_build_virtual_tree.cr -o bin/agpt_build_virtual_tree --release --link-flags="{{root}}/build/kernels.o -lstdc++"

# Build virtual-tree inspector (CPU-side validator for VTRE side-tables).
build-agpt-inspect-virtual-tree: build-stubs
    mkdir -p bin
    timeout 10m crystal build src/tools/agpt_inspect_virtual_tree.cr -o bin/agpt_inspect_virtual_tree --release --link-flags="{{root}}/build/kernels.o -lstdc++"

# Build "parrot" sampler — generation by mass-weighted trie walk with
# cap-following (cap edge as directive into depth-1 root subtree).
# d-as-parrot-knob: larger d → more cap-following → more verbatim corpus output.
build-agpt-parrot-sample: build-stubs
    mkdir -p bin
    timeout 10m crystal build src/tools/agpt_parrot_sample.cr -o bin/agpt_parrot_sample --release --link-flags="{{root}}/build/kernels.o -lstdc++"

# Build sliding-window perplexity prototype. v1 of sliding-window AGPT
# inference (see notes/sliding_window_agpt.md, rnd/sliding-window-v1).
# Logit-pooling perplexity: for each target position, runs d contributing
# windows and pools their log-prob predictions.
build-agpt-sliding-window-perplexity:
    mkdir -p bin build
    /opt/cuda/bin/nvcc --allow-unsupported-compiler -std=c++17 -c -O3 --use_fast_math -gencode=arch=compute_80,code=sm_80 -gencode=arch=compute_89,code=sm_89 -gencode=arch=compute_90,code=sm_90 lib/microgpt/src/cuda/kernels.cu -o build/kernels.o
    timeout 10m crystal build src/tools/agpt_sliding_window_perplexity.cr -o bin/agpt_sliding_window_perplexity --release -Dpreview_mt --link-flags="{{root}}/build/kernels.o -L/opt/cuda/lib64 -lcudart -lcublas -lstdc++"
    cc -c -O2 src/cuda/stubs.c -o build/kernels.o

# Build position→node map tool. Phase 0 of seq_len decoupling: walks
# each corpus position's d-window through the radix trie and reports
# what nodes corpus positions land on. Optionally dumps a binary
# array `pos_to_node[p] = radix_id` for use by decoupled-attention
# experiments.
build-agpt-position-map: build-stubs
    mkdir -p bin
    timeout 10m crystal build src/tools/agpt_position_map.cr -o bin/agpt_position_map --release --link-flags="{{root}}/build/kernels.o -lstdc++"

# Build trie-only perplexity evaluator (model-free PPL baseline).
# Walks the radix trie with held-out context, scores log-prob from
# empirical count distributions at the deepest matching node with
# backoff. Direct comparison baseline for trained-model PPLs.
build-agpt-trie-perplexity: build-stubs
    mkdir -p bin
    timeout 10m crystal build src/tools/agpt_trie_perplexity.cr -o bin/agpt_trie_perplexity --release --link-flags="{{root}}/build/kernels.o -lstdc++"

# Build distribution-similarity diagnostic for radix-trie nodes.
build-agpt-dist-sim: build-stubs
    mkdir -p bin
    timeout 10m crystal build src/tools/agpt_dist_sim.cr -o bin/agpt_dist_sim --release --link-flags="{{root}}/build/kernels.o -lstdc++"

# Build wormhole-table builder for the topological-navigation experiment.
# Per cap, emits a re-entry edge to a prefix-trie node (depth-1 by default).
# Variants:
#   v1 — first-char of cap → depth-1 root child (zero suffix info)
#   v2 — boundary-char (where suffix entropy crosses 0) → depth-1 root child
build-agpt-build-wormhole-table: build-stubs
    mkdir -p bin
    timeout 10m crystal build src/tools/agpt_build_wormhole_table.cr -o bin/agpt_build_wormhole_table --release --link-flags="{{root}}/build/kernels.o -lstdc++"

# Build wormhole sampler. Walks the prefix trie from root, sampling next-char
# at each step from the empirical distribution. At cap heads, jumps via the
# wormhole side-table to a re-entry node. Output: sampled paths suitable as
# training sequences for SGD/LM training.
build-agpt-wormhole-sample: build-stubs
    mkdir -p bin
    timeout 10m crystal build src/tools/agpt_wormhole_sample.cr -o bin/agpt_wormhole_sample --release --link-flags="{{root}}/build/kernels.o -lstdc++"

# Build prefix/suffix model comparison tool.
# Loads a forward model (trained on prefix trie) and backward model (trained on
# suffix/reversed-corpus trie); reports KL between their predictions at held-out
# positions plus per-model NLL.
build-prefix-suffix-compare: build-stubs
    mkdir -p bin
    timeout 10m crystal build src/tools/prefix_suffix_compare.cr -o bin/prefix_suffix_compare --release --link-flags="{{root}}/build/kernels.o -lstdc++"

# Build dual-view consistency trainer (forward + backward models, KL coupling).
# First-version per-position Adam fire; per-partition batching is a future
# optimization. See rnd/dual-model-fold/PLAN.md.
build-agpt-dual-train: build-stubs
    mkdir -p bin
    timeout 10m crystal build src/tools/agpt_dual_train.cr -o bin/agpt_dual_train --release --link-flags="{{root}}/build/kernels.o -lstdc++"

# Build p2s-attention match index tool (Phase 2/3 of rnd/p2s-attention).
build-p2s-match: build-stubs
    mkdir -p bin
    timeout 10m crystal build src/tools/p2s_match_index.cr -o bin/agpt_p2s_match --release --link-flags="{{root}}/build/kernels.o -lstdc++"

# Build p2s-attention match inspector tool.
build-p2s-inspect: build-stubs
    mkdir -p bin
    timeout 10m crystal build src/tools/p2s_inspect.cr -o bin/agpt_p2s_inspect --release --link-flags="{{root}}/build/kernels.o -lstdc++"

# Build Bayesian posterior density tool.
build-bayesian-posterior: build-stubs
    mkdir -p bin
    timeout 10m crystal build src/tools/bayesian_posterior.cr -o bin/bayesian-posterior --link-flags="{{root}}/build/kernels.o"

# Build trie-path-probability convergence tool.
build-convergence: build-stubs
    mkdir -p bin
    timeout 10m crystal build src/tools/convergence.cr -o bin/convergence --link-flags="{{root}}/build/kernels.o"

# Build weight-diff tool used by foundational unit tests.
build-check-weights:
    mkdir -p bin
    gcc -O2 tools/check_weights.c -o bin/check_weights

# Build checkpoint comparer used to inspect seed/init equivalence.
build-compare-checkpoints:
    mkdir -p bin
    gcc -O2 tools/compare_checkpoints.c -lm -o bin/compare_checkpoints

# Build all AGPT-native binaries.
build-all: build-agpt-train build-agpt-build-index build-agpt-build-radix build-synth-wrap-corpus build-radix-verify build-trie-profile build-bayesian-posterior build-convergence build-check-weights

# µGPT reference binaries (bin/microgpt, bin/perplexity) are not built
# from this project anymore. microgpt is moth-balled and lives in its
# own project (/home/trans/Projects/microgpt). When the AGPT
# fundamental-parity tests need bin/microgpt / bin/perplexity, build
# them there (cd /path/to/microgpt && just build-cuda build-perplexity)
# and symlink or copy the binaries into agpt/bin/. Severance started
# 2026-05-22 (kernels.cu copied to src/cuda/kernels.cu, ef3a8e9);
# this Justfile no longer touches microgpt sources directly.

# Run AGPT foundational parity tests (gradient flow, radix build, training sanity).
test-agpt:
    bash tests/test_agpt_fundamentals.sh

# Run AGPT-native Crystal specs (backward attention, leveled trie, chain compression).
# Crystal's build links to CUDA kernel symbols via build/kernels.o; the CPU stubs
# satisfy them for spec compilation.
test-crystal: build-stubs
    crystal spec --link-flags="{{root}}/build/kernels.o -lstdc++"

# Run AGPT-native specs plus AGPT foundational parity tests.
test: test-crystal test-agpt

# Generate Crystal API docs.
docs: docs-api

docs-api:
    crystal doc -o docs/api

# Simple sanity run. Trains 5 SE then evaluates held-out PPL with the
# canonical AGPT sliding-window tool. (bin/perplexity from microgpt is
# no longer built here — see severance note above.)
quick-test:
    cp data/input.random.model /tmp/quick.model && \
    bin/agpt_train \
        --model /tmp/quick.model --trie-dir /tmp/shake_baseline_d16_radix \
        --save /tmp/quick.model --epochs 5 \
        --partition-depth 1 --no-accumulate \
        --lr 3e-3 --lr-schedule warmup-cosine --warmup-epochs 1 \
        --optimizer rmsprop --rmsprop-beta 0.999 \
        --mass-weight log --entropy-lambda 1.0 \
        | tail -5 && \
    bin/agpt_sliding_window_perplexity --model /tmp/quick.model \
        --file data/input.txt --vocab-file data/input.txt \
        --d 16 --backend openblas --max-positions 4096 --workers 4 \
        | tail -4
