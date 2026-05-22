# AGPT project tasks
#
# AGPT is its own project/repo. It depends on the µGPT shard for model/runtime
# primitives, shared CUDA kernels, and reference comparison binaries.

# Resolve absolute path for linker
root := `pwd`

# CUDA stubs (CPU-only) sourced from the µGPT shard. Most Crystal-side
# tools link a CPU-only build.o against these symbols so the µGPT model
# code (which references CUDA symbols) can compile.
build-stubs:
    mkdir -p build
    cc -c -O2 lib/microgpt/src/cuda/stubs.c -o build/kernels.o

# Build AGPT CUDA training engine (standalone GPU trainer).
# Sources kernels.cu from the µGPT shard.
build-agpt-train:
    mkdir -p bin
    # -O3 + --use_fast_math: modest speedup on attention softmax (exp/log).
    # -gencode for sm_80 (Ampere A100), sm_89 (Ada RTX 40xx), and sm_90
    # (Hopper H100/H200) emits native SASS for all three — no JIT recompile
    # delay on first run on any of them.
    # If you have a 30xx (Ampere sm_86), add -gencode=arch=compute_86,code=sm_86.
    /opt/cuda/bin/nvcc --allow-unsupported-compiler -std=c++17 -O3 --use_fast_math \
        -gencode=arch=compute_80,code=sm_80 \
        -gencode=arch=compute_89,code=sm_89 \
        -gencode=arch=compute_90,code=sm_90 \
        src/cuda/agpt_train.cu lib/microgpt/src/cuda/kernels.cu -lcublas -o bin/agpt_train

# Build AGPT CUDA training engine v2 skeleton.
# This is a separate trainer-core rewrite path; keep it independent of the
# current agpt_train baseline until parity is established.
build-agpt-train-v2:
    mkdir -p bin
    /opt/cuda/bin/nvcc --allow-unsupported-compiler -std=c++17 -O3 --use_fast_math \
        -gencode=arch=compute_80,code=sm_80 \
        -gencode=arch=compute_89,code=sm_89 \
        -gencode=arch=compute_90,code=sm_90 \
        src/cudax/agpt_train_v2.cu lib/microgpt/src/cuda/kernels.cu -lcublas -o bin/agpt_train_v2

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
    cc -c -O2 lib/microgpt/src/cuda/stubs.c -o build/kernels.o

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

# Build all AGPT-native binaries.
build-all: build-agpt-train build-agpt-build-index build-agpt-build-radix build-synth-wrap-corpus build-radix-verify build-trie-profile build-bayesian-posterior build-convergence build-check-weights

# Build µGPT reference binaries from the shard.
# Used only for baseline comparison / parity tests, not for ordinary AGPT builds.
build-microgpt-tools:
    mkdir -p bin build
    /opt/cuda/bin/nvcc --allow-unsupported-compiler -std=c++17 -c -O3 --use_fast_math -gencode=arch=compute_80,code=sm_80 -gencode=arch=compute_89,code=sm_89 -gencode=arch=compute_90,code=sm_90 lib/microgpt/src/cuda/kernels.cu -o build/kernels.o
    timeout 10m crystal build lib/microgpt/src/microgpt/main.cr -o bin/microgpt --release --link-flags="{{root}}/build/kernels.o -L/opt/cuda/lib64 -lcudart -lcublas -lstdc++"
    timeout 10m crystal build lib/microgpt/src/tools/perplexity.cr -o bin/perplexity --release --link-flags="{{root}}/build/kernels.o -L/opt/cuda/lib64 -lcudart -lcublas -lstdc++"
    cc -c -O2 lib/microgpt/src/cuda/stubs.c -o build/kernels.o

# Alias to make the AGPT-vs-reference boundary explicit in local workflows.
build-reference-tools: build-microgpt-tools

# Run AGPT foundational parity tests (gradient flow, radix build, training sanity).
# These depend on the µGPT reference binaries.
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

# Simple sanity run
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
    bin/perplexity --model /tmp/quick.model --file data/input.txt \
        --seq-len 16 --backend openblas --max-positions 4096 \
        | tail -4
