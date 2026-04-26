# AGPT project tasks
#
# AGPT (Aggregated-Gradient Pretraining) — research project on top of the
# µGPT components kit. CUDA kernels and model code are sourced from the
# µGPT shard at lib/microgpt/.

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
    /opt/cuda/bin/nvcc -O2 src/cuda/agpt_train.cu lib/microgpt/src/cuda/kernels.cu -lcublas -o bin/agpt_train

# Build the leveled-trie index builder.
build-agpt-build-index: build-stubs
    mkdir -p bin
    timeout 3m crystal build src/tools/build_index.cr -o bin/agpt_build_index --release --link-flags="{{root}}/build/kernels.o -lstdc++"

# Build the radix-trie builder.
build-agpt-build-radix: build-stubs
    mkdir -p bin
    timeout 3m crystal build src/tools/build_radix.cr -o bin/agpt_build_radix --release --link-flags="{{root}}/build/kernels.o -lstdc++"

# Build the corpus → radix builder. Bypasses the leveled-trie intermediate;
# bounded memory per root-character subtree. Use this for large corpora
# (5M+ chars at d=32) that OOM the leveled-then-radix pipeline.
build-agpt-build-radix-corpus: build-stubs
    mkdir -p bin
    timeout 3m crystal build src/tools/build_radix_corpus.cr -o bin/agpt_build_radix_corpus --release --link-flags="{{root}}/build/kernels.o -lstdc++"

# Build wrap-around corpus synthesizer.
build-synth-wrap-corpus: build-stubs
    mkdir -p bin
    timeout 3m crystal build src/tools/synth_wrap_corpus.cr -o bin/synth_wrap_corpus --release --link-flags="{{root}}/build/kernels.o -lstdc++"

# Build radix-trie verify tool.
build-radix-verify: build-stubs
    mkdir -p bin
    timeout 3m crystal build src/tools/radix_verify.cr -o bin/radix-verify --link-flags="{{root}}/build/kernels.o"

# Build trie sparsity-profile tool.
build-trie-profile: build-stubs
    mkdir -p bin
    timeout 3m crystal build src/tools/trie_profile.cr -o bin/trie-profile --link-flags="{{root}}/build/kernels.o"

# Build Bayesian posterior density tool.
build-bayesian-posterior: build-stubs
    mkdir -p bin
    timeout 3m crystal build src/tools/bayesian_posterior.cr -o bin/bayesian-posterior --link-flags="{{root}}/build/kernels.o"

# Build trie-path-probability convergence tool.
build-convergence: build-stubs
    mkdir -p bin
    timeout 3m crystal build src/tools/convergence.cr -o bin/convergence --link-flags="{{root}}/build/kernels.o"

# Build weight-diff tool used by foundational unit tests.
build-check-weights:
    mkdir -p bin
    gcc -O2 tools/check_weights.c -o bin/check_weights

# Build all AGPT binaries.
build-all: build-agpt-train build-agpt-build-index build-agpt-build-radix build-synth-wrap-corpus build-radix-verify build-trie-profile build-bayesian-posterior build-convergence build-check-weights

# Build µGPT inference / SGD reference binaries from the µGPT shard.
# Used by the foundational tests to compare AGPT against window-trained baselines.
build-microgpt-tools: build-stubs
    mkdir -p bin
    timeout 3m crystal build lib/microgpt/src/microgpt/main.cr -o bin/microgpt --release --link-flags="{{root}}/build/kernels.o -lstdc++"
    /opt/cuda/bin/nvcc -c -O2 lib/microgpt/src/cuda/kernels.cu -o build/kernels.o
    timeout 3m crystal build lib/microgpt/src/tools/perplexity.cr -o bin/perplexity --release --link-flags="{{root}}/build/kernels.o -lstdc++"
    cc -c -O2 lib/microgpt/src/cuda/stubs.c -o build/kernels.o

# Run foundational AGPT CUDA-trainer unit tests (gradient flow, radix build, training sanity).
test-agpt:
    bash tests/test_agpt_fundamentals.sh

# Run Crystal-side specs (backward attention, leveled trie, chain compression).
# Crystal's build links to CUDA kernel symbols via build/kernels.o; the CPU
# stubs satisfy them for spec compilation.
test-crystal: build-stubs
    crystal spec --link-flags="{{root}}/build/kernels.o -lstdc++"

# Run all tests.
test: test-crystal test-agpt

# Generate Crystal API docs.
docs: docs-api

docs-api:
    crystal doc -o docs/api
