# MicroGPT project tasks

# Resolve absolute path for linker
root := `pwd`

# Build CLI tool (CPU backends only — uses C stubs for CUDA kernel symbols)
build:
    mkdir -p build
    cc -c -O2 src/cuda/stubs.c -o build/kernels.o
    timeout 3m crystal build src/microgpt/main.cr -o bin/microgpt --link-flags="{{root}}/build/kernels.o"

build-release:
    mkdir -p build
    cc -c -O2 src/cuda/stubs.c -o build/kernels.o
    timeout 3m crystal build src/microgpt/main.cr -o bin/microgpt --release --link-flags="{{root}}/build/kernels.o"

# Build with real CUDA kernels for GPU support
build-cuda:
    mkdir -p build
    /opt/cuda/bin/nvcc -c -O2 src/cuda/kernels.cu -o build/kernels.o
    timeout 3m crystal build src/microgpt/main.cr -o bin/microgpt --release --link-flags="{{root}}/build/kernels.o -lstdc++"

# Build radix-trie verify tool (CPU-only, Crystal)
build-radix-verify:
    mkdir -p build bin
    cc -c -O2 src/cuda/stubs.c -o build/kernels.o
    timeout 3m crystal build src/tools/radix_verify.cr -o bin/radix-verify --link-flags="{{root}}/build/kernels.o"

# Build perplexity eval tool (Crystal). Uses openblas or crystal backend by default.
# For cublas, rebuild the tool with real CUDA kernels linked.
build-perplexity:
    mkdir -p build bin
    /opt/cuda/bin/nvcc -c -O2 src/cuda/kernels.cu -o build/kernels.o
    timeout 3m crystal build src/tools/perplexity.cr -o bin/perplexity --release --link-flags="{{root}}/build/kernels.o -lstdc++"

# Build AGPT CUDA training engine (standalone GPU trainer)
build-agpt-train:
    mkdir -p bin
    /opt/cuda/bin/nvcc -O2 src/cuda/agpt_train.cu src/cuda/kernels.cu -lcublas -o bin/agpt_train

# Build cloud GPU CLI
build-cloud:
    mkdir -p bin
    timeout 3m crystal build src/cloud/cli.cr -o bin/cloud --release

# Build construction kit server (CPU)
build-kit:
    mkdir -p build bin
    cc -c -O2 src/cuda/stubs.c -o build/kernels.o
    timeout 3m crystal build src/construction_kit/server.cr -o bin/construction-kit --link-flags="{{root}}/build/kernels.o"

# Build construction kit CLI (CPU)
build-kit-cli:
    mkdir -p build bin
    cc -c -O2 src/cuda/stubs.c -o build/kernels.o
    timeout 3m crystal build src/construction_kit/cli.cr -o bin/construction-kit-cli --link-flags="{{root}}/build/kernels.o"

# Build construction kit server with CUDA
build-kit-cuda:
    mkdir -p build bin
    /opt/cuda/bin/nvcc -c -O2 src/cuda/kernels.cu -o build/kernels.o
    timeout 3m crystal build src/construction_kit/server.cr -o bin/construction-kit --release --link-flags="{{root}}/build/kernels.o -lstdc++"

# Run construction kit server
kit *ARGS:
    bin/construction-kit {{ARGS}}

# Run construction kit CLI
kit-cli *ARGS:
    bin/construction-kit-cli {{ARGS}}

# Run Svelte frontend dev server (proxy to kit on 8081)
kit-dev:
    cd src/construction_kit/frontend && npm run dev

# Build Svelte frontend for production
kit-build-frontend:
    cd src/construction_kit/frontend && npm run build

# Run with memory-limited shell (default 8 GiB virtual memory cap)
run *ARGS:
    OPENBLAS_NUM_THREADS=4 ulimit -v 8388608 && bin/microgpt {{ARGS}}

# Run cloud CLI
cloud *ARGS:
    bin/cloud {{ARGS}}

# Run all specs
test:
    crystal spec

# Generate all docs
docs: docs-tech docs-api

# Generate technical reference HTML from markdown
docs-tech:
    pandoc docs/tech/reference.md \
        -o docs/tech/index.html \
        --standalone \
        --toc \
        --toc-depth=3 \
        --metadata title="MicroGPT Technical Reference"

# Generate Crystal API docs
docs-api:
    crystal doc -o docs/api


# Run model benchmarks (default 100 steps)
bench steps="100":
    #!/usr/bin/env bash
    echo "Model          Steps    Avg Loss    Time"
    echo "-------------- -------- ----------- -----------"
    for heads in uniform exponential prime; do
        tmpout=$(mktemp)
        elapsed=$( { time just run data/input.txt --steps {{steps}} --heads $heads --backend openblas --no-save > "$tmpout" 2>&1; } 2>&1 | grep real | awk '{print $2}' )
        loss=$(grep "Final avg loss" "$tmpout" | awk '{print $4}')
        rm -f "$tmpout"
        printf "%-14s %8s %11s %11s\n" "$heads" "{{steps}}" "$loss" "$elapsed"
    done

# Compare standard window training against AGPT using a saved index
compare-agpt data="data/input.txt" steps="50" backend="crystal" seq_len="128" d_model="64" n_layers="2" agpt_starts="20000" agpt_offset="0" agpt_progress="0" seed="1234":
    STEPS={{steps}} \
    BACKEND={{backend}} \
    SEQ_LEN={{seq_len}} \
    D_MODEL={{d_model}} \
    N_LAYERS={{n_layers}} \
    AGPT_STARTS={{agpt_starts}} \
    AGPT_OFFSET={{agpt_offset}} \
    AGPT_PROGRESS={{agpt_progress}} \
    SEED={{seed}} \
    bash scripts/compare_window_agpt.sh {{data}}

# Run tokenizer benchmark (default 10 MB)
#bench-tokenizer size="10":
#    crystal build bench/bench_tokenizer.cr -o bench/bench_tokenizer --release
#    ./bench/bench_tokenizer {{size}}
#
# Run conversion benchmark (default 10000 rows)
#bench-convert rows="10000":
#    crystal build bench/bench_convert.cr -o bench/bench_convert --release
#    ./bench/bench_convert {{rows}}
#
# Run all benchmarks
#bench: bench-tokenizer bench-convert
