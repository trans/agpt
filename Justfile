# MicroGPT project tasks

# Resolve absolute path for linker
root := `pwd`

# Build CLI tool (CPU backends only — uses C stubs for CUDA kernel symbols)
build:
    mkdir -p build
    cc -c -O2 src/cuda/stubs.c -o build/kernels.o
    crystal build src/microgpt/main.cr -o bin/microgpt --link-flags="{{root}}/build/kernels.o"

build-release:
    mkdir -p build
    cc -c -O2 src/cuda/stubs.c -o build/kernels.o
    crystal build src/microgpt/main.cr -o bin/microgpt --release --link-flags="{{root}}/build/kernels.o"

# Build with real CUDA kernels for GPU support
build-cuda:
    mkdir -p build
    /opt/cuda/bin/nvcc -c -O2 src/cuda/kernels.cu -o build/kernels.o
    crystal build src/microgpt/main.cr -o bin/microgpt --release --link-flags="{{root}}/build/kernels.o -lstdc++"

# Build cloud GPU CLI
build-cloud:
    mkdir -p bin
    crystal build src/cloud/cli.cr -o bin/cloud --release

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
