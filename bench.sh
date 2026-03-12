#!/bin/bash

models=("uniform" "exponential" "prime")
steps=(10 100 1000)
trials=10

# Print header
printf "%-14s" "Model"
for s in "${steps[@]}"; do
    printf "  %12s" "${s} steps"
done
echo ""
printf "%-14s" "--------------"
for s in "${steps[@]}"; do
    printf "  %12s" "------------"
done
echo ""

for model in "${models[@]}"; do
    printf "%-14s" "$model"
    for s in "${steps[@]}"; do
        total=0.0
        for t in $(seq 1 $trials); do
            loss=$(./bin/microgpt data/input.txt "$s" "$model" 2>&1 | grep "Final avg loss" | awk '{print $4}')
            total=$(echo "$total + $loss" | bc -l)
        done
        avg=$(echo "scale=4; $total / $trials" | bc -l)
        printf "  %12s" "$avg"
    done
    echo ""
done
