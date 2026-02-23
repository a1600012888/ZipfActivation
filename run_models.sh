#!/bin/bash
# Multiple Qwen3 model sizes on OpenWebText with 10M token budget
set -e

cd /mnt/localssd/code/ZipfActivation
source venv/bin/activate

TOKENS=10000000

for MODEL in \
    "Qwen/Qwen3-0.6B-Base" \
    "Qwen/Qwen3-1.7B-Base" \
    "Qwen/Qwen3-4B-Base" \
    "Qwen/Qwen3-8B-Base" \
    "Qwen/Qwen3-14B-Base"; do

    SIZE=$(echo "$MODEL" | grep -oP '[\d.]+B')
    DIR="results/Qwen3-${SIZE}-10M-openwebtext"
    echo "============================================"
    echo "Running ${MODEL} with 10M tokens"
    echo "Output: ${DIR}"
    echo "============================================"
    python scripts/run.py \
        --config configs/default.yaml \
        --model.name "$MODEL" \
        --data.max_tokens "$TOKENS" \
        --output.results_dir "$DIR" \
        --output.plot_format png
    echo "Done: ${DIR}"
    echo ""
done

echo "All model-scaling runs complete."
