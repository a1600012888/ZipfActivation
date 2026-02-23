#!/bin/bash
# Qwen3-4B-Base on OpenWebText with varying token budgets
set -e

cd /mnt/localssd/code/ZipfActivation
source venv/bin/activate

for TOKENS in 1000000 10000000 20000000 50000000 100000000; do
    LABEL=$(python -c "print(f'{$TOKENS/1e6:.0f}M')")
    DIR="results/Qwen3-4B-${LABEL}-openwebtext"
    echo "============================================"
    echo "Running Qwen3-4B-Base with ${LABEL} tokens"
    echo "Output: ${DIR}"
    echo "============================================"
    python scripts/run.py \
        --config configs/default.yaml \
        --data.max_tokens "$TOKENS" \
        --output.results_dir "$DIR" \
        --output.plot_format png
    echo "Done: ${DIR}"
    echo ""
done

echo "All token-scaling runs complete."
