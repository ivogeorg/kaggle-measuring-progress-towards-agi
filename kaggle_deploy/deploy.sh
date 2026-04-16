#!/usr/bin/env bash
# deploy.sh — Push MCBench to Kaggle via API
# Run from repo root: bash kaggle_deploy/deploy.sh [benchmark|esma|dataset|all]

set -e
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEPLOY_DIR="$REPO_ROOT/kaggle_deploy"

source ~/.kaggle/set-api-key-var 2>/dev/null || true

# Write kaggle.json from token if not present
if [ ! -f ~/.kaggle/kaggle.json ]; then
    echo '{"username":"ivogeorg","key":"'"${KAGGLE_API_TOKEN}"'"}' > ~/.kaggle/kaggle.json
    chmod 600 ~/.kaggle/kaggle.json
    echo "Created ~/.kaggle/kaggle.json"
fi

ACTION="${1:-all}"

# ── 1. Upload dataset ────────────────────────────────────────────────────────
upload_dataset() {
    echo "=== Uploading dataset (private) ==="
    STAGING=$(mktemp -d)
    trap "rm -rf $STAGING" EXIT

    # data/ subdirectory — code resolves DATA_DIR as parents[N]/"data"
    mkdir -p "$STAGING/data"
    cp "$REPO_ROOT/data/eval_set.parquet"    "$STAGING/data/"
    cp "$REPO_ROOT/data/held_out_set.parquet" "$STAGING/data/"
    cp -r "$REPO_ROOT/src" "$STAGING/"
    cp "$REPO_ROOT/requirements.txt" "$STAGING/"
    cp "$DEPLOY_DIR/dataset_metadata.json" "$STAGING/dataset-metadata.json"

    # Create or version
    if kaggle datasets status ivogeorg/mcbench-metacognitive-benchmark &>/dev/null; then
        echo "Dataset exists — creating new version"
        kaggle datasets version -p "$STAGING" -m "Update $(date +%Y-%m-%d)"
    else
        echo "Creating new private dataset"
        kaggle datasets create -p "$STAGING"
    fi
    echo "Dataset upload complete."
}

# ── 2. Push benchmark notebook ───────────────────────────────────────────────
push_benchmark() {
    echo "=== Pushing MCBench benchmark notebook ==="
    STAGING=$(mktemp -d)
    trap "rm -rf $STAGING" EXIT

    cp "$REPO_ROOT/mcbench.ipynb" "$STAGING/notebook.ipynb"
    cp "$DEPLOY_DIR/kernel_metadata_benchmark.json" "$STAGING/kernel-metadata.json"

    kaggle kernels push -p "$STAGING"
    echo "Benchmark notebook pushed. Monitoring..."
    sleep 5
    kaggle kernels status ivogeorg/mcbench-metacognition
    echo ""
    echo "Watch progress: kaggle kernels status ivogeorg/mcbench-metacognition"
}

# ── 3. Push ESMA notebook (GPU) ──────────────────────────────────────────────
push_esma() {
    echo "=== Pushing ESMA/Gemma4 notebook (GPU) ==="
    echo "NOTE: This runs on Kaggle GPU but does NOT submit to Gemma 4 Good Hackathon"
    STAGING=$(mktemp -d)
    trap "rm -rf $STAGING" EXIT

    cp "$REPO_ROOT/gemma4_esma.ipynb" "$STAGING/notebook.ipynb"
    cp "$DEPLOY_DIR/kernel_metadata_esma.json" "$STAGING/kernel-metadata.json"

    kaggle kernels push -p "$STAGING" --accelerator gpu
    echo "ESMA notebook pushed. Monitoring..."
    sleep 5
    kaggle kernels status ivogeorg/mcbench-esma-gemma4
    echo ""
    echo "Watch progress: kaggle kernels status ivogeorg/mcbench-esma-gemma4"
    echo "Get output:     kaggle kernels output ivogeorg/mcbench-esma-gemma4"
}

case "$ACTION" in
    dataset)  upload_dataset ;;
    benchmark) push_benchmark ;;
    esma)     push_esma ;;
    all)
        upload_dataset
        push_benchmark
        ;;
    *)
        echo "Usage: $0 [dataset|benchmark|esma|all]"
        exit 1
        ;;
esac
