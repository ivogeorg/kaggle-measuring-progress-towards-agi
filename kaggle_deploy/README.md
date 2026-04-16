# Kaggle Deploy Directory

Files for pushing MCBench to Kaggle via the API.

## Structure
- `dataset_metadata.json`   → for `kaggle datasets create`
- `kernel_metadata_benchmark.json` → for `kaggle kernels push` (MCBench benchmark)
- `kernel_metadata_esma.json`      → for `kaggle kernels push` (Gemma 4 ESMA)

## Steps

### 1. Upload dataset (private, GPQA-safe)
```bash
# From repo root
source ~/.kaggle/set-api-key-var
mkdir -p /tmp/mcbench_dataset
cp data/eval_set.parquet /tmp/mcbench_dataset/
cp src/ /tmp/mcbench_dataset/ -r
cp requirements.txt /tmp/mcbench_dataset/
cp kaggle_deploy/dataset_metadata.json /tmp/mcbench_dataset/
kaggle datasets create -p /tmp/mcbench_dataset
```

### 2. Push benchmark notebook
```bash
mkdir -p /tmp/mcbench_kernel
cp mcbench.ipynb /tmp/mcbench_kernel/notebook.ipynb
cp kaggle_deploy/kernel_metadata_benchmark.json /tmp/mcbench_kernel/kernel-metadata.json
kaggle kernels push -p /tmp/mcbench_kernel
kaggle kernels status ivogeorg/mcbench-metacognition
```

### 3. Push ESMA notebook (Gemma 4 Good Hackathon — run only, don't submit yet)
```bash
mkdir -p /tmp/esma_kernel
cp gemma4_esma.ipynb /tmp/esma_kernel/notebook.ipynb
cp kaggle_deploy/kernel_metadata_esma.json /tmp/esma_kernel/kernel-metadata.json
kaggle kernels push -p /tmp/esma_kernel --accelerator gpu
kaggle kernels status ivogeorg/mcbench-esma-gemma4
```
