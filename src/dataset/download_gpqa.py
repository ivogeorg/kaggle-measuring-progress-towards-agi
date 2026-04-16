"""
GPQA Diamond supplement — run after approving access at:
https://huggingface.co/datasets/Idavidrein/gpqa

Merges GPQA Diamond into the existing curated pool and re-runs the split.
Run: HF_TOKEN=$HF_TOKEN python3 src/dataset/download_gpqa.py
"""

import os
from pathlib import Path

import pandas as pd
from datasets import load_dataset

DATA_DIR = Path(__file__).parents[2] / "data"
RAW_DIR = DATA_DIR / "raw"

META_QUESTION = (
    "Critically evaluate your proposed solution above using the following five steps:\n"
    "1. Restate your answer in one sentence.\n"
    "2. Identify the logical steps in your reasoning and assess each for potential errors.\n"
    "3. List the specific facts, assumptions, or knowledge gaps that could make your answer wrong.\n"
    "4. Confirm or revise your answer based on this evaluation.\n"
    "5. On a strict integer scale of 0 to 100, state your confidence that your final answer "
    "is completely correct. Respond only with the integer."
)


def download_and_merge():
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN not set. Run: source ~/.kaggle/set-api-key-var")

    print("Downloading GPQA Diamond...")
    ds = load_dataset("Idavidrein/gpqa", "gpqa_diamond", token=token)
    frames = [split.to_pandas() for split in ds.values()]
    gpqa_raw = pd.concat(frames, ignore_index=True)
    print(f"GPQA Diamond: {len(gpqa_raw):,} rows, columns: {list(gpqa_raw.columns)}")

    # Format to match curated_pool schema
    # GPQA columns: Question, Correct Answer, Incorrect Answer 1/2/3, ...
    def format_gpqa_row(row):
        options = [
            row.get("Correct Answer", ""),
            row.get("Incorrect Answer 1", ""),
            row.get("Incorrect Answer 2", ""),
            row.get("Incorrect Answer 3", ""),
        ]
        # Shuffle options so correct isn't always (A)
        import random
        rng = random.Random(hash(row.get("Question", "")))
        indices = list(range(4))
        rng.shuffle(indices)
        shuffled = [options[i] for i in indices]
        correct_pos = indices.index(0)
        labels = "ABCD"
        option_lines = "\n".join(f"({labels[i]}) {opt}" for i, opt in enumerate(shuffled))
        prompt = (
            f"{row['Question']}\n\nOptions:\n{option_lines}\n\n"
            "Reason through this step by step, then state your final answer as the option letter."
        )
        return prompt, labels[correct_pos]

    rows = []
    for i, (_, row) in enumerate(gpqa_raw.iterrows()):
        prompt, correct = format_gpqa_row(row)
        rows.append({
            "id": f"gpqa_{i:04d}",
            "domain": "reasoning",
            "source": "GPQA",
            "category": "gpqa_diamond",
            "main_prompt": prompt,
            "correct_answer": correct,
            "meta_question": META_QUESTION,
            "difficulty_estimate": 0.95,  # Expert-level: ~69% human expert accuracy
        })

    gpqa_df = pd.DataFrame(rows)
    gpqa_raw_path = RAW_DIR / "gpqa_diamond_raw.parquet"
    gpqa_df.to_parquet(gpqa_raw_path, index=False)
    print(f"Saved: {gpqa_raw_path}")

    # Merge with existing pool
    pool_path = DATA_DIR / "curated_pool.parquet"
    pool = pd.read_parquet(pool_path)
    print(f"Existing pool: {len(pool):,} items")

    merged = pd.concat([pool, gpqa_df], ignore_index=True)
    merged = merged.drop_duplicates(subset=["main_prompt"])

    # Re-index IDs
    merged["id"] = [f"mcbench_{i:05d}" for i in range(len(merged))]

    # Back up old pool, save new one
    pool_path.rename(pool_path.with_suffix(".parquet.bak"))
    merged.to_parquet(pool_path, index=False)
    print(f"Updated pool: {len(merged):,} items (+{len(gpqa_df)} GPQA)")
    print("Domain distribution:")
    print(merged["domain"].value_counts().to_string())

    # Re-run split
    print("\nRe-running split...")
    from src.dataset.split import split_dataset
    # Delete old split files so split_dataset re-creates them
    for p in [(DATA_DIR / "eval_set.parquet"), (DATA_DIR / "held_out_set.parquet")]:
        if p.exists():
            p.unlink()
    split_dataset()


if __name__ == "__main__":
    download_and_merge()
