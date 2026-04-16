"""
Story 1.5 — Stratified split and held-out set lock.

Produces:
  data/eval_set.parquet    — 600 items, public leaderboard
  data/held_out_set.parquet — 900 items, ESMA only, never used for leaderboard
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

DATA_DIR = Path(__file__).parents[2] / "data"

EVAL_SIZE = 600
HELD_OUT_SIZE = 900
RANDOM_SEED = 2026


def split_dataset() -> tuple[pd.DataFrame, pd.DataFrame]:
    pool_path = DATA_DIR / "curated_pool.parquet"
    eval_path = DATA_DIR / "eval_set.parquet"
    held_out_path = DATA_DIR / "held_out_set.parquet"

    if eval_path.exists() and held_out_path.exists():
        print("Split already exists.")
        return pd.read_parquet(eval_path), pd.read_parquet(held_out_path)

    pool = pd.read_parquet(pool_path)
    assert len(pool) >= EVAL_SIZE + HELD_OUT_SIZE, (
        f"Pool too small: {len(pool)} < {EVAL_SIZE + HELD_OUT_SIZE}"
    )

    # Stratify by domain to preserve distribution in both sets
    eval_df, held_out_df = train_test_split(
        pool,
        test_size=HELD_OUT_SIZE,
        train_size=EVAL_SIZE,
        stratify=pool["domain"],
        random_state=RANDOM_SEED,
    )
    eval_df = eval_df.reset_index(drop=True)
    held_out_df = held_out_df.reset_index(drop=True)

    eval_df.to_parquet(eval_path, index=False)
    held_out_df.to_parquet(held_out_path, index=False)

    print(f"Eval set:     {len(eval_df):,} items → {eval_path}")
    print(f"Held-out set: {len(held_out_df):,} items → {held_out_path}")
    print("\nEval domain distribution:")
    print(eval_df["domain"].value_counts().to_string())
    print("\nHeld-out domain distribution:")
    print(held_out_df["domain"].value_counts().to_string())

    return eval_df, held_out_df


if __name__ == "__main__":
    eval_df, held_out_df = split_dataset()
    print(f"\nEval difficulty: mean={eval_df['difficulty_estimate'].mean():.3f}")
    print(f"Held-out difficulty: mean={held_out_df['difficulty_estimate'].mean():.3f}")
    # Verify no overlap
    overlap = set(eval_df["id"]) & set(held_out_df["id"])
    assert not overlap, f"OVERLAP DETECTED: {overlap}"
    print("\nNo overlap between eval and held-out sets. ✓")
