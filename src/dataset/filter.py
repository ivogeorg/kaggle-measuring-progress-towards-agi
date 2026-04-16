"""
Stories 1.2 + 1.4 — Filter, balance, and curate the question pool.

Sources: MMLU-Pro (12k) → target 1,500 items
Strategy:
  - Format each MC question as an open-ended prompt (show options, ask for reasoning)
  - Cap each category at MAX_PER_CATEGORY items
  - Drop categories below MIN_CATEGORY_SIZE
  - Deduplicate by exact question text
  - Assign difficulty_estimate per category from empirical model accuracy literature

Output: data/curated_pool.parquet
"""

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parents[2] / "data"
RAW_DIR = DATA_DIR / "raw"

TARGET_POOL_SIZE = 1500
MAX_PER_CATEGORY = 160   # cap to force diversity
MIN_CATEGORY_SIZE = 30   # drop tiny categories

# Empirical difficulty per MMLU-Pro category
# Based on GPT-4 accuracy reported in TIGER-Lab/MMLU-Pro paper (lower = harder)
# Source: https://arxiv.org/abs/2406.01574, Table 2
CATEGORY_DIFFICULTY = {
    "math":             0.45,
    "physics":          0.52,
    "chemistry":        0.54,
    "engineering":      0.54,
    "law":              0.50,
    "economics":        0.58,
    "biology":          0.63,
    "health":           0.63,
    "psychology":       0.68,
    "business":         0.67,
    "philosophy":       0.66,
    "computer science": 0.68,
    "history":          0.70,
    "other":            0.62,
}

META_QUESTION = (
    "Critically evaluate your proposed solution above using the following five steps:\n"
    "1. Restate your answer in one sentence.\n"
    "2. Identify the logical steps in your reasoning and assess each for potential errors.\n"
    "3. List the specific facts, assumptions, or knowledge gaps that could make your answer wrong.\n"
    "4. Confirm or revise your answer based on this evaluation.\n"
    "5. On a strict integer scale of 0 to 100, state your confidence that your final answer "
    "is completely correct. Respond only with the integer."
)


def format_mc_prompt(question: str, options: list[str]) -> str:
    """Convert MC question + options to open-ended prompt that encourages full reasoning."""
    labels = "ABCDEFGHIJ"
    option_lines = "\n".join(
        f"({labels[i]}) {opt}" for i, opt in enumerate(options)
    )
    return (
        f"{question}\n\n"
        f"Options:\n{option_lines}\n\n"
        "Reason through this step by step, then state your final answer as the option letter."
    )


def build_pool() -> pd.DataFrame:
    out_path = DATA_DIR / "curated_pool.parquet"
    if out_path.exists():
        print(f"Curated pool already exists: {out_path}")
        return pd.read_parquet(out_path)

    # Load MMLU-Pro test split
    raw = pd.read_parquet(RAW_DIR / "mmlu_pro_raw.parquet")
    test = raw[raw["split"] == "test"].copy()
    print(f"MMLU-Pro test split: {len(test):,} items")

    # Drop items with malformed options (fewer than 4 choices)
    test["n_options"] = test["options"].apply(len)
    test = test[test["n_options"] >= 4]
    print(f"After option-count filter: {len(test):,}")

    # Deduplicate by exact question text
    before = len(test)
    test = test.drop_duplicates(subset=["question"])
    print(f"After dedup: {len(test):,} (removed {before - len(test)})")

    # Format as open-ended prompt
    test["main_prompt"] = test.apply(
        lambda r: format_mc_prompt(r["question"], r["options"]), axis=1
    )

    # Add correct_answer label for judge (letter form)
    labels = list("ABCDEFGHIJ")
    test["correct_answer"] = test["answer_index"].apply(
        lambda i: labels[i] if isinstance(i, int) and i < len(labels) else str(i)
    )

    # Add difficulty estimate
    test["difficulty_estimate"] = test["category"].map(CATEGORY_DIFFICULTY).fillna(0.6)

    # Map category to our domain taxonomy
    DOMAIN_MAP = {
        "math":             "math",
        "physics":          "reasoning",
        "chemistry":        "reasoning",
        "engineering":      "reasoning",
        "law":              "ethics",
        "economics":        "factual",
        "biology":          "factual",
        "health":           "factual",
        "psychology":       "epistemic",
        "business":         "factual",
        "philosophy":       "epistemic",
        "computer science": "reasoning",
        "history":          "factual",
        "other":            "factual",
    }
    test["domain"] = test["category"].map(DOMAIN_MAP).fillna("factual")

    # --- Balance: cap per category, prefer harder categories ---
    # Sort by difficulty ascending (hardest first) within each category
    test = test.sort_values("difficulty_estimate")

    capped_frames = []
    for cat, group in test.groupby("category"):
        if len(group) < MIN_CATEGORY_SIZE:
            print(f"  Skipping {cat}: only {len(group)} items")
            continue
        n = min(len(group), MAX_PER_CATEGORY)
        capped_frames.append(group.head(n))

    balanced = pd.concat(capped_frames, ignore_index=True)
    print(f"\nAfter per-category cap ({MAX_PER_CATEGORY}): {len(balanced):,} items")
    print("Category counts:", balanced["category"].value_counts().to_dict())

    # Sample down to TARGET_POOL_SIZE with stratified sampling
    if len(balanced) > TARGET_POOL_SIZE:
        rng = np.random.default_rng(42)
        domain_counts = balanced["domain"].value_counts()
        sampled_parts = []
        for domain, group in balanced.groupby("domain"):
            quota = int(np.ceil(TARGET_POOL_SIZE * len(group) / len(balanced)))
            quota = min(quota, len(group))
            idx = rng.choice(len(group), size=quota, replace=False)
            sampled_parts.append(group.iloc[idx])
        sampled = pd.concat(sampled_parts, ignore_index=True)
        # Trim to exact target if over
        if len(sampled) > TARGET_POOL_SIZE:
            sampled = sampled.sample(TARGET_POOL_SIZE, random_state=42)
        # Pad if under
        elif len(sampled) < TARGET_POOL_SIZE:
            remaining = balanced[~balanced.index.isin(sampled.index)]
            extra = remaining.sample(
                min(TARGET_POOL_SIZE - len(sampled), len(remaining)), random_state=42
            )
            sampled = pd.concat([sampled, extra], ignore_index=True)
        balanced = sampled.reset_index(drop=True)

    print(f"\nFinal pool: {len(balanced):,} items")

    # Build final schema
    balanced["id"] = [f"mcbench_{i:05d}" for i in range(len(balanced))]
    balanced["source"] = "MMLU-Pro"
    balanced["meta_question"] = META_QUESTION

    pool = balanced[[
        "id", "domain", "source", "category",
        "main_prompt", "correct_answer", "meta_question", "difficulty_estimate"
    ]].copy()

    pool.to_parquet(out_path, index=False)
    print(f"\nSaved to {out_path}")
    print("\nDomain distribution:")
    print(pool["domain"].value_counts().to_string())
    print("\nDifficulty stats:")
    print(pool["difficulty_estimate"].describe().to_string())

    return pool


if __name__ == "__main__":
    pool = build_pool()
    print(f"\nSample prompt:\n{pool.iloc[0]['main_prompt'][:400]}")
    print(f"\nMeta question (first 100 chars): {pool.iloc[0]['meta_question'][:100]}")
    print(f"Correct answer: {pool.iloc[0]['correct_answer']}")
