"""
Reformat MCQ items to open-ended format.

Input:  data/curated_pool.parquet  (MCQ: correct_answer = letter, main_prompt has Options block)
Output: data/curated_pool.parquet  (open-ended: correct_answer = full option text, no Options block)
        data/eval_set.parquet      regenerated (600 items)
        data/held_out_set.parquet  regenerated (900 items)

Transformations per item:
  1. Parse options block "(A) text\n(B) text\n..." from main_prompt
  2. Look up full text of the correct option via current correct_answer letter
  3. Remove options block from main_prompt
  4. Replace "state your final answer as the option letter" instruction with
     "clearly state your final answer" (the reasoning instruction is kept)
  5. Prepend explicit reasoning instruction if not already present
  6. Set correct_answer = full text of the correct option
"""

import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

DATA_DIR = Path(__file__).parents[2] / "data"

EVAL_SIZE = 600
HELD_OUT_SIZE = 900
RANDOM_SEED = 2026

# Matches the full options section: from "Options:\n" to just before the reasoning line
# Captures everything between them, including blank lines and multi-line option bodies
_OPTIONS_SECTION_RE = re.compile(
    r"\nOptions:\n([\s\S]+?)(?=\nReason through|\Z)",
    re.MULTILINE,
)

# Old instruction variants to remove / replace
_OLD_INSTRUCTION_RE = re.compile(
    r"(?:Reason through this step by step,? then )?state your final answer as the option letter\.?",
    re.IGNORECASE,
)

_NEW_INSTRUCTION = (
    "Please think through this carefully and show your complete reasoning "
    "step by step before clearly stating your final answer."
)


def _parse_options(options_section: str) -> dict[str, str]:
    """
    Return {letter: full_text} from the raw options section string.

    Handles:
    - Single-line options: "(A) text\n(B) text\n..."
    - Blank lines between options: "(A) text\n\n(B) text\n..."
    - Multi-line option bodies: "(A) 1. step one\n2. step two\n(B) ..."
    """
    # Split on option markers (\n?(A), \n?(B), etc.)
    parts = re.split(r"\n?\(([A-J])\)\s*", options_section.strip())
    # parts layout: [pre_text, letter, body, letter, body, ...]
    options: dict[str, str] = {}
    for i in range(1, len(parts) - 1, 2):
        letter = parts[i]
        body = parts[i + 1].strip()
        options[letter] = body
    return options


def reformat_item(main_prompt: str, correct_answer: str) -> tuple[str, str]:
    """
    Returns (new_main_prompt, new_correct_answer).
    Raises ValueError if correct_answer letter not found in options block.
    """
    match = _OPTIONS_SECTION_RE.search(main_prompt)
    if match is None:
        # Already open-ended (re-run safe): return as-is
        return main_prompt, correct_answer

    options = _parse_options(match.group(1))
    letter = correct_answer.strip().upper()

    if letter not in options:
        raise ValueError(
            f"Correct answer letter '{letter}' not in options {list(options.keys())}"
        )

    new_correct_answer = options[letter]

    # Remove the options section ("\nOptions:\n..." through end of block)
    new_prompt = _OPTIONS_SECTION_RE.sub("", main_prompt)

    # Replace old instruction with new one
    if _OLD_INSTRUCTION_RE.search(new_prompt):
        new_prompt = _OLD_INSTRUCTION_RE.sub(_NEW_INSTRUCTION, new_prompt)
    else:
        # Append if no existing instruction found
        new_prompt = new_prompt.rstrip() + "\n" + _NEW_INSTRUCTION

    new_prompt = new_prompt.strip()
    return new_prompt, new_correct_answer


def reformat_pool(dry_run: bool = False) -> pd.DataFrame:
    pool_path = DATA_DIR / "curated_pool.parquet"
    pool = pd.read_parquet(pool_path)
    print(f"Loaded pool: {len(pool):,} items")
    print(f"Columns: {pool.columns.tolist()}")

    errors: list[str] = []
    new_prompts: list[str] = []
    new_answers: list[str] = []

    for idx, row in pool.iterrows():
        try:
            p, a = reformat_item(row["main_prompt"], row["correct_answer"])
            new_prompts.append(p)
            new_answers.append(a)
        except ValueError as e:
            errors.append(f"Row {idx} (id={row['id']}): {e}")
            new_prompts.append(row["main_prompt"])
            new_answers.append(row["correct_answer"])

    if errors:
        print(f"\n{len(errors)} errors:")
        for err in errors:
            print(" ", err)

    pool["main_prompt"] = new_prompts
    pool["correct_answer"] = new_answers

    # Spot-check
    sample = pool.sample(3, random_state=42)
    for _, row in sample.iterrows():
        print(f"\n--- id={row['id']} ---")
        print(f"main_prompt:\n{row['main_prompt']}")
        print(f"correct_answer: {row['correct_answer']!r}")

    if not dry_run:
        pool.to_parquet(pool_path, index=False)
        print(f"\nSaved reformatted pool → {pool_path}")

    return pool


def rebuild_splits(pool: pd.DataFrame) -> None:
    eval_path = DATA_DIR / "eval_set.parquet"
    held_out_path = DATA_DIR / "held_out_set.parquet"

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

    print(f"\nEval set:     {len(eval_df):,} → {eval_path}")
    print(f"Held-out set: {len(held_out_df):,} → {held_out_path}")

    overlap = set(eval_df["id"]) & set(held_out_df["id"])
    assert not overlap, f"ID overlap: {overlap}"
    print("No overlap between eval and held-out sets. ✓")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Reformat MCQ pool to open-ended")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print spot-check without writing files",
    )
    args = parser.parse_args()

    pool = reformat_pool(dry_run=args.dry_run)

    if not args.dry_run:
        rebuild_splits(pool)
        print("\nAll done. Re-upload dataset to Kaggle:")
        print("  bash kaggle_deploy/deploy.sh dataset")
