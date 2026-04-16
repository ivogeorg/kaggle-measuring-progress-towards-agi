"""
Story 2.1 — Run 1: Zero-shot extraction with judge scoring.

For each prompt in the eval set:
  - Send main_prompt to the target model in a fresh chat context
  - Judge evaluates correctness against the known correct answer
  - Persist (model_name, prompt_id, response_text, is_correct) for decoy construction

kbench runtime: kbench.llm and kbench.judge_llm are injected by the Kaggle notebook env.
This module defines the task and the persistence helpers used between runs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import kaggle_benchmarks as kbench
import pandas as pd

DATA_DIR = Path(__file__).parents[2] / "data"
RUN1_RESPONSES_PATH = DATA_DIR / "run1_responses.parquet"


# ---------------------------------------------------------------------------
# Persistence helpers (called as side-effects inside the kbench task)
# ---------------------------------------------------------------------------

_run1_buffer: list[dict] = []


def _store_run1_response(
    prompt_id: str,
    model_name: str,
    response: str,
    is_correct: bool,
) -> None:
    """Append a single Run 1 result to the in-memory buffer."""
    _run1_buffer.append({
        "prompt_id": prompt_id,
        "model_name": model_name,
        "response": response,
        "is_correct": is_correct,
    })


def flush_run1_responses() -> pd.DataFrame:
    """Write the buffer to parquet, merging with any existing rows."""
    df_new = pd.DataFrame(_run1_buffer)
    if RUN1_RESPONSES_PATH.exists():
        df_old = pd.read_parquet(RUN1_RESPONSES_PATH)
        df = pd.concat([df_old, df_new], ignore_index=True).drop_duplicates(
            subset=["prompt_id", "model_name"]
        )
    else:
        df = df_new
    RUN1_RESPONSES_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(RUN1_RESPONSES_PATH, index=False)
    _run1_buffer.clear()
    return df


# ---------------------------------------------------------------------------
# kbench task definition
# ---------------------------------------------------------------------------

@kbench.task(name="run1_zeroshot")
def run1_zeroshot(llm, id: str, main_prompt: str, correct_answer: str) -> bool:
    """
    Type 1 cognition: zero-shot response + judge binary accuracy.

    Returns True if the model's response is judged correct.
    The response and judgment are also persisted for Run 2 decoy construction.
    """
    # Fresh isolated context — no carry-over between prompts
    with kbench.chats.new(f"run1_{id}"):
        response = llm.prompt(main_prompt)

    # Judge evaluates against the known correct answer
    # For MC questions: check if the response contains the correct option letter
    # The judge also catches paraphrased or equivalent answers for non-MC questions
    judge_criteria = [
        f"The response must select or clearly indicate answer option '{correct_answer}' "
        f"as the final answer. Partial reasoning toward the right answer does not count — "
        f"only the final stated answer matters.",
    ]

    assessment = kbench.assertions.assess_response_with_judge(
        criteria=judge_criteria,
        response_text=response,
        judge_llm=kbench.judge_llm,
    )
    is_correct = all(r.passed for r in assessment.results)

    # Persist for Run 2 decoy construction
    _store_run1_response(
        prompt_id=id,
        model_name=llm.name,
        response=response,
        is_correct=is_correct,
    )

    return is_correct


# ---------------------------------------------------------------------------
# Orchestration helper: run Run 1 across all models and persist results
# ---------------------------------------------------------------------------

def run_run1_all_models(
    eval_df: pd.DataFrame,
    models: list,
    n_jobs: int = 1,
) -> pd.DataFrame:
    """
    Execute Run 1 horizontally across all models.
    Must complete before Run 2 can start (decoy matrix depends on all responses).

    Returns a combined DataFrame with columns: model, id, result (bool).
    """
    all_results = []
    for model in models:
        print(f"\nRun 1: {model.name} ({len(eval_df)} prompts)")
        results = run1_zeroshot.evaluate(
            llm=[model],
            evaluation_data=eval_df,
            n_jobs=n_jobs,
            timeout=120,
        )
        flush_run1_responses()

        result_df = results.as_dataframe()
        result_df["model"] = model.name
        all_results.append(result_df)

        accuracy = result_df["result"].mean()
        print(f"  Accuracy: {accuracy:.3f} ({int(accuracy * len(eval_df))}/{len(eval_df)})")

    combined = pd.concat(all_results, ignore_index=True)
    return combined
