"""
Story 2.4 — Run 3: Two-turn metacognitive sensitivity.

For each prompt in the eval set, in a fresh context:
  Turn 1: model answers main_prompt  (commits to answer)
  Turn 2: model answers meta_question (evaluates itself, outputs confidence 0-100)

The two-turn separation is mandatory: if both prompts are sent together,
the model optimises the primary answer to maximise the confidence score.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import kaggle_benchmarks as kbench
import pandas as pd

DATA_DIR = Path(__file__).parents[2] / "data"


@dataclass
class MetaCognitionOutput:
    """Structured output enforced via kbench schema for Run 3 Turn 2."""
    final_answer: str    # model's committed answer (for audit)
    confidence_score: int  # 0–100 integer confidence


@kbench.task(name="run3_metacognition")
def run3_metacognition(
    llm,
    id: str,
    main_prompt: str,
    meta_question: str,
) -> float:
    """
    Two-turn metacognitive elicitation.

    Returns the model's self-reported confidence score (0.0–100.0).
    Aligned post-hoc with Run 1 binary accuracy to compute M-ratio.
    """
    with kbench.chats.new(f"run3_{id}_{llm.name}"):
        # Turn 1 — model commits to its answer; meta_question NOT yet visible
        kbench.user.send(main_prompt)
        llm.respond()  # model generates response without seeing meta_question

        # Turn 2 — metacognitive self-evaluation
        kbench.user.send(meta_question)
        meta_output: MetaCognitionOutput = llm.prompt(
            schema=MetaCognitionOutput
        )

    # Clamp to valid range in case the model returns out-of-bounds
    score = max(0, min(100, int(meta_output.confidence_score)))
    return float(score)


def run_run3(eval_df: pd.DataFrame, model, n_jobs: int = 1) -> tuple[pd.DataFrame, list[float]]:
    """Run Run 3 for a single model and return (results_df, confidence_scores)."""
    print(f"\nRun 3: {model.name} ({len(eval_df)} prompts)")
    results = run3_metacognition.evaluate(
        llm=[model],
        evaluation_data=eval_df,
        n_jobs=n_jobs,
        timeout=180,   # longer timeout: two turns + structured output
    )
    results_df = results.as_dataframe()
    confidence_scores = results_df["result"].tolist()

    import numpy as np
    scores = np.array(confidence_scores)
    print(f"  Confidence — mean: {scores.mean():.1f}, std: {scores.std():.1f}, "
          f"min: {scores.min():.0f}, max: {scores.max():.0f}")
    return results_df, confidence_scores
