"""
Story 2.2 — Decoy construction between Run 1 and Run 2.

For each (target_model, prompt_id) pair, build a 3-item decoy set from Run 1 responses:
  - self:     target model's own Run 1 response
  - frontier: response from the model with highest overall Run 1 accuracy
  - inferior: response from the model with lowest overall Run 1 accuracy

The presentation order is randomised per trial (self_position stored for scoring).

Output: data/run2_decoys.parquet
Schema: { id, model_name, self_response, frontier_response, inferior_response,
          self_position (0|1|2), option_a, option_b, option_c }
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parents[2] / "data"
RUN1_PATH = DATA_DIR / "run1_responses.parquet"
DECOYS_PATH = DATA_DIR / "run2_decoys.parquet"


def _stable_position(prompt_id: str, model_name: str) -> int:
    """Deterministic shuffle position in [0,1,2] based on prompt+model hash."""
    h = int(hashlib.md5(f"{prompt_id}:{model_name}".encode()).hexdigest(), 16)
    return h % 3


def build_decoys() -> pd.DataFrame:
    """
    Build the decoy DataFrame from Run 1 responses.
    Requires all models to have completed Run 1.
    """
    if not RUN1_PATH.exists():
        raise FileNotFoundError(
            f"Run 1 responses not found at {RUN1_PATH}. "
            "Complete run_run1_all_models() first."
        )

    run1 = pd.read_parquet(RUN1_PATH)
    models = run1["model_name"].unique().tolist()

    if len(models) < 3:
        raise ValueError(
            f"Need at least 3 models for decoy construction; found {len(models)}: {models}. "
            "Run Run 1 on more models first."
        )

    # Compute per-model overall accuracy to identify frontier and inferior
    model_accuracy = (
        run1.groupby("model_name")["is_correct"]
        .mean()
        .sort_values(ascending=False)
    )
    print("Model accuracy ranking:")
    print(model_accuracy.to_string())

    frontier_model = model_accuracy.index[0]   # highest accuracy
    inferior_model = model_accuracy.index[-1]  # lowest accuracy
    print(f"\nFrontier decoy model: {frontier_model} ({model_accuracy[frontier_model]:.3f})")
    print(f"Inferior decoy model: {inferior_model} ({model_accuracy[inferior_model]:.3f})")

    # Pivot to wide format: columns = model_name, rows = prompt_id
    pivot = run1.pivot(index="prompt_id", columns="model_name", values="response")

    rows = []
    for model_name in models:
        if model_name == frontier_model and model_name == inferior_model:
            # Edge case: only two models, skip (shouldn't happen with >= 3 check above)
            continue

        # Determine which model serves as frontier/inferior for this target
        # If target IS the frontier model, use 2nd-best as frontier decoy
        if model_name == frontier_model:
            frontier_decoy_model = model_accuracy.index[1]
        else:
            frontier_decoy_model = frontier_model

        # If target IS the inferior model, use 2nd-worst as inferior decoy
        if model_name == inferior_model:
            inferior_decoy_model = model_accuracy.index[-2]
        else:
            inferior_decoy_model = inferior_model

        for prompt_id, row in pivot.iterrows():
            self_resp = row.get(model_name)
            frontier_resp = row.get(frontier_decoy_model)
            inferior_resp = row.get(inferior_decoy_model)

            # Skip if any response is missing
            if pd.isna(self_resp) or pd.isna(frontier_resp) or pd.isna(inferior_resp):
                continue

            # Randomise presentation order deterministically
            self_pos = _stable_position(str(prompt_id), model_name)
            options = ["", "", ""]
            options[self_pos] = self_resp
            # Place frontier and inferior in remaining slots
            remaining = [i for i in range(3) if i != self_pos]
            options[remaining[0]] = frontier_resp
            options[remaining[1]] = inferior_resp

            rows.append({
                "id": str(prompt_id),
                "model_name": model_name,
                "self_response": self_resp,
                "frontier_response": frontier_resp,
                "inferior_response": inferior_resp,
                "self_position": self_pos,   # 0=A, 1=B, 2=C
                "option_a": options[0],
                "option_b": options[1],
                "option_c": options[2],
            })

    decoys = pd.DataFrame(rows)
    decoys.to_parquet(DECOYS_PATH, index=False)
    print(f"\nDecoy table: {len(decoys):,} rows → {DECOYS_PATH}")
    return decoys
