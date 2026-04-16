"""
MCBench end-to-end pipeline orchestrator.

Execution order
---------------
1. Run 1  — zero-shot answers + judge scoring for all solo models
2. Decoys — build Run 2 stimulus table from the Run 1 response matrix
3. MetaMind selection (if requested) — pick base model from Run 1 accuracy
4. Run 2 + Run 3 — self-recognition and metacognitive confidence for all models
                   (solo models + MetaMind if included)
5. Metrics — compute MCI per model, build leaderboard

MetaMind base-model selection
------------------------------
MetaMind is added AFTER Run 1 so the base model can be chosen from actual
accuracy data rather than from an arbitrary ordering.  The default strategy
is "mid": the model with median accuracy, which has the most room to show
the multi-agent metacognitive advantage without ceiling or floor effects.

Available strategies:
  "mid"   — median-accuracy solo model (default, maximises contrast)
  "best"  — highest-accuracy solo model (tests ceiling improvement)
  "worst" — lowest-accuracy solo model (stress test)
  <name>  — exact model name string

Usage (Kaggle / model-proxy mode):
    from src.pipeline import run_mcbench
    leaderboard = run_mcbench(include_metamind=True)   # strategy="mid" default

Usage (explicit model):
    leaderboard = run_mcbench(include_metamind=True, metamind_strategy="gemini-1.5-flash")

Usage (local / single-model mode):
    leaderboard = run_mcbench(models=[my_model], max_items=50)
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parents[1] / "data"


def _load_eval_set(max_items: int | None = None) -> pd.DataFrame:
    path = DATA_DIR / "eval_set.parquet"
    df = pd.read_parquet(path)
    if max_items is not None:
        df = (
            df.groupby("domain", group_keys=False)
            .apply(lambda g: g.sample(
                min(len(g), max(1, int(max_items * len(g) / len(df)))),
                random_state=42,
            ))
            .reset_index(drop=True)
        )
        df = df.iloc[:max_items]
    return df


def _select_metamind_base(
    solo_models: list,
    run1_results_df: pd.DataFrame,
    strategy: str,
) -> object:
    """
    Select the base model for MetaMind from Run 1 accuracy results.

    strategy : "mid" | "best" | "worst" | <exact model name>
    """
    # Compute per-model accuracy from Run 1
    acc_by_model = (
        run1_results_df.groupby("model")["result"]
        .mean()
        .sort_values()
    )

    if strategy in ("mid", "median"):
        # Median-accuracy model — most room to show MetaMind advantage
        median_idx = len(acc_by_model) // 2
        chosen_name = acc_by_model.index[median_idx]
        reason = "median accuracy"
    elif strategy == "best":
        chosen_name = acc_by_model.index[-1]
        reason = "highest accuracy"
    elif strategy == "worst":
        chosen_name = acc_by_model.index[0]
        reason = "lowest accuracy"
    else:
        # Treat as an explicit model name
        chosen_name = strategy
        reason = f"explicit selection"

    base = next((m for m in solo_models if m.name == chosen_name), None)
    if base is None:
        fallback = solo_models[len(solo_models) // 2]
        print(f"  MetaMind: model '{chosen_name}' not found; falling back to {fallback.name}")
        return fallback

    chosen_acc = acc_by_model.get(chosen_name, float("nan"))
    print(
        f"MetaMind base model: {chosen_name} "
        f"(strategy={strategy}, {reason}, Run 1 accuracy={chosen_acc:.3f})"
    )
    print(
        f"  Accuracy ranking: " +
        ", ".join(f"{n}={v:.2f}" for n, v in acc_by_model.items())
    )
    return base


def run_mcbench(
    models: list | None = None,
    max_items: int | None = None,
    run1_n_jobs: int = 4,
    run2_n_jobs: int = 2,
    run3_n_jobs: int = 1,
    include_metamind: bool = False,
    metamind_strategy: str = "mid",
    metamind_n_hypotheses: int = 3,
) -> pd.DataFrame:
    """
    Full MCBench pipeline.

    Parameters
    ----------
    models :               list of LLMChat instances; if None, loads from Kaggle model proxy
    max_items :            cap the number of eval items (useful for smoke tests)
    run1_n_jobs :          parallelism for Run 1 (IO-bound, safe to parallelize)
    run3_n_jobs :          parallelism for Run 3 (two-turn; keep low to avoid rate limits)
    include_metamind :     add MetaMindAgent after Run 1 completes
    metamind_strategy :    how to pick MetaMind's base model from Run 1 results:
                           "mid" (default) | "best" | "worst" | <model name string>
                           "mid" is recommended: maximises the contrast between
                           solo and multi-agent metacognition without ceiling effects.
    metamind_n_hypotheses: number of ToM hypotheses per question (default 3)

    Returns
    -------
    Leaderboard DataFrame sorted by MCI descending.
    """
    from src.benchmark.run1_zeroshot import run_run1_all_models
    from src.benchmark.decoys import build_decoys
    from src.benchmark.run2_recognition import run_run2
    from src.benchmark.run3_metacognition import run_run3
    from src.metrics.mci import build_leaderboard, compute_mci

    # ── 0. Load solo models ──────────────────────────────────────────────────
    if models is None:
        try:
            from kaggle_benchmarks.kaggle import load_available_models
            models_dict = load_available_models()
            if not models_dict:
                raise RuntimeError("LLMS_AVAILABLE is empty")
            models = list(models_dict.values())
        except Exception as e:
            raise RuntimeError(
                f"Could not load models from Kaggle proxy: {e}. "
                "Pass `models=[...]` explicitly for local runs."
            ) from e

    solo_models = list(models)
    print(f"MCBench: {len(solo_models)} solo model(s), eval_set max_items={max_items or 'all'}")
    if include_metamind:
        print(f"MetaMind: enabled (strategy={metamind_strategy}, n_hypotheses={metamind_n_hypotheses})")

    # ── 1. Load data ─────────────────────────────────────────────────────────
    eval_df = _load_eval_set(max_items)
    print(f"Eval set: {len(eval_df)} items")

    # ── 2. Run 1 — zero-shot + judge (solo models only) ───────────────────────
    run1_results_df = run_run1_all_models(eval_df, solo_models, n_jobs=run1_n_jobs)
    print(f"Run 1 complete. Responses persisted to data/run1_responses.parquet")

    # ── 3. Select MetaMind base model from Run 1 accuracy ────────────────────
    all_models = list(solo_models)
    metamind_base_name: str | None = None

    if include_metamind:
        from src.benchmark.agents import MetaMindAgent
        base = _select_metamind_base(solo_models, run1_results_df, metamind_strategy)
        metamind_base_name = base.name
        metamind = MetaMindAgent(base_llm=base, n_hypotheses=metamind_n_hypotheses)
        all_models = solo_models + [metamind]

        # Run 1 for MetaMind — its three-agent pipeline produces different responses
        # from the base model alone, so we benchmark it independently
        run1_meta_df = run_run1_all_models(eval_df, [metamind], n_jobs=1)
        run1_results_df = pd.concat([run1_results_df, run1_meta_df], ignore_index=True)
        print(f"MetaMind Run 1 complete.")

    # ── 4. Build decoys ───────────────────────────────────────────────────────
    decoys_df = build_decoys()
    print(f"Decoys built: {len(decoys_df)} rows")

    # ── 5. Run 2 + Run 3 per model ────────────────────────────────────────────
    mci_inputs: dict[str, dict] = {}

    for model in all_models:
        model_name = model.name

        # Run 2 — self-recognition
        run2_df, srs = run_run2(decoys_df, model)

        # Run 3 — metacognitive confidence (two-turn)
        run3_df, _ = run_run3(eval_df.copy(), model, n_jobs=run3_n_jobs)

        # Align Run 3 confidence with Run 1 accuracy on matched prompt IDs
        run1_model = run1_results_df[run1_results_df["model"] == model_name]
        merged = (
            eval_df[["id"]]
            .merge(run1_model[["id", "result"]].rename(columns={"result": "accuracy"}),
                   on="id", how="inner")
            .merge(run3_df[["id", "result"]].rename(columns={"result": "confidence"}),
                   on="id", how="inner")
        )

        mci_inputs[model_name] = {
            "accuracy_vector":   merged["accuracy"].astype(int).tolist(),
            "confidence_vector": merged["confidence"].astype(float).tolist(),
            "srs":               srs,
        }

    # ── 6. Compute MCI and build leaderboard ─────────────────────────────────
    results_by_model = {}
    for model_name, inputs in mci_inputs.items():
        with warnings.catch_warnings(record=True):
            results_by_model[model_name] = compute_mci(
                inputs["accuracy_vector"],
                inputs["confidence_vector"],
                inputs["srs"],
            )
        # Tag MetaMind row with its base model for readability
        if include_metamind and model_name == "MetaMind-3Agent" and metamind_base_name:
            results_by_model[model_name]["base_model"] = metamind_base_name

    leaderboard = build_leaderboard(results_by_model)

    print("\n=== MCBench Leaderboard ===")
    cols = ["model", "mci", "m_ratio", "d_prime", "srs", "ece", "reliable"]
    print(leaderboard[cols].to_string())

    if include_metamind and metamind_base_name:
        solo_row = leaderboard[leaderboard["model"] == metamind_base_name]
        meta_row = leaderboard[leaderboard["model"] == "MetaMind-3Agent"]
        if not solo_row.empty and not meta_row.empty:
            delta = meta_row.iloc[0]["mci"] - solo_row.iloc[0]["mci"]
            print(f"\nMetaMind vs {metamind_base_name}: ΔMCI = {delta:+.4f}")

    return leaderboard
