"""
MCBench end-to-end pipeline orchestrator.

Execution order
---------------
1. Run 1  — zero-shot answers + judge scoring for all models (parallel per-model)
2. Decoys — build Run 2 stimulus table from the Run 1 response matrix
3. Run 2  — self-recognition for all models
4. Run 3  — two-turn metacognitive confidence for all models
5. Metrics — compute MCI per model, build leaderboard

Usage (Kaggle / model-proxy mode):
    from src.pipeline import run_mcbench
    leaderboard = run_mcbench()

Usage (with MetaMind included):
    from src.pipeline import run_mcbench, add_metamind
    leaderboard = run_mcbench(include_metamind=True)

Usage (local / single-model mode):
    from src.pipeline import run_mcbench
    leaderboard = run_mcbench(models=[my_model], max_items=50)
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parents[1] / "data"


def _load_eval_set(max_items: int | None = None) -> pd.DataFrame:
    path = DATA_DIR / "eval_set.parquet"
    df = pd.read_parquet(path)
    if max_items is not None:
        # Stratified subsample to keep domain proportions
        df = (
            df.groupby("domain", group_keys=False)
            .apply(lambda g: g.sample(
                min(len(g), max(1, int(max_items * len(g) / len(df)))),
                random_state=42,
            ))
            .reset_index(drop=True)
        )
        df = df.iloc[:max_items]  # exact cap
    return df


def run_mcbench(
    models: list | None = None,
    max_items: int | None = None,
    run1_n_jobs: int = 4,
    run2_n_jobs: int = 2,
    run3_n_jobs: int = 1,
    include_metamind: bool = False,
    metamind_base_model_name: str | None = None,
) -> pd.DataFrame:
    """
    Full MCBench pipeline.

    Parameters
    ----------
    models :                   list of LLMChat instances; if None, loads from Kaggle model proxy
    max_items :                cap the number of eval items (useful for smoke tests)
    run1_n_jobs :              parallelism for Run 1 (IO-bound, safe to parallelize)
    run3_n_jobs :              parallelism for Run 3 (two-turn; keep low to avoid rate limits)
    include_metamind :         add MetaMindAgent to the model list (uses first available model)
    metamind_base_model_name : name of the model to use as MetaMind's base LLM

    Returns
    -------
    Leaderboard DataFrame sorted by MCI descending.
    """
    from src.benchmark.run1_zeroshot import run_run1_all_models
    from src.benchmark.decoys import build_decoys
    from src.benchmark.run2_recognition import run_run2
    from src.benchmark.run3_metacognition import run_run3
    from src.metrics.mci import build_leaderboard, compute_mci

    # ── 0. Models ────────────────────────────────────────────────────────────
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

    # Optionally add MetaMind to the model list
    if include_metamind and models:
        from src.benchmark.agents import MetaMindAgent
        if metamind_base_model_name:
            base = next((m for m in models if m.name == metamind_base_model_name), models[0])
        else:
            base = models[0]  # use the first available model as base
        metamind = MetaMindAgent(base_llm=base, n_hypotheses=3)
        # Insert MetaMind after the base models (not as a regular benchmarked model for Run 2 decoy pool)
        models = list(models) + [metamind]
        print(f"MetaMind added (base: {base.name})")

    print(f"MCBench: {len(models)} model(s), eval_set max_items={max_items or 'all'}")

    # ── 1. Load data ─────────────────────────────────────────────────────────
    eval_df = _load_eval_set(max_items)
    print(f"Eval set: {len(eval_df)} items")

    # ── 2. Run 1 — zero-shot + judge ─────────────────────────────────────────
    run1_results_df = run_run1_all_models(eval_df, models, n_jobs=run1_n_jobs)
    print(f"Run 1 complete. Responses persisted to data/run1_responses.parquet")

    # ── 3. Build decoys ───────────────────────────────────────────────────────
    decoys_df = build_decoys()
    print(f"Decoys built: {len(decoys_df)} rows")

    # ── 4. Run 2 + Run 3 per model ────────────────────────────────────────────
    mci_inputs: dict[str, dict] = {}

    for model in models:
        model_name = model.name

        # Run 2 — self-recognition
        run2_df, srs = run_run2(decoys_df, model)

        # Run 3 — metacognitive confidence
        model_eval_df = eval_df.copy()
        run3_df, confidence_scores = run_run3(
            model_eval_df, model, n_jobs=run3_n_jobs
        )

        # Align confidence with Run 1 accuracy (on common prompt IDs)
        run1_model = run1_results_df[run1_results_df["model"] == model_name]
        merged = eval_df[["id"]].merge(
            run1_model[["id", "result"]].rename(columns={"result": "accuracy"}),
            on="id", how="inner"
        ).merge(
            run3_df[["id", "result"]].rename(columns={"result": "confidence"}),
            on="id", how="inner"
        )

        mci_inputs[model_name] = {
            "accuracy_vector":   merged["accuracy"].astype(int).tolist(),
            "confidence_vector": merged["confidence"].astype(float).tolist(),
            "srs":               srs,
        }

    # ── 5. Compute MCI and build leaderboard ─────────────────────────────────
    results_by_model = {}
    for model_name, inputs in mci_inputs.items():
        with warnings.catch_warnings(record=True):
            results_by_model[model_name] = compute_mci(
                inputs["accuracy_vector"],
                inputs["confidence_vector"],
                inputs["srs"],
            )

    leaderboard = build_leaderboard(results_by_model)

    print("\n=== MCBench Leaderboard ===")
    print(leaderboard[["model", "mci", "m_ratio", "d_prime", "srs", "ece",
                        "reliable"]].to_string())
    return leaderboard
