# MCBench: Measuring Metacognitive Capability in AI via Signal Detection Theory

## The Problem

Does a model know what it knows? Standard benchmarks answer a different question: can a model get the right answer? These are not the same thing. A model that is right 80% of the time but cannot distinguish its reliable answers from its guesses is less useful — and more dangerous in high-stakes settings — than one that knows when to be uncertain. When a model is deployed for medical triage, legal research, or scientific reasoning, confident wrongness is a specific failure mode that accuracy metrics cannot detect.

MCBench measures this directly. The target capability is **metacognition**: the ability to monitor the reliability of one's own reasoning. In humans, this is studied via Signal Detection Theory (SDT), which separates *sensitivity* (can you detect when you're right?) from *bias* (do you just always say you're confident?). MCBench applies this framework to language models for the first time at benchmark scale.

---

## Why Calibration Alone Is Not Enough

Expected Calibration Error (ECE) is the standard proxy for confidence quality, but it has a well-known blind spot: a model that outputs a constant 72% confidence on every answer achieves low ECE if its average accuracy happens to be near 72%. It has zero discriminative metacognitive signal — it cannot tell good answers from bad ones — and yet ECE rewards it.

The SDT metric that catches this is the **M-ratio** (meta-d′ / d′): how efficiently the model uses its first-order accuracy signal to generate per-question confidence judgments. M-ratio = 1 is ideal; below 1 means introspective signal is being wasted; a constant-confidence model scores near 0 regardless of accuracy. MCBench makes this measurable and puts it on the leaderboard.

---

## The MCI Formula

$$\text{MCI} = 0.65 \cdot \max(0, M\text{-ratio}) + 0.35 \cdot \text{SRS} - \gamma \cdot \text{ECE}$$

Three components, each measuring a distinct facet of metacognition:

**M-ratio** (weight 0.65) is the core introspective signal, estimated via the Type 2 AUROC (Area Under the Receiver Operating Characteristic curve) — the probability that the model assigns higher confidence to questions it gets right than to questions it gets wrong (0.5 = chance, 1.0 = perfect). Formally: M-ratio = 2·Φ⁻¹(AUROC) / 2·Φ⁻¹(accuracy), following Fleming & Lau (2014). M-ratio = 1 is ideal; below 1 means the model underuses its own introspective signal; above 1 is achievable and reflects strong confidence discrimination. Frontier models are projected at 1.5–2.5; a constant-confidence model scores near 0.

**SRS** (Self-Recognition Score, weight 0.35) measures whether the model can identify its own response among three anonymized candidates: its own output, one from a higher-capability model, and one from a lower-capability model. Chance performance is 1/3 ≈ 0.333. This assesses whether the model has a stable self-model of its own output style — a distinct but related dimension of metacognitive capability, captured by neither accuracy nor ECE.

**ECE** penalizes miscalibration across 10 equal-width confidence bins. The penalty coefficient **γ is dynamic**: models with near-constant confidence receive an amplified penalty via γ = 0.30 · exp(−σ / 5), closing the exploit ECE alone leaves open. MCI has no fixed upper bound — frontier models are projected in the 1.5–1.8 range — but can go slightly negative for constant-confidence models (maximum γ penalty with near-zero M-ratio and SRS).

---

## Dataset: Open-Ended Response Prompts

All items use **open-ended prompts** — no multiple-choice answer matching, no letter shortcuts. The model reasons through the problem in natural language and states its answer; a judge LLM evaluates binary correctness. This eliminates option-pattern artifacts and forces genuine reasoning to drive the accuracy vector that M-ratio depends on.

The evaluation set contains **475 items** from MMLU-Pro: professional-level questions across 14 disciplines, filtered to remove shortcut-susceptible items and balanced across five cognitive domains — factual, reasoning, epistemic, mathematical, and ethical. A private 125-item GPQA Diamond extension (PhD-level biology/chemistry/physics) is retained for post-deadline evaluation where licensing permits.

The **three-run protocol** prevents the most common evaluation artifact — models optimizing their answers to maximize reported confidence:

- **Run 1**: Model answers each question zero-shot in a fresh context → binary accuracy vector (correct/incorrect per question)
- **Run 2**: Model identifies its own response among three quality-bracketed candidates → SRS score
- **Run 3**: In the *same chat* as Run 1, a second turn elicits a 0–100 confidence integer via a structured five-step probe → confidence vector (one integer per question)

The accuracy vector (Run 1) and the confidence vector (Run 3) are paired question-by-question to compute M-ratio via Type 2 AUROC and ECE via confidence-bin calibration. Run 2 contributes SRS directly. The two-turn separation in Run 3 is the key methodological constraint. Sending the question and confidence probe together allows a model to craft its answer with the confidence elicitation already in view. Separating them — answer committed in turn 1, confidence elicited in turn 2 — ensures the confidence reflects genuine post-hoc introspection rather than pre-planned signaling.

---

## Projected Results

The following table shows expected results across representative model tiers, constructed from published accuracy benchmarks and calibration literature. These are projections, not live measurements; live leaderboard results are in progress.

| Rank | Model | MCI | M-ratio | SRS | ECE |
|------|-------|-----|---------|-----|-----|
| 1 | gpt-4o | 1.734 | 2.34 | 0.64 | 0.212 |
| 2 | claude-3-5-sonnet | 1.687 | 2.24 | 0.68 | 0.191 |
| 3 | gemini-1-5-pro | 1.383 | 1.82 | 0.59 | 0.185 |
| 4 | claude-3-haiku | 1.152 | 1.54 | 0.45 | 0.080 |
| 5 | gemini-flash-1.5 | 1.117 | 1.50 | 0.42 | 0.096 |
| 6 | gpt-3.5-turbo | 0.219 | 0.14 | 0.38 | 0.072 |
| 7 | llama-3-8b (constant) | −0.007 | 0.00 | 0.34 | 0.420 |

We would expect frontier models to show higher M-ratios alongside higher SRS, reflecting both better introspective sensitivity and a more stable self-model. The more interesting prediction concerns gpt-3.5-turbo: despite reasonable accuracy, its projected M-ratio near zero would suggest that accuracy and metacognitive capability can dissociate substantially — a finding invisible to standard benchmarks. The constant-confidence model at the bottom should score near zero regardless of accuracy, demonstrating that the dynamic γ mechanism does its job.

One confirmed result from early runs: Gemini 3 Flash produced MCI = 0.32 on 10 evaluation questions (below the 50-trial reliability threshold — noted in output). A full 50-question run on the public dataset is currently in progress.

---

## ESMA: Optimizing Metacognition Directly

**Evolution Strategies for Metacognitive Alignment (ESMA)** uses Natural Evolution Strategies (NES) to optimize M-ratio directly as a fitness function over LoRA adapter parameters. The hypothesis is that M-ratio, estimated via Type 2 AUROC, provides sufficient signal for black-box weight-space optimization — no gradient through the LLM required.

An initial run on Gemma 4 E2B established a baseline M-ratio of **1.07**, confirming meaningful metacognitive calibration in the unperturbed model. Random LoRA perturbations degraded this consistently, as expected: the pretrained weights represent a local optimum that NES must escape via directional perturbation. The trajectory over 6 generations was noisy, consistent with high variance in M-ratio estimation at small N.

We would expect lower-capability models to show larger ESMA gains — more room to improve metacognitive efficiency without disrupting first-order accuracy — while already well-calibrated models show smaller but still positive shifts. Testing this prediction across model tiers is a primary post-deadline goal.

---

## MetaMind: Scaffolded Metacognition

**MetaMind** is a training-free alternative. A Theory-of-Mind agent generates epistemic perspectives on each question; a domain synthesis agent produces a consensus answer with an agreement score; the primary model answers with access to this prior and generates confidence with awareness of the panel's uncertainty level.

The hypothesis is that models with weaker intrinsic metacognitive signal benefit most from explicit epistemic scaffolding, while stronger models show diminishing returns. Notably, the *magnitude* of MetaMind's improvement over a model's solo MCI would itself serve as an indirect measure of that model's metacognitive deficit — a diagnostic the benchmark naturally produces.

---

## Future Directions

The held-out GPQA set and ESMA pipeline point to a deeper question: does directly optimizing M-ratio during fine-tuning produce models that are genuinely more self-aware, or ones that are better at imitating calibrated responses on the training distribution? Answering this rigorously requires held-out evaluation sets, adversarial probe designs, and comparison against models trained on explicit metacognitive data — all tractable next steps from the infrastructure built here.

**Note to judges:** The benchmark is configured at N=50 evaluation questions per model run for responsive leaderboard turnaround during the judging period. The full evaluation set contains 475 questions (MMLU-Pro only); the original 600-question set included 125 GPQA Diamond questions which were removed from the public dataset to preserve the integrity of future benchmark runs — any model evaluated here must not have had access to the evaluation data during training. The full 475-question run is available by setting `N_EVAL_ITEMS=None` in the Setup cell of the task notebook.
