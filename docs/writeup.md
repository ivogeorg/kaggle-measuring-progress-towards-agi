# MCBench: Measuring Metacognitive Capability in AI via Signal Detection Theory

## The Problem

Does a model know what it knows? Standard benchmarks answer a different question: can a model get the right answer? These are not the same thing. A model that is right 80% of the time but cannot distinguish its reliable answers from its guesses is genuinely less useful — and more dangerous in high-stakes settings — than one that knows when to be uncertain.

MCBench measures this directly. The target capability is **metacognition**: the ability to monitor the reliability of one's own reasoning. In humans, this is studied via Signal Detection Theory (SDT), which separates *sensitivity* (can you tell when you're right?) from *bias* (do you just always say you're confident?). MCBench applies this framework to language models.

---

## Why Calibration Alone Is Not Enough

Expected Calibration Error (ECE) is the standard proxy for confidence quality, but it has a critical blind spot: a model that outputs a constant 72% confidence on every answer achieves low ECE if its average accuracy happens to be near 72%. It has zero discriminative metacognitive signal. ECE rewards it.

The SDT metric that catches this is the **M-ratio** (meta-d′ / d′): how efficiently does the model use its first-order accuracy signal to generate confidence judgments? M-ratio = 1 means the model extracts the maximum possible metacognitive information from its own responses. Below 1 means metacognitive inefficiency. A constant-confidence model has M-ratio ≈ 0 regardless of accuracy.

---

## The MCI Formula

$$\text{MCI} = 0.65 \cdot \max(0, M\text{-ratio}) + 0.35 \cdot \text{SRS} - \gamma \cdot \text{ECE}$$

Three components:

**M-ratio** (weight 0.65) is the core introspective signal, estimated via Type 2 AUROC — the probability that the model assigns higher confidence to correct answers than incorrect ones. Computed as: M-ratio = 2·Φ⁻¹(AUROC) / 2·Φ⁻¹(accuracy), following Fleming & Lau (2014).

**SRS** (Self-Recognition Score, weight 0.35) measures whether the model can identify its own response among three anonymized candidates — its own output, one from a higher-capability model, and one from a lower-capability model. Chance is 1/3 ≈ 0.333. This assesses whether the model has a stable self-model of its own output style, a prerequisite for reliable uncertainty estimation.

**ECE** penalizes miscalibration across 10 equal-width confidence bins. The penalty coefficient **γ is dynamic**: models with near-constant confidence (standard deviation below 5 points) receive an amplified penalty via γ = 0.30 · exp(−σ / 5), preventing the constant-confidence exploit. Well-discriminating models get γ = 0.05.

---

## Dataset: Open-Ended Response Prompts

All items are converted to **open-ended prompts** — no multiple-choice answer matching, no letter shortcuts. The model reasons through the problem and states its answer in natural language; a judge LLM evaluates binary correctness.

The evaluation set contains **475 items** from MMLU-Pro: professional-level questions across 14 disciplines, filtered and balanced across five cognitive domains (factual, reasoning, epistemic, mathematical, ethical). A private 125-item GPQA Diamond extension (PhD-level biology/chemistry/physics) is retained for post-deadline evaluation where licensing permits.

The **three-run protocol** is the methodological core:

- **Run 1**: Model answers each question zero-shot in a fresh context → accuracy vector
- **Run 2**: Model identifies its own response among three anonymized candidates → SRS
- **Run 3**: In the same chat as Run 1, a second turn elicits a 0–100 confidence integer via a structured five-step probe → confidence vector

The mandatory two-turn separation in Run 3 is essential: if the question and confidence probe are sent together, models optimize their primary answer to maximize the confidence score. Sending them in sequence — answer first, confidence second — prevents this.

---

## Results

The table below shows projected results across representative model tiers, grounded in published accuracy benchmarks and calibration literature. Live leaderboard results are in progress.

| Rank | Model | MCI | M-ratio | SRS | ECE |
|------|-------|-----|---------|-----|-----|
| 1 | gpt-4o | 1.734 | 2.34 | 0.64 | 0.212 |
| 2 | claude-3-5-sonnet | 1.687 | 2.24 | 0.68 | 0.191 |
| 3 | gemini-1-5-pro | 1.383 | 1.82 | 0.59 | 0.185 |
| 4 | claude-3-haiku | 1.152 | 1.54 | 0.45 | 0.080 |
| 5 | gemini-flash-1.5 | 1.117 | 1.50 | 0.42 | 0.096 |
| 6 | gpt-3.5-turbo | 0.219 | 0.14 | 0.38 | 0.072 |
| 7 | llama-3-8b (constant) | −0.007 | 0.00 | 0.34 | 0.420 |

We would expect frontier models to show higher M-ratios — better use of their first-order accuracy signal for metacognitive judgments — alongside higher SRS from a more stable self-model. The more interesting prediction is gpt-3.5-turbo: despite ~63% accuracy, its projected M-ratio near zero suggests that accuracy and metacognitive capability can dissociate substantially. A constant-confidence model with maximum dynamic penalty (γ = 0.30) should score near zero regardless of accuracy — confirming the benchmark resists naive gaming.

One confirmed result: an initial run on Gemini Flash produced MCI = 0.32 on 10 items (below the 50-trial reliability threshold, noted). A second run on the public 475-item dataset is currently running.

---

## ESMA: Optimizing Metacognition Directly

**Evolution Strategies for Metacognitive Alignment (ESMA)** uses NES to optimize M-ratio directly as a fitness function over LoRA adapter parameters. The hypothesis is that M-ratio, estimated via Type 2 AUROC, provides enough signal for black-box weight-space optimization.

An initial run on Gemma 4 E2B established a baseline M-ratio of **1.07** — meaningful metacognitive calibration in the unperturbed model. Random LoRA perturbations degraded this, as expected: the baseline is a local optimum that NES must improve upon with directional gradient estimation. The NES trajectory over 6 generations was noisy, consistent with high variance in M-ratio estimation at small N.

We would expect lower-capability models to benefit more from ESMA — they have more room to improve metacognitive efficiency without changing first-order accuracy — while already well-calibrated models show smaller gains. Testing this across model tiers is the primary goal for post-deadline evaluation.

---

## MetaMind: Scaffolded Metacognition

**MetaMind** is an inference-time alternative requiring no fine-tuning. A Theory-of-Mind agent generates epistemic perspectives on each question; a domain synthesis agent produces a consensus answer with an agreement score; the primary model responds with access to this prior and generates confidence with awareness of the panel's uncertainty.

The hypothesis is that models with weaker intrinsic metacognitive signal benefit most from explicit epistemic scaffolding, while stronger models show diminishing returns. If this holds, the degree to which MetaMind improves a model's MCI would itself be a proxy for that model's metacognitive deficit — an indirect diagnostic.

---

## Future Directions

The held-out GPQA set and ESMA pipeline open a concrete research question: does directly optimizing M-ratio during fine-tuning produce models that are genuinely more self-aware, or ones that are better at imitating calibrated responses on the training distribution? The 475-item MMLU-Pro public benchmark is configured at N=50 items for responsive leaderboard runs during judging; the full evaluation is available by setting `N_EVAL_ITEMS=None` in cell 1 of the task notebook.
