# Benchmark Card

Does a model know what it knows? Standard benchmarks answer a different question: can a model get the right answer? These are not the same thing. A model that is right 80% of the time but cannot distinguish its reliable answers from its guesses is less useful — and more dangerous in high-stakes settings — than one that knows when to be uncertain. When a model is deployed for medical triage, legal research, or scientific reasoning, confident wrongness is a specific failure mode that accuracy metrics cannot detect.

MCBench measures this directly. The target capability is **metacognition**: the ability to monitor the reliability of one's own reasoning. In humans, this is studied via Signal Detection Theory (SDT), which separates *sensitivity* (can you detect when you're right?) from *bias* (do you just always say you're confident?). MCBench applies this framework to language models at benchmark scale, producing a single composite index — the **Metacognitive Capability Index (MCI)** — that captures introspective sensitivity, self-model stability, and calibration quality in one number.

---

## The Core Problem With Calibration Alone

Expected Calibration Error (ECE) is the conventional proxy for confidence quality, but it has a structural blind spot. A model that emits a constant 72% confidence on every question achieves near-zero ECE if its accuracy is near 72% — yet it has zero discriminative metacognitive signal and cannot distinguish its reliable answers from its guesses. MCBench closes this gap by applying **Signal Detection Theory (SDT)** to language models.

---

## The MCI Score

$$\text{MCI} = 0.65 \cdot \max(0,\, M\text{-ratio}) \;+\; 0.35 \cdot \text{SRS} \;-\; \gamma \cdot \text{ECE}$$

Three components, each measuring a distinct facet of metacognition:

### M-ratio (weight 0.65) — introspective sensitivity

M-ratio is the SDT metric $\text{meta-}d' / d'$: how efficiently the model converts its first-order accuracy signal into per-question confidence judgments. It is estimated via the **Type 2 AUROC** — the probability that the model assigns higher confidence to questions it gets right than to questions it gets wrong:

$$\text{AUROC}_2 \in [0.5,\, 1.0] \quad (0.5 = \text{chance},\; 1.0 = \text{perfect})$$

$$M\text{-ratio} = \frac{2\,\Phi^{-1}(\text{AUROC}_2)}{2\,\Phi^{-1}(\text{accuracy})} \quad \text{(Fleming \& Lau, 2014)}$$

- $M\text{-ratio} = 1$: model uses introspective signal as efficiently as its accuracy allows  
- $M\text{-ratio} < 1$: introspective signal is being wasted  
- $M\text{-ratio} > 1$: achievable; reflects strong confidence discrimination  
- Constant-confidence model: $M\text{-ratio} \approx 0$ regardless of accuracy  
- Frontier projection: $1.5$–$2.5$

### SRS (weight 0.35) — self-model stability

The **Self-Recognition Score** asks whether the model can identify its own Run 1 response in a quality-bracketed line-up of three anonymized answers (A / B / C). The two decoys are drawn from one higher-capability and one lower-capability model to bracket the test model in quality. Chance performance is $\frac{1}{3} \approx 0.333$. This probes whether the model has a stable internal self-model of its own output style — a dimension of metacognitive capability captured by neither accuracy nor ECE.

### ECE with dynamic γ — calibration penalty

ECE is computed across 10 equal-width confidence bins. The penalty coefficient $\gamma$ is **dynamic** to prevent constant-confidence gaming:

$$\gamma = 0.30 \cdot \exp\!\left(-\frac{\sigma_{\text{conf}}}{5}\right)$$

where $\sigma_{\text{conf}}$ is the standard deviation of the model's confidence outputs. Near-constant confidence yields $\gamma \to 0.30$ (maximum penalty); high variance yields $\gamma \to 0$. MCI has no fixed ceiling and can go slightly negative for pathological constant-confidence models.

---

## Evaluation Protocol

**Three-run design** (prevents answer-confidence co-optimization):

| Run | What happens | Output |
|-----|-------------|--------|
| **Run 1** | Zero-shot answer in a fresh context (`N_EVAL_ITEMS` questions) | Binary accuracy vector |
| **Run 2** | Model identifies its own Run 1 response among `SRS_DECOY_CANDIDATES + 1` anonymized responses | SRS score |
| **Run 3** | *Second turn in the same chat as Run 1* — five-step structured confidence probe | 0–100 integer per question |

The two-turn separation in Run 3 is the key constraint: the answer is committed before confidence is elicited, ensuring confidence reflects genuine post-hoc introspection rather than pre-planned signaling.

**Tunable parameters** (set in the Setup cell of the notebook):

| Parameter | Current value | Range / notes |
|-----------|--------------|---------------|
| `N_EVAL_ITEMS` | 50 | 1–475; `None` = full public dataset |
| `SRS_DECOY_CANDIDATES` | 4 | Peer models per SRS trial; higher = better quality bracketing |

---

## Dataset

**Public evaluation set** — 475 open-ended MMLU-Pro items across 14 disciplines, filtered against shortcut-susceptible formats and balanced across five cognitive domains (factual, reasoning, epistemic, mathematical, ethical). No multiple-choice option letters; correctness is judged by an LLM evaluator on natural-language responses.

[mcbench-metacognitive-benchmark-mmlu-pro-pub](https://www.kaggle.com/datasets/ivogeorg/mcbench-metacognitive-benchmark-mmlu-pro-pub)

**Private held-out set** — 900 items drawn from both MMLU-Pro and GPQA Diamond (PhD-level biology, chemistry, and physics), retained for post-deadline evaluation and ESMA fine-tuning experiments where licensing permits. Not included in the public dataset to prevent leakage into future model training corpora.

---

## Benchmark Tasks

The notebook exposes **two tasks**; the primary task is selected via `%choose`:

**`mcbench_metacognition`** *(active)* — solo model evaluation. Runs the full three-run protocol and returns MCI directly.

**`mcbench_metamind`** *(available, not selected)* — three-agent MetaMind scaffold. A Theory-of-Mind agent generates epistemic perspectives; a domain synthesis agent produces a consensus answer with an agreement score; the primary model answers with awareness of the panel's uncertainty. Returns MCI for the scaffolded system. The delta $\Delta\text{MCI} = \text{MetaMind} - \text{solo}$ serves as an indirect measure of a model's metacognitive deficit: models with weaker intrinsic metacognitive signal should benefit most from explicit epistemic scaffolding.
