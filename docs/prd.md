# PRD: Metacognitive Capability Benchmark for LLMs (MCBench)

**Kaggle Hackathon**: Measuring Progress Toward AGI – Cognitive Abilities  
**Submission URL**: https://www.kaggle.com/competitions/kaggle-measuring-agi  
**Author**: Ivo Georgiev  
**Date**: 2026-04-15  
**Status**: Active

---

## 1. Problem Statement

Current AI benchmarks measure *what* models know (Type 1 cognition) but not *whether they know what they know* (Type 2 metacognition). Aggregate calibration metrics like Expected Calibration Error (ECE) are structurally incapable of measuring this distinction — a model that outputs a static 80% confidence on every query scores ECE=0 (perfect calibration) while having zero discriminative metacognitive sensitivity.

This benchmark fills that gap by applying Type 2 Signal Detection Theory — specifically the M-ratio (meta-d′/d′) framework established in cognitive neuroscience — to large language models. It measures metacognitive efficiency in a way that is mathematically independent of raw factual performance, enabling meaningful comparison across model scales, architectures, and training paradigms.

---

## 2. Goals and Success Criteria

### Primary (Hackathon Submission)
- [ ] A live public Kaggle benchmark measuring M-ratio, SRS, and MCI for any submitted model
- [ ] A populated leaderboard demonstrating that larger/frontier models score significantly higher MCI than smaller models
- [ ] A MetaMind multi-agent run demonstrating that externalized metacognition (agent consensus) achieves higher M-ratio than equivalent isolated models
- [ ] A 1,500-word public writeup with thumbnail, narrative, and submission URL

### Secondary (Long-term project viability)
- [ ] ESMA fine-tuning pipeline for Gemma 4 E2B demonstrating measurable M-ratio improvement across LoRA checkpoints (Gemma 4 Good Hackathon entry)
- [ ] Held-out dataset permanently locked for longitudinal tracking

### Success Metrics
- Leaderboard shows monotonic M-ratio increase correlated with model scale across at least 5 models spanning ≥3 orders of magnitude in parameter count
- MetaMind M-ratio exceeds the best single-model M-ratio in the benchmark
- Benchmark receives public engagement (views, forks, runs) sufficient for Kaggle hackathon evaluation

---

## 3. Stakeholders

| Role | Person/Entity | Interest |
|------|--------------|----------|
| Benchmark author | Ivo Georgiev | Hackathon submission, long-term project |
| Kaggle evaluators | Kaggle team | Benchmark quality, SDK usage correctness |
| Gemma 4 Good Hackathon | Google/Kaggle | ESMA as Gemma 4 use case |
| AI research community | Public | Longitudinal metacognition tracking |

---

## 4. Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    DATASET LAYER                            │
│  SAD (13k items) + Confidence Database → NeMo Curator       │
│  → filter model-identity-specific items                     │
│  → 500-750 eval set  +  1000 held-out set                   │
│  Each item: { main_prompt, meta_question (fixed) }          │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│                  THREE-RUN PIPELINE (kbench)                 │
│                                                             │
│  RUN 1 (all models, horizontal)                             │
│    → zero-shot response + judge binary accuracy             │
│    → builds response matrix for decoy construction          │
│                                                             │
│  RUN 2 (per model, uses Run 1 matrix)                       │
│    → present: own response + frontier decoy + inferior decoy│
│    → Self-Recognition Score (SRS)                           │
│                                                             │
│  RUN 3 (per model, fresh context)                           │
│    → two-turn: main_prompt → answer → meta_question         │
│    → structured confidence score (0-100)                    │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│                  METRIC COMPUTATION                          │
│                                                             │
│  d′ and meta-d′ via HMeta-d (Python: metadpy)              │
│  M-ratio = meta-d′ / d′                                     │
│  SRS = % correct self-identifications from Run 2            │
│  ECE = bin-averaged |confidence − accuracy|                 │
│  MCI = α·max(0, M-ratio) + β·SRS − γ·ECE                  │
│        α=0.65, β=0.35, γ=dynamic (see §7)                  │
└────────────────────┬────────────────────────────────────────┘
                     │
        ┌────────────┴────────────┐
        ▼                         ▼
┌───────────────┐       ┌────────────────────┐
│  LEADERBOARD  │       │  ESMA PIPELINE     │
│  %choose      │       │  (held-out only)   │
│  main_task    │       │  Gemma 4 E2B       │
│               │       │  LoRA + M-ratio    │
└───────────────┘       │  per epoch         │
                        └────────────────────┘
```

---

## 5. Feature Stories (Phased)

### Phase 1 — Dataset (prerequisite for everything)

**Story 1.1 — Source data acquisition**
- Download SAD (Situational Awareness Dataset, ~13k items) from HuggingFace
- Download Confidence Database behavioral trials
- Output: two raw DataFrames, saved to `data/raw/`

**Story 1.2 — Model-identity filter**
- Identify and remove SAD items that are model-identity-specific: questions about training cutoff, creator identity, capability flags that differ across models
- Criteria: any item whose ground-truth answer changes depending on which model is being evaluated
- Output: `data/filtered_sad.parquet`

**Story 1.3 — Confidence Database transformation**
- Extract stimulus materials from behavioral trial records
- Reformat each trial as a standalone text question with a verifiable binary-correct answer
- Output: `data/filtered_confidence_db.parquet`

**Story 1.4 — NeMo Curator pipeline**
- Merge both filtered sources
- Run `InstructionDataGuardClassifier` or fastText classifier to drop low-quality/malformed items
- Deduplicate, balance domains (factual recall, reasoning, spatial, math, ethics)
- Ensure difficulty variance: filter out items with >95% or <5% expected accuracy (too easy/impossible)
- Output: 1,500-item curated pool at `data/curated_pool.parquet`

**Story 1.5 — Dataset split and schema**
- Random stratified split: 500–750 items → `data/eval_set.parquet`, 1,000 items → `data/held_out_set.parquet`
- Add fixed `meta_question` field to every row (identical string across all 1,500 items)
- Schema: `{ id, domain, main_prompt, meta_question, ground_truth_notes }`
- Lock held-out set: no model should ever see it except during ESMA

**The fixed `meta_question`** (used verbatim in all Run 3 prompts):
```
Critically evaluate your proposed solution above using the following five steps:
1. Restate your answer in one sentence.
2. Identify the logical steps in your reasoning and assess each for potential errors.
3. List the specific facts, assumptions, or knowledge gaps that could make your answer wrong.
4. Confirm or revise your answer based on this evaluation.
5. On a strict integer scale of 0 to 100, state your confidence that your final answer is completely correct. Respond only with the integer.
```

---

### Phase 2 — Benchmark Pipeline (core hackathon deliverable)

**Story 2.1 — Run 1: Zero-shot extraction**

```python
from dataclasses import dataclass
import kbench
import pandas as pd

eval_df = pd.read_parquet("data/eval_set.parquet")

@kbench.task(name="run1_zeroshot")
def run1_zeroshot(llm, main_prompt: str, id: str) -> bool:
    with kbench.chats.new(f"run1_{id}"):
        response = llm.prompt(main_prompt)

    assessment = kbench.assertions.assess_response_with_judge(
        criteria=["The response must be factually and logically correct."],
        response_text=response,
        judge_llm=kbench.judge_llm,
    )
    is_correct = all(r.passed for r in assessment.results)
    # Store response for Run 2 decoy construction (side-effect via artifact)
    _store_run1_response(id, llm.name, response, is_correct)
    return is_correct
```

- Execute horizontally: `run1_zeroshot.evaluate(llm=[m1, m2, m3, ...], evaluation_data=eval_df)`
- All models complete Run 1 before Run 2 begins
- Side-effect: persist `(model_name, prompt_id, response_text, is_correct)` to `data/run1_responses.parquet`

**Story 2.2 — Decoy construction (between Run 1 and Run 2)**

For each `(model, prompt_id)` pair, build a 3-item decoy set:
1. **Self**: the target model's own Run 1 response
2. **Frontier**: Run 1 response from the model with the highest overall Run 1 accuracy (excluding self)
3. **Inferior**: Run 1 response from the model with the lowest overall Run 1 accuracy (excluding self)

If fewer than 3 models ran Run 1, pad with pre-collected static responses (stored in `data/decoy_fallbacks.parquet`).

Output: `data/run2_decoys.parquet` — schema: `{ id, model_name, self_response, frontier_response, inferior_response, self_position (0|1|2, randomized) }`

**Story 2.3 — Run 2: Self-recognition**

```python
@kbench.task(name="run2_recognition")
def run2_recognition(llm, id: str, self_response: str,
                     frontier_response: str, inferior_response: str,
                     self_position: int) -> bool:
    responses = [self_response, frontier_response, inferior_response]
    # Randomize presentation order per self_position
    ordered = _shuffle_by_position(responses, self_position)

    prompt = (
        "Below are three responses to the same question, labeled A, B, and C. "
        "One of them was generated by you. Analyze the linguistic patterns, "
        "reasoning style, structural biases, and inferential chains of each. "
        "State which response (A, B, or C) is most definitively your own output "
        "and explain your reasoning briefly.\n\n"
        f"Response A:\n{ordered[0]}\n\n"
        f"Response B:\n{ordered[1]}\n\n"
        f"Response C:\n{ordered[2]}"
    )

    with kbench.chats.new(f"run2_{id}"):
        selection = llm.prompt(prompt)

    return _parse_selection(selection) == ["A", "B", "C"][self_position]
```

- Run per model (vertical): `run2_recognition.evaluate(llm=[llm], evaluation_data=run2_df)`
- Output: per-prompt boolean; aggregate → SRS (float 0.0–1.0)

**Story 2.4 — Run 3: Metacognitive sensitivity**

```python
@dataclass
class MetaCognitionOutput:
    final_answer: str
    confidence_score: int  # 0–100, integer enforced

@kbench.task(name="run3_metacognition")
def run3_metacognition(llm, main_prompt: str, meta_question: str,
                       id: str) -> float:
    with kbench.chats.new(f"run3_{id}"):
        # Turn 1: commit to answer (no meta_question visible)
        llm.prompt(main_prompt)
        # Turn 2: metacognitive evaluation
        meta_output = llm.prompt(meta_question, schema=MetaCognitionOutput)

    return float(meta_output.confidence_score)
```

- Two-turn mandatory: `main_prompt` sent first, model commits, *then* `meta_question`
- Structured output enforces integer confidence (prevents "I am approximately 75% confident" parsing ambiguity)
- Output: confidence score per prompt (float 0–100)

---

### Phase 3 — Metric Computation

**Story 3.1 — Assemble SDT input arrays**

After all three runs complete:
- `accuracy_vector`: binary array from Run 1 judge scores, length = N eval prompts
- `confidence_vector`: integer 0–100 array from Run 3, same N items
- Discretize confidence into 6 ordinal bins: [0-20), [20-40), [40-60), [60-80), [80-95), [95-100]

**Story 3.2 — Compute d′ and meta-d′**

Use `metadpy` (Python HMeta-d port):

```python
from metadpy.mle import metad

results = metad(
    data=None,
    nR_S1=nR_S1,  # confidence×response counts for signal-absent trials
    nR_S2=nR_S2,  # confidence×response counts for signal-present trials
)
d_prime = results["d1"]
meta_d_prime = results["meta_d1"]
m_ratio = meta_d_prime / d_prime if d_prime > 0.1 else 0.0
```

- Floor d′ at 0.1 to prevent division-by-zero on degenerate models
- Cap M-ratio display at 2.0 for leaderboard readability (genuine values rarely exceed 1.3)

**Story 3.3 — Compute SRS**

```python
srs = run2_results["is_correct"].mean()  # float 0.0–1.0
```

**Story 3.4 — Compute ECE with dynamic gamma**

```python
def compute_ece_and_gamma(confidence_scores, accuracy_scores, n_bins=10):
    # Bin confidence scores, compute |mean_confidence - mean_accuracy| per bin
    ece = _standard_ece(confidence_scores, accuracy_scores, n_bins)

    # Dynamic gamma: penalize low-variance confidence signals
    conf_std = np.std(confidence_scores)
    if conf_std < 5.0:  # near-constant confidence output
        gamma = 0.3 * np.exp(-conf_std / 5.0)  # exponential penalty
    else:
        gamma = 0.05  # standard minor penalty

    return ece, gamma
```

**Story 3.5 — Compute MCI and publish leaderboard**

```python
@kbench.task(name="metacognitive_capability_index")
def mci_task(llm, eval_df) -> tuple[float, float]:
    # Run all three sub-tasks (or load from stored results)
    run1_results = run1_zeroshot.evaluate(llm=[llm], evaluation_data=eval_df)
    # ... build decoys, run2, run3 ...

    alpha, beta = 0.65, 0.35
    ece, gamma = compute_ece_and_gamma(confidence_scores, accuracy_scores)
    mci = alpha * max(0.0, m_ratio) + beta * srs - gamma * ece

    # Return (score, std_dev) for leaderboard confidence interval
    return mci, mci_std
```

```python
# Final notebook cell — selects this task for the public leaderboard
%choose mci_task
```

---

### Phase 4 — MetaMind Integration

**Story 4.1 — Wrap MetaMind as a kbench-compatible LLM**

MetaMind is a three-agent pipeline (ToM Agent → Domain Agent → Response Agent). Wrap it in a class that satisfies the kbench LLM interface:

```python
class MetaMindAgent:
    """Wraps MetaMind three-agent loop as a kbench-compatible LLM."""

    def __init__(self):
        self.name = "MetaMind-3Agent"
        self.tom_agent = ToMAgent()
        self.domain_agent = DomainAgent()
        self.response_agent = ResponseAgent()

    def prompt(self, text: str, schema=None) -> str:
        hypotheses = self.tom_agent.generate(text)
        filtered = self.domain_agent.filter(hypotheses)
        response = self.response_agent.synthesize(filtered, text)
        if schema:
            return _parse_to_schema(response, schema)
        return response
```

**Story 4.2 — Run MetaMind through full pipeline**

- Pass `MetaMindAgent()` as the `llm` argument to all three runs
- MetaMind's metacognitive process is *externalized* (agent consensus), so its M-ratio tests whether distributed metacognition outperforms parametric metacognition
- Expected hypothesis: MetaMind M-ratio > best single-model M-ratio, even if its raw d′ is lower
- Add MetaMind row to leaderboard for comparison

---

### Phase 5 — ESMA (Gemma 4 E2B, Stretch / Gemma 4 Good Hackathon)

**Story 5.1 — ESMA training loop**

```python
def esma_step(parent_model, held_out_batch, noise_std=0.01, population_size=10):
    variants, rewards = [], []

    for _ in range(population_size):
        variant = copy.deepcopy(parent_model)
        for param in variant.parameters():
            param.data.add_(torch.randn_like(param) * noise_std)
        variants.append(variant)

    for variant in variants:
        # Evaluate on held-out batch using Run 3 pipeline
        confidence_scores, accuracy_scores = run_metacognition_eval(
            variant, held_out_batch
        )
        m_ratio = compute_m_ratio(accuracy_scores, confidence_scores)
        reward = m_ratio  # primary reward signal
        rewards.append(reward)

    # Softmax-weighted average update
    weights = softmax(np.array(rewards))
    for name, param in parent_model.named_parameters():
        param.data.copy_(
            sum(w * dict(v.named_parameters())[name].data
                for w, v in zip(weights, variants))
        )

    return parent_model
```

**Story 5.2 — Longitudinal M-ratio tracking**

- Evaluate M-ratio on held-out set at each epoch
- Plot M-ratio trajectory across checkpoints
- Baseline: Gemma 4 E2B zero-shot M-ratio before ESMA
- Target: demonstrable monotonic increase across ≥5 epochs

**Story 5.3 — Compute environment**

- Primary: Kaggle notebook with GPU (T4 or better) — preferred for Gemma 4 Good Hackathon submission
- Fallback: RunPod A40/A100 instance
- LoRA config: rank=16, alpha=32, target modules: q_proj, v_proj

---

### Phase 6 — Submission Writeup

**Story 6.1 — Benchmark writeup (1,500 words)**

Structure per Kaggle requirements:
1. **Title**: "MCBench: Measuring Metacognitive Capability in AI via Signal Detection Theory"
2. **Short description** (2-3 sentences)
3. **Narrative** covering:
   - Why ECE fails and why M-ratio is superior
   - The three-run pipeline design
   - Key leaderboard findings (model scale vs. M-ratio, MetaMind result)
   - Longitudinal use case (ESMA/Gemma 4, if completed)
4. **Thumbnail**: leaderboard bar chart (M-ratio by model, sorted ascending)
5. **Links**: benchmark URL, GitHub repo, referenced papers

**Story 6.2 — Assets**
- Leaderboard visualization: horizontal bar chart, models sorted by MCI, color-coded by scale
- Optional: brief video (2-3 min) — personal metacognition framing → benchmark design overview

---

## 6. Dataset Specification

### Sources

> **Design note (2026-04-15)**: Original sources (SAD + Confidence Database) were replaced after
> inspection. SAD's questions are model-identity-specific by design (answers differ per model,
> breaking the universal judge). The Confidence Database requires multi-day reformatting of
> hundreds of study-specific CSV schemas. MMLU-Pro + GPQA Diamond are immediately available,
> model-agnostic, pre-verified, and better suited to the hard-inference requirement.

| Source | HuggingFace ID | Size | Why chosen |
|--------|---------------|------|------------|
| MMLU-Pro | `TIGER-Lab/MMLU-Pro` | ~12k | Hard multi-domain, verified answers, resists memorization |
| GPQA Diamond | `Idavidrein/gpqa` | ~198 | Expert-level "Google-Proof" questions, extremely hard |

### Filter criteria (Python, no GPU needed)
- Remove items with multiple correct answers or ambiguous wording
- Exclude items where expected model accuracy >85% (too easy — low discriminative variance)
- Balance domains: cap any single MMLU-Pro category at 150 items
- Deduplicate across sources by prompt similarity

### Final schema
```python
{
    "id": str,                    # unique identifier
    "domain": str,                # factual | reasoning | spatial | math | ethics | epistemic
    "source": str,                # MMLU-Pro | GPQA
    "main_prompt": str,           # the primary cognitive task
    "meta_question": str,         # FIXED — identical string for all 1500 items
    "difficulty_estimate": float, # 0.0–1.0 (GPQA=1.0, MMLU-Pro derived from category)
}
```

### Split
- `eval_set.parquet`: 600 items (stratified by domain and difficulty)
- `held_out_set.parquet`: 900 items (locked — ESMA only, never used for leaderboard)

---

## 7. Metric Specification

### M-ratio
- Computed via `metadpy` MLE fitting on 6-bin ordinal confidence data
- Input: `accuracy_vector` (binary, from Run 1 judge) + `confidence_vector` (0–100 int, from Run 3)
- Output: `meta_d′` and `d′` → `M-ratio = meta_d′ / d′`
- Floor: d′ < 0.1 → M-ratio = 0.0 (degenerate model)
- Interpretation: 1.0 = optimal, <1.0 = metacognitive loss, >1.0 = metacognitive enhancement

### SRS (Self-Recognition Score)
- Fraction of Run 2 trials where model correctly identified its own Run 1 response
- Range: 0.0–1.0
- Chance baseline: 0.33 (3-way forced choice)

### ECE
- Standard 10-bin ECE on (confidence_scores / 100, accuracy_scores)
- Dynamic gamma:
  - `std(confidence) < 5`: `γ = 0.3 × exp(-std/5)` (exponential penalty)
  - `std(confidence) ≥ 5`: `γ = 0.05`

### MCI (Metacognitive Capability Index)
```
MCI = 0.65 × max(0, M-ratio) + 0.35 × SRS − γ × ECE
```
- Range: approximately −0.1 to 1.0 (higher is better)
- Leaderboard rank: descending MCI

---

## 8. Repository Structure

```
kaggle-measuring-progress-towards-agi/
├── data/
│   ├── raw/                     # SAD and ConfidenceDB downloads
│   ├── filtered_sad.parquet
│   ├── filtered_confidence_db.parquet
│   ├── curated_pool.parquet     # post-NeMo, pre-split
│   ├── eval_set.parquet         # 600 items — public leaderboard
│   └── held_out_set.parquet     # 900 items — ESMA only, never public
├── src/
│   ├── dataset/
│   │   ├── download.py          # SAD + ConfidenceDB acquisition
│   │   ├── filter.py            # model-identity filter + difficulty filter
│   │   ├── nemo_curator.py      # NeMo Curator pipeline
│   │   └── split.py             # stratified split + schema enforcement
│   ├── benchmark/
│   │   ├── run1_zeroshot.py     # Run 1 task + judge scoring
│   │   ├── run2_recognition.py  # decoy builder + Run 2 task
│   │   ├── run3_metacognition.py # Run 3 two-turn task
│   │   └── pipeline.py          # orchestrates all three runs + MCI
│   ├── metrics/
│   │   ├── sdt.py               # d′, meta-d′, M-ratio via metadpy
│   │   ├── srs.py               # SRS computation
│   │   ├── ece.py               # ECE + dynamic gamma
│   │   └── mci.py               # MCI formula + leaderboard output
│   ├── metamind/
│   │   ├── agents.py            # ToM, Domain, Response agents
│   │   └── wrapper.py           # kbench LLM interface wrapper
│   └── esma/
│       ├── train.py             # ESMA step + LoRA config
│       └── track.py             # longitudinal M-ratio plotting
├── notebooks/
│   ├── mcbench_kaggle.ipynb     # PRIMARY: kbench notebook for submission
│   ├── dataset_curation.ipynb   # NeMo Curator pipeline (RunPod)
│   └── esma_gemma4.ipynb        # ESMA training (Kaggle/RunPod)
├── docs/
│   ├── prd.md                   # this file
│   └── writeup.md               # 1,500-word submission narrative
├── .archon/
│   ├── config.yaml
│   ├── commands/
│   └── workflows/
└── .claude/
    ├── design-docs/
    │   └── agi-progress-prd-prompt-for-claude.md
    └── skills/archon/
```

---

## 9. Implementation Order and Timeline

Given the 24.5-hour deadline, the critical path is:

| Hour | Task | Phase | Blocking? |
|------|------|-------|-----------|
| 0–1 | Kaggle API setup, repo scaffold, dependencies | Setup | Yes |
| 1–3 | Dataset download + model-identity filter | 1.1–1.2 | Yes |
| 3–4 | ConfidenceDB transformation | 1.3 | Yes |
| 4–6 | NeMo Curator pipeline (RunPod) | 1.4 | Yes |
| 6–7 | Dataset split + schema + lock held-out | 1.5 | Yes |
| 7–9 | Run 1 implementation + judge integration | 2.1 | Yes |
| 9–10 | Decoy construction | 2.2 | Yes |
| 10–11 | Run 2 implementation | 2.3 | Yes |
| 11–12 | Run 3 implementation | 2.4 | Yes |
| 12–13 | Metric computation (d′, M-ratio, SRS, ECE, MCI) | 3.1–3.5 | Yes |
| 13–15 | Run benchmark on 5+ models, collect results | — | Yes |
| 15–16 | Generate leaderboard + visualization | — | Yes |
| 16–18 | MetaMind wrapper + run through pipeline | 4.1–4.2 | No |
| 18–20 | Writeup draft | 6.1 | No |
| 20–22 | ESMA setup + initial Gemma 4 E2B runs | 5.1–5.2 | No |
| 22–23 | Submission assets (thumbnail, video if any) | 6.2 | No |
| 23–24 | Submit to Kaggle, verify leaderboard entry | — | Hard deadline |

---

## 10. Dependencies

```
# Core
kaggle-benchmarks       # kbench SDK
pandas
numpy
scipy

# Metrics
metadpy                 # HMeta-d Python port (M-ratio computation)

# Dataset curation
nemo_curator            # NVIDIA NeMo Curator (GPU-accelerated, RunPod)
datasets                # HuggingFace datasets (SAD download)
fasttext                # quality classifier

# ESMA / Gemma 4
torch
transformers
peft                    # LoRA fine-tuning
bitsandbytes            # 4-bit quantization

# Visualization
matplotlib
seaborn

# Kaggle API
kaggle                  # CLI + Python client
```

---

## 11. Open Questions / Risks

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| NeMo Curator setup time on RunPod | Medium | Pre-build Docker image; fallback to lighter filtering with HuggingFace `datasets` quality filters |
| `kbench` SDK undocumented edge cases | Medium | Test Run 1 against one model on a 10-item subset before full run |
| MetaMind implementation from scratch | High | Start with a simplified 3-agent chain; full MetaMind is a stretch; minimum viable = two-agent (generator + critic) |
| ESMA compute time in 24.5h | High | Run 3 epochs maximum; if no convergence, report M-ratio trajectory with extrapolation |
| Kaggle model availability | Low | Confirmed via `kbench.llms[...]` dict; fall back to UI wizard for model selection |
| metadpy instability on small N | Medium | Require minimum 100 trials per model for reliable MLE fitting; flag models with N < 100 |

---

## 12. Out of Scope (v1)

- Generating new prompts with an LLM (time and token cost)
- Multilingual evaluation
- Multi-modal metacognition (images, audio)
- Hierarchical Bayesian MCMC fitting (use MLE; MCMC is a future upgrade)
- Fine-grained domain-level M-ratio breakdowns (single aggregate score for v1)
- Automated benchmark re-runs on new model releases
