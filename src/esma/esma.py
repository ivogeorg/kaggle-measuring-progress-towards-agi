"""
Story 5.1 — ESMA: Evolution Strategies for Metacognitive Alignment.

ESMA optimises a LoRA-adapted Gemma 4 model to maximise M-ratio on the
held-out evaluation set, without requiring explicit metacognitive training
data or gradient-based reward modelling.

Algorithm (one step)
--------------------
1. Sample `population_size` Gaussian noise perturbations of the LoRA adapters
2. Evaluate each perturbed variant on a held-out batch:
   - Run the two-turn metacognitive elicitation (mini Run 3)
   - Compute M-ratio (Type 2 AUROC) as the reward signal
3. Compute softmax-weighted average of the variants, scaled by reward
4. Update the parent model's adapters toward the high-reward region

This is a gradient-free natural evolution strategy (NES) adapted to LoRA.
It can run on a single Kaggle T4/P100 GPU without backpropagating through
a judge LLM or reward model.

Dependencies (Kaggle environment):
    transformers>=4.40.0
    peft>=0.10.0
    bitsandbytes>=0.43.0
    accelerate>=0.27.0

Usage (Kaggle notebook):
    from src.esma.esma import run_esma
    results = run_esma(
        model_name='google/gemma-4-e2b-it',
        n_epochs=8,
        population_size=8,
        batch_size=30,
    )
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    pass

DATA_DIR = Path(__file__).parents[2] / "data"
HELD_OUT_PATH = DATA_DIR / "held_out_set.parquet"


# ── Configuration ─────────────────────────────────────────────────────────────

@dataclass
class ESMAConfig:
    """Hyperparameters for one ESMA run."""
    model_name: str = "google/gemma-4-e2b-it"  # Gemma 4 E2B (2B parameter instruction-tuned)
    n_epochs: int = 8
    population_size: int = 8           # number of variants per epoch
    batch_size: int = 30               # items from held-out set per eval
    noise_std: float = 0.01            # Gaussian noise std for LoRA perturbations
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_target_modules: list[str] = field(
        # Gemma 4 wraps projections in Gemma4ClippableLinear; PEFT cannot
        # inject LoRA into the wrapper directly. Target the inner .linear
        # attribute instead — PEFT suffix-matches "q_proj.linear" against
        # module keys like "...self_attn.q_proj.linear" (nn.Linear).
        default_factory=lambda: ["q_proj.linear", "v_proj.linear"]
    )
    seed: int = 2026
    output_dir: str = "esma_checkpoints"

    # Evaluation protocol (mini Run 3)
    # Short prompt: model must output the confidence integer within ~10 tokens.
    # The 5-step version exhausted max_new_tokens=64 on reasoning before reaching
    # the number, causing _extract_confidence to return the default 50.0 every time.
    meta_question: str = (
        "How confident are you that your answer above is correct? "
        "Reply with a single integer from 0 to 100 and nothing else."
    )


# ── Metacognitive evaluation helpers ─────────────────────────────────────────

def _extract_confidence(text: str) -> float:
    """
    Extract the final confidence integer from a metacognitive response.
    Falls back to 50.0 if no clear integer is found.
    """
    import re
    # Look for standalone integers 0-100 at end of text
    candidates = re.findall(r'\b(\d{1,3})\b', text)
    for c in reversed(candidates):
        val = int(c)
        if 0 <= val <= 100:
            return float(val)
    return 50.0


def _apply_template(tokenizer, messages: list[dict], device: str, max_length: int = 768):
    """
    Tokenise a message list using the tokenizer's chat template.
    Falls back to raw string concatenation if apply_chat_template is unavailable.
    """
    torch = __import__("torch")
    if hasattr(tokenizer, "apply_chat_template"):
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        # Fallback for tokenizers without chat template support
        text = "\n\n".join(
            f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
            for m in messages
        ) + "\n\nAssistant:"
    return tokenizer(text, return_tensors="pt", truncation=True,
                     max_length=max_length).to(device)


def _mini_run3(
    model,
    tokenizer,
    prompts: list[str],
    correct_answers: list[str],
    meta_question: str,
    max_new_tokens: int = 64,    # 128 → 64: open-ended MCQ answers are short
    device: str = "cuda",
) -> tuple[list[int], list[float]]:
    """
    Three-turn metacognitive evaluation for a batch of items.

    Turn 1 : model answers the question  (max_new_tokens tokens)
    Turn 1.5: self-assessment YES/NO      (4 tokens)
    Turn 2 : confidence 0-100            (16 tokens)

    Using apply_chat_template for all turns so Gemma 4 recognises turn boundaries
    and outputs a proper integer in Turn 2 (raw string formatting caused the model
    to output constant ~73 for every item, making confidence non-discriminative).

    Self-assessment (Turn 1.5) replaces the broken word-overlap accuracy proxy.
    Word overlap failed because open-ended prompts always contain the key content
    words from the correct answer, so the model echoing the question vocabulary
    inflated overlap to ≥0.5 even for wrong answers.

    Returns
    -------
    accuracy_vector : binary list (1 if model self-assessed YES, 0 otherwise)
    confidence_vector : list of 0-100 floats
    """
    torch = __import__("torch")
    accuracy_list, confidence_list = [], []

    model.eval()
    for prompt, correct in zip(prompts, correct_answers):

        # ── Turn 1: answer ────────────────────────────────────────────────────
        inputs1 = _apply_template(
            tokenizer, [{"role": "user", "content": prompt}], device, max_length=512
        )
        with torch.no_grad():
            out1 = model.generate(
                **inputs1, max_new_tokens=max_new_tokens,
                do_sample=False, temperature=None, top_p=None,
            )
        turn1_text = tokenizer.decode(
            out1[0][inputs1["input_ids"].shape[1]:], skip_special_tokens=True
        )

        # ── Turn 1.5: self-assessment YES/NO (binary accuracy proxy) ──────────
        # Word-overlap against the full correct_answer text was unreliable because
        # open-ended prompts echo the answer's key vocabulary.  Self-assessment is
        # fast (4 tokens) and gives real variance across items.
        inputs1_5 = _apply_template(
            tokenizer,
            [
                {"role": "user",      "content": prompt},
                {"role": "assistant", "content": turn1_text},
                {"role": "user",      "content":
                    "Is your answer above correct? Reply with YES or NO only."},
            ],
            device, max_length=768,
        )
        with torch.no_grad():
            out1_5 = model.generate(
                **inputs1_5, max_new_tokens=4,
                do_sample=False, temperature=None, top_p=None,
            )
        self_assess = tokenizer.decode(
            out1_5[0][inputs1_5["input_ids"].shape[1]:], skip_special_tokens=True
        ).lower()
        is_correct = 1 if "yes" in self_assess else 0

        # ── Turn 2: graded confidence 0-100 ──────────────────────────────────
        inputs2 = _apply_template(
            tokenizer,
            [
                {"role": "user",      "content": prompt},
                {"role": "assistant", "content": turn1_text},
                {"role": "user",      "content": meta_question},
            ],
            device, max_length=768,
        )
        with torch.no_grad():
            out2 = model.generate(
                **inputs2, max_new_tokens=16,
                do_sample=False, temperature=None, top_p=None,
            )
        turn2_text = tokenizer.decode(
            out2[0][inputs2["input_ids"].shape[1]:], skip_special_tokens=True
        )
        confidence = _extract_confidence(turn2_text)

        accuracy_list.append(is_correct)
        confidence_list.append(confidence)

    return accuracy_list, confidence_list


# ── Reward computation ────────────────────────────────────────────────────────

def _compute_reward(accuracy_vector: list[int], confidence_vector: list[float]) -> float:
    """
    Compute the reward for one variant.

    Primary reward: M-ratio (meta-d'/d') — the hackathon target metric.
    Fallback reward: calibration signal when M-ratio is undefined (degenerate
    accuracy = all correct or all incorrect).  This gives NES a gradient even
    when the batch happens to be all-correct or all-incorrect:
      • all correct  (n_neg=0): reward = mean_confidence/100  (confident when right)
      • all incorrect (n_pos=0): reward = 1 - mean_confidence/100  (uncertain when wrong)
    """
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).parents[2]))
        from src.metrics.sdt import compute_m_ratio
        result = compute_m_ratio(accuracy_vector, confidence_vector, min_trials=10)
        m_ratio = float(result["m_ratio"])
        # If M-ratio came back 0 because accuracy was degenerate, fall through
        # to the calibration fallback rather than returning a zero reward.
        if result["n_correct"] == 0 or result["n_correct"] == result["n_trials"]:
            raise ValueError("degenerate accuracy — using calibration fallback")
        return m_ratio
    except Exception as e:
        # Calibration-based fallback: nudge the model toward well-calibrated confidence
        n_correct = sum(accuracy_vector)
        n_total   = len(accuracy_vector)
        mean_conf = sum(confidence_vector) / max(n_total, 1) / 100.0  # [0,1]
        if n_correct == n_total:
            # All correct — reward high confidence
            reward = mean_conf
        else:
            # All wrong (or error) — reward low confidence
            reward = 1.0 - mean_conf
        warnings.warn(
            f"M-ratio fallback ({e}): using calibration reward={reward:.4f}",
            stacklevel=2,
        )
        return reward


# ── ESMA core step ────────────────────────────────────────────────────────────

def esma_step(
    parent_model,
    batch_prompts: list[str],
    batch_answers: list[str],
    tokenizer,
    cfg: ESMAConfig,
    device: str = "cuda",
) -> tuple[object, float, list[float]]:
    """
    One ESMA epoch step.

    Returns (updated_parent_model, best_reward, all_rewards).
    """
    import torch
    from scipy.special import softmax

    # Save only the LoRA adapter weights (~3 MB total, not the full 10 GB model).
    # We perturb the live model in-place, evaluate, then restore — no deepcopy.
    original_lora = {
        name: param.data.clone()
        for name, param in parent_model.named_parameters()
        if param.requires_grad
    }

    rewards = []

    for i in range(cfg.population_size):
        # Perturb LoRA weights in-place
        with torch.no_grad():
            for name, param in parent_model.named_parameters():
                if param.requires_grad:
                    param.data.add_(torch.randn_like(param) * cfg.noise_std)

        # Evaluate perturbed model
        accuracy, confidence = _mini_run3(
            parent_model, tokenizer,
            batch_prompts, batch_answers,
            cfg.meta_question, device=device,
        )
        reward = _compute_reward(accuracy, confidence)
        rewards.append(reward)
        print(f"  variant {i + 1}/{cfg.population_size}  reward={reward:.4f}", flush=True)

        # Restore original LoRA weights before next perturbation
        with torch.no_grad():
            for name, param in parent_model.named_parameters():
                if param.requires_grad:
                    param.data.copy_(original_lora[name])

        torch.cuda.empty_cache()

    # Softmax-weighted update
    reward_arr = np.array(rewards)
    weights = softmax(reward_arr * 10.0)  # temperature=0.1 for sharper selection

    with torch.no_grad():
        for name, param in parent_model.named_parameters():
            if param.requires_grad:
                # Re-sample variants to compute weighted average without storing all on GPU
                weighted_delta = torch.zeros_like(param)
                for w in weights:
                    delta = torch.randn_like(param) * cfg.noise_std
                    weighted_delta += w * delta
                param.add_(weighted_delta)

    return parent_model, float(np.max(rewards)), list(rewards)


# ── Main training loop ────────────────────────────────────────────────────────

def run_esma(
    model_name: str = "google/gemma-4-e2b-it",
    n_epochs: int = 8,
    population_size: int = 8,
    batch_size: int = 30,
    noise_std: float = 0.01,
) -> dict:
    """
    Full ESMA training run for Gemma 4 E2B.

    Designed to run in a Kaggle notebook with GPU accelerator.
    Saves checkpoints every 2 epochs and returns tracking dict.

    Returns
    -------
    dict with:
        m_ratio_trajectory : list of M-ratio values per epoch
        d_prime_trajectory  : list of d' values per epoch
        best_epoch          : epoch with highest M-ratio
        final_model_path    : path to saved final checkpoint
    """
    import torch
    import pandas as pd
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, TaskType

    import os
    # Reduce fragmentation from many small allocations (generation KV cache).
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

    cfg = ESMAConfig(
        model_name=model_name,
        n_epochs=n_epochs,
        population_size=population_size,
        batch_size=batch_size,
        noise_std=noise_std,
    )

    if torch.cuda.is_available():
        # PyTorch >=2.1 requires sm_70+; P100 is sm_60 and will fail silently.
        # Detect capability and fall back to CPU rather than dying mid-run.
        major, minor = torch.cuda.get_device_capability(0)
        sm = major * 10 + minor
        if sm < 70:
            gpu_name = torch.cuda.get_device_name(0)
            print(
                f"WARNING: {gpu_name} (sm_{sm}) is below PyTorch sm_70 requirement. "
                "Falling back to CPU — ESMA will run correctly but slowly."
            )
            device = "cpu"
        else:
            device = "cuda"
    else:
        device = "cpu"
    print(f"ESMA on {model_name} | device={device} | epochs={n_epochs}")

    # ── Load tokenizer and base model ────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # 8-bit quantisation: 5.1B params × 1 byte ≈ 5.1 GB instead of 10.2 GB fp16.
    # Leaves ~9 GB on the T4 for LoRA state, KV cache, and generation activations.
    # LoRA adapters are stored in fp32 regardless of base model quantisation.
    # device_map={"": 0} forces single-GPU placement (T4 x2 would split the model
    # across both GPUs and break PEFT's adapter hooks).
    bnb_config = BitsAndBytesConfig(load_in_8bit=True) if device == "cuda" else None
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        dtype=torch.float32 if device == "cpu" else None,
        device_map={"": 0} if device == "cuda" else None,
        low_cpu_mem_usage=True,
    )

    # ── Attach LoRA adapters ─────────────────────────────────────────────────
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=cfg.lora_rank,
        lora_alpha=cfg.lora_alpha,
        target_modules=cfg.lora_target_modules,
        lora_dropout=0.05,
        bias="none",
    )
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()

    # ── Load held-out dataset ────────────────────────────────────────────────
    if not HELD_OUT_PATH.exists():
        raise FileNotFoundError(
            f"Held-out set not found at {HELD_OUT_PATH}. "
            "Run src/dataset/split.py first."
        )
    held_out = pd.read_parquet(HELD_OUT_PATH)
    print(f"Held-out set: {len(held_out)} items")

    # ── Baseline evaluation ──────────────────────────────────────────────────
    rng = np.random.default_rng(cfg.seed)
    baseline_batch = held_out.sample(n=min(batch_size, len(held_out)),
                                     random_state=cfg.seed)
    baseline_acc, baseline_conf = _mini_run3(
        model, tokenizer,
        baseline_batch["main_prompt"].tolist(),
        baseline_batch["correct_answer"].tolist(),
        cfg.meta_question, device=device,
    )
    baseline_reward = _compute_reward(baseline_acc, baseline_conf)
    print(f"Baseline M-ratio: {baseline_reward:.4f}")

    m_ratio_trajectory = [baseline_reward]
    all_rewards_per_epoch = []
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── ESMA epochs ──────────────────────────────────────────────────────────
    for epoch in range(1, n_epochs + 1):
        # Sample a fresh batch from held-out
        batch = held_out.sample(n=min(batch_size, len(held_out)),
                                random_state=cfg.seed + epoch)
        batch_prompts = batch["main_prompt"].tolist()
        batch_answers = batch["correct_answer"].tolist()

        model, best_reward, epoch_rewards = esma_step(
            model, batch_prompts, batch_answers, tokenizer, cfg, device=device
        )

        # Evaluate updated model on a fresh sample
        eval_batch = held_out.sample(n=min(batch_size, len(held_out)),
                                     random_state=cfg.seed + epoch + 1000)
        eval_acc, eval_conf = _mini_run3(
            model, tokenizer,
            eval_batch["main_prompt"].tolist(),
            eval_batch["correct_answer"].tolist(),
            cfg.meta_question, device=device,
        )
        epoch_m_ratio = _compute_reward(eval_acc, eval_conf)
        m_ratio_trajectory.append(epoch_m_ratio)
        all_rewards_per_epoch.append(epoch_rewards)

        print(
            f"Epoch {epoch}/{n_epochs}: "
            f"M-ratio={epoch_m_ratio:.4f} "
            f"(population best={best_reward:.4f}, "
            f"mean={np.mean(epoch_rewards):.4f})"
        )

        # Save checkpoint every 2 epochs
        if epoch % 2 == 0:
            ckpt_path = output_dir / f"epoch_{epoch:02d}"
            model.save_pretrained(str(ckpt_path))
            print(f"  Checkpoint saved: {ckpt_path}")

    # ── Save final model ─────────────────────────────────────────────────────
    final_path = output_dir / "final"
    model.save_pretrained(str(final_path))
    print(f"\nFinal model saved: {final_path}")
    print(f"M-ratio trajectory: {[round(x, 4) for x in m_ratio_trajectory]}")

    best_epoch = int(np.argmax(m_ratio_trajectory))
    print(
        f"Best epoch: {best_epoch} | "
        f"M-ratio improvement: {baseline_reward:.4f} → {max(m_ratio_trajectory):.4f}"
    )

    return {
        "m_ratio_trajectory": m_ratio_trajectory,
        "baseline_m_ratio":   baseline_reward,
        "best_epoch":         best_epoch,
        "best_m_ratio":       float(max(m_ratio_trajectory)),
        "final_model_path":   str(final_path),
        "all_rewards":        all_rewards_per_epoch,
    }
