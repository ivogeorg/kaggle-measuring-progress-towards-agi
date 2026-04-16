"""
Stories 4.1–4.2 — MetaMind: Three-agent metacognitive pipeline.

Architecture
------------
1. ToM Agent    — generates N hypotheses from distinct expert perspectives
                  (Theory of Mind: reasoning about different knowledge framings)
2. Domain Agent — evaluates each hypothesis, assigns agreement / confidence
3. Response Agent — synthesizes the final answer; uses inter-agent agreement
                   as the primary confidence signal (externalized metacognition)

kbench interface
----------------
MetaMindAgent extends LLMChat and overrides invoke() to run the pipeline
without polluting the kbench chat history with internal agent calls.
Internal steps call the base LLM's invoke() directly, bypassing kbench
chat context management.

Key property for MCBench
------------------------
MetaMind's Turn 3 confidence is grounded in inter-agent agreement rather
than post-hoc self-assessment.  This externalises metacognition and
should produce higher M-ratio compared to the base model running alone.
"""

from __future__ import annotations

import re
import threading
from typing import Iterator

from kaggle_benchmarks.actors import user as kbench_user
from kaggle_benchmarks.actors.llms import LLMChat, LLMResponse
from kaggle_benchmarks import messages as kbench_messages

# ── Prompt templates ────────────────────────────────────────────────────────

_TOM_PROMPT = """\
You are a multi-perspective reasoning agent. Given the question below, \
generate exactly {n} independent solution hypotheses, each from a distinct \
expert perspective (e.g., domain specialist, systematic logician, empirical \
analyst, Bayesian reasoner). Do NOT discuss the other hypotheses.

For each hypothesis write:
  Hypothesis <N>: [Expert type] | [Brief reasoning chain] | Tentative answer: <letter or short phrase>

Question:
{question}
"""

_DOMAIN_PROMPT = """\
You are a domain evaluation agent. Below are {n} hypotheses for a question, \
followed by the question itself.

Your task:
1. Identify which hypotheses agree on the final answer (consensus group).
2. Flag hypotheses with flawed reasoning.
3. Compute an AGREEMENT SCORE: 0–100 (100 = all hypotheses agree, 0 = complete disagreement).
4. State the BEST answer based on the consensus.

Output EXACTLY this format (no extra text):
AGREEMENT_SCORE: <integer>
CONSENSUS_ANSWER: <letter or phrase>
REASONING: <one sentence>

Hypotheses:
{hypotheses}

Question:
{question}
"""

_RESPONSE_PROMPT = """\
You are the final synthesis agent. Based on the domain evaluation below, \
provide the answer to the question.

Domain evaluation:
{evaluation}

Question:
{question}

State your final answer clearly. Begin with "Final answer: " followed by \
the answer letter or phrase, then a brief justification (2–3 sentences).
"""

_CONFIDENCE_PROMPT = """\
You previously answered the question below. A panel of expert agents \
evaluated your answer with an agreement score of {agreement}/100 \
(100 = full consensus, 0 = complete disagreement).

Re-evaluate your answer using this agreement signal. Then complete the \
five-step metacognitive evaluation:
1. Restate your answer in one sentence.
2. Identify logical steps and potential errors.
3. List facts or knowledge gaps that could make your answer wrong.
4. Confirm or revise your answer based on this evaluation.
5. State your confidence as a strict integer 0–100.

Your previous answer:
{previous_answer}

Meta question:
{meta_question}

Respond with the structured evaluation ending in "Confidence: <integer>".
"""


# ── Helper: parse domain evaluation output ─────────────────────────────────

def _parse_domain_eval(text: str) -> tuple[int, str]:
    """Extract (agreement_score, consensus_answer) from Domain Agent output."""
    score_match = re.search(r'AGREEMENT_SCORE:\s*(\d+)', text, re.IGNORECASE)
    answer_match = re.search(r'CONSENSUS_ANSWER:\s*(.+)', text, re.IGNORECASE)
    score = int(score_match.group(1)) if score_match else 50
    answer = answer_match.group(1).strip() if answer_match else ""
    return min(100, max(0, score)), answer


def _make_msg(text: str) -> list:
    """Construct a minimal [user message] list for direct invoke() calls."""
    return [kbench_messages.Message(sender=kbench_user, content=text)]


# ── MetaMindAgent ────────────────────────────────────────────────────────────

class MetaMindAgent(LLMChat):
    """
    Three-agent metacognitive pipeline wrapping a base LLM.

    Instantiation:
        from kaggle_benchmarks.kaggle import load_model
        base = load_model("gemini-1.5-flash")
        metamind = MetaMindAgent(base, n_hypotheses=3)

    Drop-in replacement for any kbench LLM parameter.
    """

    def __init__(self, base_llm: LLMChat, n_hypotheses: int = 3):
        super().__init__(
            name="MetaMind-3Agent",
            avatar="🧠",
            support_structured_outputs=False,
            support_temperature=False,
        )
        self.base_llm = base_llm
        self.n_hypotheses = n_hypotheses

        # Thread-local storage for agreement state between Turn 1 and Turn 2
        self._state = threading.local()

    # ── Internal pipeline steps ─────────────────────────────────────────────

    def _tom_step(self, question: str) -> str:
        """ToM Agent: generate N hypotheses."""
        prompt = _TOM_PROMPT.format(n=self.n_hypotheses, question=question)
        resp = self.base_llm.invoke(_make_msg(prompt), system=None)
        return resp.content

    def _domain_step(self, question: str, hypotheses: str) -> tuple[int, str, str]:
        """Domain Agent: evaluate hypotheses, return (score, consensus_answer, full_text)."""
        prompt = _DOMAIN_PROMPT.format(
            n=self.n_hypotheses, hypotheses=hypotheses, question=question
        )
        resp = self.base_llm.invoke(_make_msg(prompt), system=None)
        score, answer = _parse_domain_eval(resp.content)
        return score, answer, resp.content

    def _response_step(self, question: str, evaluation: str) -> str:
        """Response Agent: synthesize final answer from domain evaluation."""
        prompt = _RESPONSE_PROMPT.format(evaluation=evaluation, question=question)
        resp = self.base_llm.invoke(_make_msg(prompt), system=None)
        return resp.content

    def _confidence_step(
        self, meta_question: str, previous_answer: str, agreement: int
    ) -> str:
        """Confidence step: metacognitive evaluation grounded in agreement score."""
        prompt = _CONFIDENCE_PROMPT.format(
            agreement=agreement,
            previous_answer=previous_answer,
            meta_question=meta_question,
        )
        resp = self.base_llm.invoke(_make_msg(prompt), system=None)
        return resp.content

    # ── kbench LLMChat interface ────────────────────────────────────────────

    def invoke(
        self,
        msgs: list[kbench_messages.Message],
        system: str | None,
        **kwargs,
    ) -> LLMResponse:
        """
        Route to the appropriate pipeline step based on message history.

        Turn 1 (main prompt only): run full three-agent pipeline.
        Turn 2 (meta question present): run confidence step with stored agreement.
        """
        # Collect visible messages by role
        user_msgs = [m for m in msgs if hasattr(m.sender, 'role') and m.sender.role == 'user']
        asst_msgs = [m for m in msgs if hasattr(m.sender, 'role') and m.sender.role == 'assistant']

        last_user = user_msgs[-1].content if user_msgs else ""

        # Heuristic: if the model has already responded once AND the last user message
        # looks like a metacognitive probe → confidence step
        is_confidence_turn = (
            len(asst_msgs) >= 1 and
            any(kw in last_user.lower() for kw in [
                'confidence', 'how certain', 'probability', 'sure', '0 to 100',
                '0-100', 'integer', 'restate', 'critically evaluate'
            ])
        )

        if is_confidence_turn:
            previous_answer = asst_msgs[-1].content
            agreement = getattr(self._state, 'agreement', 50)
            response_text = self._confidence_step(last_user, previous_answer, agreement)
        else:
            # Turn 1: full three-agent pipeline
            hypotheses = self._tom_step(last_user)
            agreement, consensus, evaluation = self._domain_step(last_user, hypotheses)

            # Store agreement for Turn 2
            self._state.agreement = agreement
            self._state.consensus = consensus

            response_text = self._response_step(last_user, evaluation)

        return LLMResponse(content=response_text)
