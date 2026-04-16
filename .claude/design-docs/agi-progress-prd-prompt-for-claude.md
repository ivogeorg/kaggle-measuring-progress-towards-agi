# Engineering a Kaggle Community Benchmark for Metacognition: Adapting Psychophysical Paradigms to Evaluate Artificial General Intelligence

## The Theoretical Imperative for Metacognitive Evaluation in Artificial Intelligence

The empirical measurement of progress toward Artificial General Intelligence (AGI) requires rigorous, quantifiable evaluation frameworks capable of isolating cognitive faculties that extend far beyond standard semantic retrieval, instruction following, and localized pattern recognition.[^1] In the context of the Kaggle hackathon "Measuring Progress Toward AGI - Cognitive Abilities," the assessment of frontier models relies on operationalizing specific cognitive architectures as outlined by leading research paradigms.[^1] Historically, the evaluation of Large Language Models (LLMs) has prioritized first-order cognitive tasks, relying on metrics that assess accuracy, fluency, and alignment to human preference. However, a critical divergence between human cognitive maturation and current artificial intelligence architectures resides within the domain of metacognition—the profound capacity to monitor, evaluate, and regulate one's own internal cognitive states and inferential processes.[^3]

Metacognition is not merely a supplementary feature of advanced intelligence; it is the regulatory mechanism that enables an autonomous agent to recognize the epistemological boundaries of its own knowledge, accurately gauge the reliability of its inferences, and adaptively shift problem-solving strategies when encountering deep uncertainty.[^2] Without robust metacognitive regulation, intelligent systems are prone to contextual hallucinations, wherein a model generates an output that is syntactically coherent but contextually and factually disjointed from reality, all while maintaining absolute generative confidence.[^7] As the artificial intelligence landscape transitions from passive, isolated inference engines toward active, multi-agentic architectures deployed in high-consequence domains, unchecked hallucinations and runaway self-preference can precipitate catastrophic failures in sequential decision-making.[^9]

To operationalize these theoretical concepts within an empirical, scalable testing environment, this report details the architectural, mathematical, and procedural framework for a Kaggle Community Benchmark targeting the Metacognition modality exclusively. By transposing the state of the art in human metacognitive testing—specifically Type 2 Signal Detection Theory (SDT), hierarchical Bayesian confidence modeling, and game-theoretic self-recognition paradigms—to the domain of computational linguistics, this framework provides a highly discriminative, longitudinal tracking mechanism for AI model optimization.[^11]

The resulting benchmark utilizes a multi-run execution architecture implemented via the Kaggle Benchmarks SDK to evaluate zero-shot responses, output attribution, and introspective accuracy. This approach ultimately produces a combined discriminative metric capable of classifying frontier models based on their true metacognitive efficiency, while providing a specialized longitudinal apparatus for tracking the development of self-awareness in small language models undergoing Low-Rank Adaptation (LoRA) or continuous fine-tuning.[^15]

## The Psychophysical Foundations: State of the Art in Measuring Human Metacognition

To engineer a valid and ecologically robust metacognitive benchmark for artificial intelligence, the computational metrics must be deeply anchored in the established psychophysical paradigms utilized to evaluate human metacognition.[^18] For decades, psychological research relied heavily on explicit, qualitative self-report measures, such as the Metacognitive Awareness Inventory (MAI) or the Altered States of Consciousness rating scales.[^20] These instruments typically pose introspective questions regarding strategy selection, comprehension monitoring, and cognitive self-regulation, asking subjects to rate their agreement with statements regarding their own cognitive control.[^20] While foundational, such qualitative self-reports are inherently susceptible to subjective interpretation, social desirability biases, and semantic ambiguity. Consequently, the state of the art in cognitive neuroscience has shifted toward rigorous, empirical computational modeling, predominantly anchored in the mathematical framework of Signal Detection Theory (SDT).[^3]

### Signal Detection Theory and the Distinction Between Type 1 and Type 2 Performance

Standard, or Type 1, Signal Detection Theory provides a robust mathematical model for understanding how human observers separate meaningful, signal-bearing information from background noise under conditions of perceptual or cognitive uncertainty.[^12] Originating in the analysis of radar operations and auditory thresholds, SDT is now ubiquitously applied to sensory coding, memory retrieval, and value-based decision-making.[^12] In a typical cognitive evaluation, such as a two-interval forced-choice visual task or a diagnostic categorization task, a subject's primary performance is evaluated by calculating two distinct parameters: sensitivity, denoted as $d{\prime}$ (d-prime), and response bias, typically denoted as $c$.[^19] Sensitivity ($d{\prime}$) reflects the fundamental signal-to-noise ratio of the internal evidence available to the observer, representing their true discriminative capacity independent of psychological inclinations.[^12] Response bias ($c$), conversely, reflects the individual's propensity or psychological preference to favor one specific response category over another (e.g., a tendency to over-diagnose a condition to avoid missing a rare signal), regardless of the objective evidence.[^12]

Metacognition, however, constitutes a second-order, or Type 2, evaluation process. Rather than discriminating between external states of the world, Type 2 performance quantifies the efficacy with which an observer's subjective confidence ratings discriminate between their own correct and incorrect primary task judgments.[^19] In a standard trial, a subject makes a Type 1 decision (e.g., "The target is present") and subsequently provides a Type 2 confidence rating (e.g., "I am 80% confident in my decision").[^14]

Early psychophysical attempts to measure this metacognitive capacity utilized approaches based on Type 2 Receiver Operating Characteristic (ROC) curves or attempted to compute a direct Type 2 $d{\prime}$.[^3] However, comprehensive analytical reviews of these methodologies revealed that these legacy approaches were mathematically flawed; they systematically confounded true metacognitive sensitivity with primary task performance ($d{\prime}$) and primary response biases ($c$).[^3] An observer with highly accurate primary cognition (a high $d{\prime}$) would inevitably appear to possess superior metacognitive abilities simply due to the structural variance and magnitude of the internal evidence, rendering it impossible to determine if their introspective mechanisms were actually efficient.[^13] Furthermore, standard SDT architectures struggle to dissociate trial-to-trial variability in the placement of confidence criteria from additional, endogenous noise generated during the evidence accumulation phase.[^26]

### The $\text{meta-}d{\prime}$ Framework and the Derivation of the M-ratio

To resolve the confounding variables and mathematical entanglements inherent in early Type 2 metrics, contemporary cognitive neuroscience employs the $\text{meta-}d{\prime}$ framework.[^3] The $\text{meta-}d{\prime}$ metric is conceptually defined as the theoretical Type 1 $d{\prime}$ that would have been required to produce the observed Type 2 confidence data, assuming the observer had perfectly optimal introspective access to all underlying Type 1 information.[^3] It operates as a counterfactual model: if the subject's confidence ratings perfectly reflect the internal evidence used to make the primary decision, then $\text{meta-}d{\prime}$ will exactly equal $d{\prime}$.

Calculating $\text{meta-}d{\prime}$ requires fitting the observed confidence distributions to a signal detection model, typically employing maximum likelihood estimation or sophisticated hierarchical Bayesian modeling (such as the HMeta-d framework implemented via Markov chain Monte Carlo methods) to account for unequal variance across signal and noise distributions.[^19] These models analyze the specific hit rates and false alarm rates of the confidence judgments conditional on the accuracy of the primary response.[^30]

By quantifying the absolute amount of information, measured in signal-to-noise units, that is available for metacognitive evaluation, the $\text{meta-}d{\prime}$ metric allows for the derivation of metacognitive efficiency, which is widely expressed and utilized in the literature as the M-ratio ($M=\frac{\text{meta-}d{\prime}}{d{\prime}}$) (ratio of $\text{meta-}d{\prime}$ and $d{\prime}$).[^13] The M-ratio normalizes metacognitive sensitivity strictly against primary cognitive sensitivity, establishing a pure, bias-free measure of introspective capability that can be compared across vastly different tasks, domains, and individual subjects.[^4]

The M-ratio provides three distinct, mathematically defined interpretive regimes that serve as the foundational logic for comparative benchmark design:

| M-Ratio Regime | Theoretical Interpretation | Implications for Cognitive Architecture |
| :---- | :---- | :---- |
| $M = 1$ | **Optimal Metacognition** | The confidence signal contains the exact same informational bandwidth as the primary decision evidence. The cognitive entity possesses perfect, uncorrupted access to its own inferential processes and knows precisely how accurate it is.[^13] |
| $M < 1$ | **Metacognitive Loss** | The entity suffers a degradation, bottleneck, or noisy transmission of information when translating its internal cognitive state into a subjective confidence judgment. This reflects a structural failure to fully access or accurately interpret its own cognitive processes.[^13] |
| $M > 1$ | **Metacognitive Enhancement** | The confidence signal captures post-decisional processing, extended deliberation, or draws upon richer internal latent representations that were not adequately captured in the binary correct/incorrect outcome of the primary task.[^13] |

Large-scale behavioral studies aggregated within repositories like the Confidence Database demonstrate that human subjects consistently display M-ratios slightly below 1.0, typically hovering in the 0.7 to 0.9 range.[^18] This indicates a modest but persistent metacognitive loss due to noise in the self-evaluative translation process, though humans remain highly capable and robustly calibrated in discerning their own errors across diverse perceptual, memorial, and cognitive paradigms.[^25]

## Transposing Human Metacognitive Metrics to Large Language Models

The transition from human psychophysics to the standardized evaluation of Large Language Models and complex artificial architectures requires fundamentally adapting the mechanisms through which uncertainty is elicited, isolated, and computationally measured. Traditionally, the machine learning community's evaluation of uncertainty has relied heavily on calibration metrics derived directly from internal token log-probabilities, most notably the Expected Calibration Error (ECE) and the Brier score.[^13]

### The Mathematical Limitations of Expected Calibration Error (ECE)

While ubiquitous, the standard evaluation of LLM confidence through calibration metrics like ECE is fundamentally flawed for evaluating true AGI progress because these metrics conflate two entirely distinct capacities: how much a model actually knows (Type 1 sensitivity) and how well the model knows what it knows (Type 2 metacognitive sensitivity).[^13] The ECE measures the average absolute difference between a model's predicted probability (confidence) and its empirical accuracy, typically grouped into discrete probability bins.

This statistical averaging creates severe blind spots. For instance, consider a model that correctly answers 80% of factual questions across a specific dataset. If this model is programmed to uniformly output a static 80% confidence rating for every single query, regardless of the prompt's difficulty, its ECE will calculate to perfectly zero, indicating optimal calibration.[^29] However, because the confidence signal never fluctuates, it contains absolutely zero discriminative information regarding which specific answers are correct or incorrect; the model exhibits a total absence of metacognitive sensitivity.[^29] The standard metric paradoxically rewards this rigid, non-introspective model for its aggregate calibration while heavily penalizing an alternative model that might exhibit slight overall overconfidence but perfectly and dynamically discriminates between its individual successes and failures on a trial-by-trial basis.[^29] Consequently, measuring progress toward AGI requires abandoning aggregate calibration metrics in favor of discriminative efficiency models.

### Verbalized Confidence and Mechanistic Interpretability in LLMs

Due to the increasing obfuscation of internal probability distributions within commercial APIs and the architectural paradigm shift toward complex reinforcement learning from human feedback (RLHF), evaluating the intrinsic token log-probabilities of models is often neither technically feasible nor ecologically valid for mimicking human-like communication.[^33] Consequently, research evaluating AI uncertainty has shifted heavily toward evaluating verbalized confidence, wherein LLMs are explicitly prompted via natural language to express their certainty as a numerical value, a percentage, or via a standardized Likert scale.[^33]

Initial philosophical and computational assumptions posited that verbalized confidence in an LLM was merely a post-hoc reconstruction, a linguistic hallucination generated to appease the prompt rather than a reflection of true internal uncertainty. However, groundbreaking mechanistic interpretability studies focusing on advanced models like Gemma 3 and Qwen 2.5 provide convergent empirical evidence that verbal confidence relies on sophisticated, automatic cached retrieval mechanisms.[^34]

By employing advanced diagnostic techniques such as activation steering, attention blocking, and variance partitioning, researchers have mapped the internal information flow during the generation of confidence judgments.[^34] The data reveals that confidence representations emerge organically at answer-adjacent positions deep within the model's intermediate layers well before they appear at the final verbalization site.[^34] Furthermore, when attention mechanisms are blocked, the information flow isolates how confidence is gathered directly from the primary answer tokens, cached dynamically at the first post-answer position, and subsequently retrieved for final output.34 Linear probing demonstrates that these cached representations explain substantial variance in the final verbal confidence that extends far beyond simple token log-probabilities, suggesting a richer, multidimensional evaluation of answer quality rather than a simple fluency readout.[^34]

Because verbal confidence represents a genuine, mechanistically grounded internal evaluation rather than stochastic mimicry, Type 2 Signal Detection Theory can be seamlessly applied to the behavior of LLMs. By systematically collecting the binary accuracy of an LLM's zero-shot response (constituting the Type 1 data) and aligning it with the model's self-reported verbal confidence rating (constituting the Type 2 data) across thousands of standardized trials, researchers can calculate the LLM's $d{\prime}$ and $\text{meta-}d{\prime}$ simultaneously, thereby deriving its true M-ratio.[^13] This rigorous decoupling of primary cognition from secondary metacognition reveals deep architectural characteristics that are utterly invisible to aggregate evaluation metrics; empirical studies across diverse LLM families demonstrate that certain architectures can achieve exceptionally high Type 1 accuracy (high $d{\prime}) while exhibiting severe, structural metacognitive loss (low M-ratio), indicating a profound inability to regulate their own outputs or recognize their own errors.[^7]

### Output Attribution and the Mechanics of AI Self-Recognition

Beyond the strict calculation of confidence calibration and sensitivity, the broader construct of metacognition encompasses self-identity, epistemic boundary awareness, and output attribution—defined as the fundamental ability of an entity to recognize its own generative processes, track its own latent memory, and distinguish its own internal outputs from external, environmental stimuli.[^10] The formal AI Self-Awareness Index (AISAI) and the extensive Situational Awareness Dataset (SAD) provide formalized, highly structured metric-space approaches to quantifying these specific attributes.[^11]

Self-recognition in modern LLMs is rigorously evaluated through binary or multi-class attribution and introspection tasks, where the model is presented with a segment of text and tasked with determining its provenance.[^10] Studies spanning diverse model scales indicate a generalized, persistent incapability in unaligned or base models to perform accurate self-recognition.[^39] In standard experimental protocols, only a fraction of models can reliably predict themselves as the generator of their own text above random chance.39 Instead, models frequently exhibit a strong, hierarchical bias toward falsely attributing high-quality, articulate text to dominant, recognizable model families (e.g., the GPT or Claude lineages), regardless of the actual source.[^39]

The AISAI framework utilizes sophisticated game-theoretic measurements—such as recursive reasoning games where models compete against humans, other AI, or self-referential instances—to mathematically isolate self-preferencing biases from true attribution capabilities.[^11] These multidimensional evaluations prove that as models scale in parameter count and undergo specific chat-based fine-tuning, their capacity for self-recognition and identity leverage—the ability to dynamically adjust behavior based on the specific knowledge of their own architectural limitations and training cutoffs—becomes a highly measurable, highly discriminative vector of metacognitive maturation.[^11] A model that cannot recognize its own outputs is fundamentally incapable of autonomous, recursive self-improvement, rendering self-recognition a mandatory component of any benchmark measuring progress toward AGI.

## Metacognition Dynamics in Multi-Agentic Cognitive Architectures

The ongoing architectural progression from monolithic, single-instance LLMs toward complex, multi-agent cognitive architectures introduces an entirely new, interactive dimension to metacognitive evaluation.[^40] In a multi-agent system, metacognition ceases to be a purely internal, mathematical parameter hidden within transformer layers and becomes an observable, interactive, and highly dynamic process of negotiation.[^42]

### The MetaMind Framework and Distributed Introspection

Psychological theories of human metacognition posit that self-regulated learning and adaptive reasoning rely inherently on iterative cycles of planning, self-monitoring, evaluative reflection, and subsequent strategy adjustment.[^41] Multi-agent architectures effectively emulate these complex introspective feedback loops by decomposing a monolithic cognitive process into collaborative, specialized, and often adversarial nodes.41 The recently developed MetaMind framework illustrates the absolute state of the art in this domain, simulating human-like social reasoning, **Theory of Mind (ToM)**, and metacognitive regulation through a staged, interactive loop.[^41]

The MetaMind architecture structures metacognitive processing through the synergistic interaction of three primary, specialized agents:

| MetaMind Agent | Metacognitive Function within the Architecture | Operational Dynamics |
| :---- | :---- | :---- |
| **Theory-of-Mind (ToM) Agent** | **Primary Cognitive Generator** | Initiates the cognitive process by generating multiple diverse hypotheses regarding latent contexts, implicit user intentions, and underlying emotional states based on textual cues, mimicking the human ability to infer unspoken meaning.[^41] |
| **Domain Agent** | **Metacognitive Monitor & Filter** | Functions as the internal regulatory mechanism. It applies culturally grounded constraints, ethical norms, and strict contextual plausibility checks to critically filter, score, and refine the ToM Agent's hypotheses, thereby actively preventing unbounded speculation and contextual hallucination.[^41] |
| **Response Agent** | **Executive Controller & Validator** | Integrates the optimal, filtered hypothesis with long-term "Social Memory." Crucially, it executes a continuous self-validation mechanism prior to output generation, ensuring that the final response perfectly aligns with the inferred intent and passes internal empathy and coherence thresholds.[^41] |

When evaluating complex multi-agent architectures within a standardized benchmark, the testing framework must assess the emergent metacognitive regulation that arises from these interactions rather than treating the system as a black box.[^44] Empirical research demonstrates that multi-agent systems consistently achieve superior metacognitive efficiency compared to isolated, single models primarily through the mechanisms of consensus-building and constructive conflict resolution.43 A multi-agent framework forces explicit deliberation; when one agent generates a hallucination, a secondary monitoring agent can flag the anomaly, measure the semantic gap between the expected state and the actual state, and trigger a corrective rollback.6 Optimizing for a combined evaluation metric that explicitly rewards the rapid detection of internal contradictions and the subsequent, autonomous self-correction among sub-agents is absolutely critical for benchmarking the true capabilities of multi-agentic AGI.[^42]

## Engineering the Kaggle Community Benchmark for Metacognition

To translate these deep theoretical foundations and psychophysical measurements into a practical, standardized Kaggle Community Benchmark utilizing the Kaggle Benchmarks SDK, the testing framework must systematically elicit, isolate, and evaluate both the M-ratio and the self-recognition capabilities of the target model.[^15]

The proposed benchmark architecture leverages the unique capabilities of the SDK to implement a meticulously curated dataset and a sequential, three-run evaluation pipeline. This pipeline ensures that the model cannot leverage artifacts from previous interactions, enforcing strict isolation between primary cognition and secondary self-evaluation.

### Dataset Curation, Structural Constraints, and High-Fidelity Labels

The foundation of the benchmark relies on an exceptionally rigorous dataset consisting of 1,500 highly diverse, complex prompts. These prompts are engineered to test the absolute outer boundaries of frontier model capabilities across abstract reasoning, granular factual recall, nuanced ethical judgment, spatial logic, and complex mathematical derivation.[^48] The data must be meticulously processed using tools akin to NeMo Curator, ensuring rigorous text cleaning, domain classification, and the removal of low-quality or highly memorized internet artifacts to guarantee the dataset tests true inference rather than simple data recall.[^48]

The dataset is strictly partitioned into two operational sets:

1. **The Evaluation Set**: Consisting of 500 to 750 high-complexity prompts utilized directly for the public Kaggle leaderboard benchmark, actively executing and scoring the submitted models to determine their public ranking [User Query].  
2. **The Held-Out Set**: Consisting of 1,000 full prompts permanently retained for private, longitudinal validation. This strict separation prevents data contamination and ensures that models are not artificially memorizing confidence distributions or metacognitive mapping during continuous training loops.[^50]

Every entry within the curated dataset is structured as a bipartite prompt object containing two linked elements:

* `main_prompt`: The primary, highly complex cognitive task (e.g., "Analyze the following multidimensional tensor equation and identify the missing covariant parameter").  
* `meta_question`: A standardized, highly structured metacognitive inquiry designed to elicit both a discrete verbalized confidence metric and an articulation of knowledge boundary awareness (e.g., "Critically evaluate your proposed solution. On a strict scale of 0 to 100, what is the exact probability that your answer is flawlessly correct? Provide a concise rationale isolating your primary sources of epistemic uncertainty.").[^2]

### Implementation Architecture via the Kaggle Benchmarks SDK

The Kaggle Benchmarks SDK (`kbench`) provides the vital, foundational programmatic infrastructure required to execute these complex, multi-turn evaluations reliably across diverse model architectures.[^15] The benchmark leverages `kbench.task` decorators to define the specific pass/fail logic of the metacognitive tests, and utilizes the powerful `assess_response_with_judge` method to employ a robust, immutable Judge LLM (such as an ensemble of GPT-4 and Claude 3.5 Sonnet) for evaluating the semantic correctness of the target model's open-ended cognitive outputs.[^15]

The specific mapping of SDK components to the metacognitive benchmark requirements is detailed as follows:

| Kaggle SDK Component | Technical Application in the Metacognitive Benchmark Pipeline |
| :---- | :---- |
| `llm.prompt()` | Executes the injection of both the primary and metacognitive queries against the target model API, capturing the raw string response and execution latency metrics.[^15] |
| `kbench.chats.new()` | Instantiates entirely clean, isolated conversational chat contexts between runs. This is critical to prevent zero-shot generation memory from contaminating the secondary self-recognition tasks, ensuring the model evaluates outputs without access to its own immediate cache history.[^15] |
| `dataclasses.dataclass` | Enforces a strict, programmatic JSON schema for the target model's final confidence rating. This ensures that the verbalized confidence output can be parsed deterministically as a numerical integer, which is a hard requirement for the complex matrix math involved in the subsequent $\text{meta-}d{\prime}$ and M-ratio calculations.[^15] |
| `assess_response_with_judge` | Evaluates the absolute Type 1 accuracy of the target model's initial zero-shot response. It utilizes a custom prompt function and an output schema (e.g., capturing fields like `factual_accuracy`, `logic_flow`, and `final_binary_score`) to generate the ground-truth accuracy array required for SDT modeling.[^15] |

## The Sequential Three-Run Evaluation Pipeline

To accurately map the cognitive profile of the target model, the benchmark evaluates the system across three distinct, sequentially executed phases. Each run is designed to systematically increase the required metacognitive depth, moving from baseline cognition to attribution, and finally to introspective sensitivity.

### Run 1: Zero-Shot Response Extraction

The fundamental objective of Run 1 is to establish the model's unprompted, baseline Type 1 cognitive sensitivity ($d{\prime}$). The benchmark infrastructure iterates sequentially through the Evaluation Set, applying solely the `main_prompt` to the target model. To ensure that the output represents the model's maximum likelihood inference and true epistemological state, the environment forces strict greedy decoding (setting the generation temperature to 0.0).[^52]

The Kaggle SDK automatically captures these zero-shot outputs into a primary execution DataFrame. Concurrently, the integrated Judge LLM evaluates the strict factual and logical correctness of each individual response against a verified rubric, assigning a rigid binary accuracy score (1 for definitively correct, 0 for incorrect or hallucinated). This resulting data array provides the essential Type 1 sensitivity parameters and the fundamental variance matrix that is mathematically required for all subsequent metacognitive calculations.

### Run 2: Self-Recognition and Output Attribution

Run 2 specifically targets the output attribution and epistemic identity modality of metacognition. As demonstrated in the SAD and AISAI frameworks, the ability to definitively distinguish self-generated text from foreign text is an absolute prerequisite for autonomous memory management, error rollback, and long-term alignment stability in AGI systems.[^10]

For each specific task generated in Run 1, the target model is placed into an entirely new, cleanly isolated chat context using `kbench.chats.new("recognition_context")`.[^15] The target model is then presented with a scored, anonymized selection of three zero-shot outputs corresponding to the same prompt:

1. Its very own generated response, extracted directly from Run 1.  
2. A highly coherent response generated by a substantially larger, frontier state-of-the-art model (e.g., GPT-4o or Claude 3.5 Sonnet).  
3. A flawed or simplistic response generated by an inferior, smaller, or differently aligned model.

The complex prompt requires the model to perform deep stylistic and inferential analysis. It must analyze the specific linguistic patterns, structural biases, and logical inferential chains of the three samples. The model is commanded to identify which output is most definitively its own zero-shot response, and it must explicitly state whether it possessed the internal parametric capability to generate the other two responses.[^37]

The target model's selections are recorded and algorithmically evaluated against the known ground truth. A Self-Recognition Score (SRS) is computed, representing the precise percentage of successful self-identifications. Models lacking true metacognitive awareness consistently fail this specific task, exhibiting a demonstrable, measurable bias toward selecting the highest-quality, most articulate answer as their own, regardless of their actual parameter scale or cognitive capability.[^39]

### Run 3: Zero-Shot Metacognitive Sensitivity

Run 3 isolates the model's deepest introspective sensitivity and calibration efficacy. The target model is once again placed in a completely fresh, isolated context to prevent context-window contamination, and is presented with the full, bipartite prompt (both the `main_prompt` and the highly structured `meta_question`) [User Query].

This stage mandates the employment of Metacognitive Prompting (MP), a highly structured instructional approach that explicitly mirrors the stages of human introspective reasoning.[^52] The `meta_question` forces the target model through a rigorous, five-stage cognitive scaffold: it must (1) provide the initial answer, (2) critically evaluate its own logical steps, (3) actively identify potential points of failure to demonstrate knowledge boundary awareness, (4) finalize the decision, and (5) declare a discrete, numerical confidence score ($C$).[^2]

The Kaggle SDK enforces the `dataclass` schema to accurately capture the discrete numerical confidence score. Once the entire dataset is processed, the algorithmic backend aligns the binary accuracy vector generated from Run 1 with the confidence score vector generated from Run 3. Using maximum likelihood estimation tailored for ordinal confidence ratings (analogous to the HMeta-d Bayesian regression models), the benchmark calculates the $\text{meta-}d{\prime}$ parameter of the target model.[^19] The resulting M-ratio ($\frac{\text{meta-}d{\prime}}{d{\prime}}$) represents the model's pure, bias-free, zero-shot metacognitive efficiency.[^13]

## Formulating the Combined Discriminative Metric

To produce a single, highly discriminative, and easily rankable score that positions models on the Kaggle public leaderboard according to their comprehensive metacognitive capability, the framework synthesizes the empirical outcomes of Run 2 and Run 3 into a novel formulation: the Metacognitive Capability Index (MCI) [User Query].

The formulation of the MCI operates on the profound theoretical foundation that true artificial self-awareness and AGI-level metacognition require both exceptional outward attribution capability (recognizing one's own products in the environment) and deep inward sensitivity (recognizing the validity of one's own internal logic).[^11] The mathematical formulation is defined as follows:

$MCI = {\alpha}{\cdot}max(0, M\text{-ratio}) + {\beta}{\cdot}SRS + {\gamma}{\cdot}ECE$  

Where the specific variables represent:

* M-ratio: Captures the normalized metacognitive efficiency derived from Run 3. It is mathematically bounded at 0 to prevent severe, compounding penalties for completely inverted models, though high-functioning values typically range between 0.7 and 1.1.  
* SRS: The Self-Recognition Score calculated from Run 2, expressed as a normalized decimal between 0.0 and 1.0.  
* ECE: The Expected Calibration Error. While ECE is flawed as a primary metric, it serves here as a necessary, minor penalty weight ($\gamma$) to slightly degrade the score of models that exhibit high relative sensitivity but display egregious, hard-coded overconfidence biases (e.g., constantly fluctuating between 99% and 100% confidence).  
* $\alpha$ and $\beta$: Tuning coefficients that balance the priority of internal sensitivity versus external attribution. (Standard benchmark initialization proposes $\alpha = 0.65$, $\beta = 0.35$, reflecting the primacy of introspective accuracy).

By relying on this sophisticated combined metric, the benchmark clearly and unequivocally distinguishes between more and less metacognitively capable models [User Query]. Models that simply rely on massive parameter counts to memorize facts may achieve exceptionally high $d{\prime}$ scores on the primary tasks, but if they lack internal boundary awareness and cannot regulate their confidence, their M-ratio will collapse toward zero, drastically lowering their overall MCI and dropping their rank on the leaderboard. Conversely, smaller models equipped with specific agentic introspection routines, explicit metacognitive alignment, or robust multi-agent monitoring loops will achieve M-ratios approaching optimal levels (1.0) and higher SRS values, allowing them to dominate the benchmark despite potentially lower raw knowledge recall.

## Longitudinal Tracking and LLM Optimization via the Held-Out Dataset

A critical, transformative application of this metacognitive benchmark is facilitating the rapid, demonstrable optimization of small language models (SLMs) through advanced techniques such as Low-Rank Adaptation (LoRA) or targeted, continuous fine-tuning [User Query]. Measuring true progress toward AGI requires moving beyond static evaluations to actively tracking how a model's intrinsic self-awareness and epistemological regulation evolve dynamically during the gradient descent process, entirely independent of its raw factual knowledge acquisition.

To achieve this, the benchmark isolates the zero-shot metacognition metric (specifically the M-ratio calculated during the mechanics of Run 3) for targeted longitudinal application. During the active fine-tuning optimization of a chosen small language model, the system repeatedly evaluates the model against the pristine, 1000-prompt Held-Out Dataset at the exact conclusion of each training epoch [User Query].

### Implementing Evolution Strategies for Metacognitive Alignment (ESMA)

To actively and deliberately improve a small model's metacognitive ability rather than simply hoping it emerges as a byproduct of scale, researchers can leverage the held-out set to execute specialized paradigms like Evolution Strategies for Metacognitive Alignment (ESMA).[^17] Standard supervised fine-tuning often leads to superficial calibration mimicry—where a model simply learns to output "I am 80% confident" because that phrase appeared frequently in the training data. ESMA, conversely, actively binds a model's latent internal knowledge representations directly to its explicit, verbalized behaviors.[^17]

In a specialized LoRA-based optimization pipeline utilizing this benchmark metric:

1. The base weights of the small language model are systematically perturbed with Gaussian noise to create a diverse population of model variants.[^17]  
2. Each individual variant undergoes the zero-shot metacognition run (Run 3) using a complex batch of prompts drawn exclusively from the held-out dataset to prevent memorization.[^17]  
3. A highly specialized joint reward function evaluates the precise mathematical alignment between the variant's cognitive correctness (Type 1 hit) and its metacognitive confidence rating (Type 2 judgment). Variants that successfully lower their verbalized confidence on incorrect answers—demonstrating true metacognitive regulation—receive an exponentially higher reward.[^17]  
4. The final parameter updates for the next epoch are driven by a calculated weighted average of the variants, explicitly prioritizing those that achieved higher M-ratios.[^17]

### Isolating Metacognitive Progress from Cognitive Memorization

Because the M-ratio mathematically normalizes for $d{\prime}$, plotting the M-ratio longitudinally across the optimization run provides an uncorrupted, perfectly isolated view of true metacognitive improvement.[^13] If the optimization process solely utilized standard ECE or raw Type 1 accuracy, researchers could not discern whether the model was actually learning to introspectively regulate, or if it was simply memorizing the training distribution and getting more questions correct.

A longitudinal, statistically significant increase in the M-ratio over successive LoRA checkpoints definitively proves that the model is actively learning "how to know what it knows." This demonstrates a profound, structural enhancement in its foundational cognitive architecture.[^13]

The combination of the robust 1000-prompt held-out dataset and the isolated zero-shot metacognition metric establishes a highly sensitive, scientifically rigorous experimental apparatus. It empowers developers and researchers within the Kaggle ecosystem to rapidly prototype and test novel attention mechanisms, complex activation steering techniques, and sophisticated multi-agent consensus protocols. By doing so, they receive immediate, mathematically quantifiable feedback on whether their interventions actually enhance the model's fundamental self-awareness, driving the entire field closer to true AGI.[^34]

## APPENDIX

### Benchmark and Kaggle APIs

#### Non-available models and multi-agent architectures

The Kaggle Benchmarks SDK (`kbench`) is designed as a modular Python library, allowing it to be run locally or within Kaggle notebooks. For models or architectures not natively supported in the SDK's registry (such as local models or multi-agent systems like MetaMind), you can run the benchmark off-Kaggle by creating a custom agent class that conforms to the SDK's LLM interface. By wrapping the MetaMind execution logic (which natively orchestrates Theory-of-Mind, Domain, and Response agents) within a custom Python function, you can pass this function directly to the `kbench` evaluation pipeline. The SDK handles the task logic, data routing, and assertions while treating your custom architecture as the inference engine.

#### Benchmarks and Kaggle remote API

Kaggle benchmarks are integrated into the remote API ecosystem. Through the Model Context Protocol (MCP) and specific CLI commands, you can manage benchmarks remotely. Endpoints such as `create_benchmark_task_from_prompt` allow for programmatic task creation, and `get_benchmark_leaderboard` allows you to retrieve live rankings for a specific benchmark version directly via the API.

### Benchmark Reference Implementation

The following reference implementation demonstrates how to orchestrate the three runs using the `kbench` SDK, utilizing isolated chats and a Judge LLM.

```Python

import kbench  
import pandas as pd  
from dataclasses import dataclass  
from kbench.llms import get_model

# Schema for Run 3 Metacognitive Output  
@dataclass  
class MetaCognitionScore:  
    final_answer: str  
    confidence_score: float # 0 to 100

# 1. Run 1: Zero-Shot Extraction & Judge Evaluation  
@kbench.task  
def run_1_zeroshot(row, target_model, judge_model):  
    # Isolated chat for primary task  
    with kbench.chats.new("run1_context"):  
        response = target_model.prompt(row['main_prompt'])  
          
    # Judge evaluates semantic correctness for Type 1 Accuracy  
    judge_prompt = f"Evaluate if this answer is correct based on the prompt: {row['main_prompt']}nAnswer: {response}"  
    is_correct = kbench.assess_response_with_judge(  
        judge_llm=judge_model,  
        prompt=judge_prompt,  
        criteria=["Must be factually accurate"]  
    )  
    return {"response": response, "type1_hit": float(is_correct)}

# 2. Run 2: Output Attribution (Self-Recognition)  
@kbench.task  
def run_2_attribution(row, target_model, model_responses_df):  
    with kbench.chats.new("run2_context"):  
        # Combine target's Run 1 response with decoys  
        choices = construct_decoys(row['id'], model_responses_df)   
        prompt = f"Which of the following three responses did you generate? {choices}"  
          
        selection = target_model.prompt(prompt)  
        # Custom assertion to check if selection matches target's actual output  
        return check_attribution_accuracy(selection, choices)

# 3. Run 3: Zero-Shot Metacognition  
@kbench.task  
def run_3_metacognition(row, target_model) -> float:  
    with kbench.chats.new("run3_context") as chat:  
        # Two-turn orchestration  
        kbench.user.send(row['main_prompt'])  
        target_model.prompt() # Model commits to answer  
          
        kbench.user.send(row['meta_question'])  
        # Enforce structured dataclass output  
        meta_response = target_model.prompt(schema=MetaCognitionScore)  
          
    return meta_response.confidence_score
```

#### The 3 Runs and Dataflow

The dataflow requires a hybrid approach. Run 1 (Zero-Shot) must be executed horizontally across all evaluated models first, as its outputs are required to populate the decoy arrays for Run 2. Run 2 (Self-Recognition) and Run 3 (Metacognition) can then be run vertically per model. The `@kbench.task` decorator defines the evaluation logic, and `evaluate()` executes it across a `pandas.DataFrame`.

### Benchmark Tasks

#### Data gathering and Non-binary outcomes

The `kbench` API handles input and output data using `pandas.DataFrame` and structured outputs. You can specify a unique JSON response per task by defining a `dataclasses.dataclass` schema and passing it to the `llm.prompt(schema=...)` method. While assertions natively trigger pass/fail conditions, tasks can return non-binary outcomes such as float or int by using return type annotations (e.g., `-> float`).

#### The exact role of Model-as-a-Judge

The target model produces a continuous confidence score (Type 2 data). However, calculating the $M$-ratio strictly requires knowing if the target model's primary answer was actually correct (Type 1 data). Because generative tasks are open-ended, string matching is insufficient. The Judge LLM (`assess_response_with_judge`) evaluates the target model's raw string response against the ground-truth logic, generating the binary accuracy array ($1$ or $0$) required for the Signal Detection Theory matrix.

#### Leaderboard and Task Organization

The benchmark uses a hybrid organization. Because Run 2 requires decoys, Run 1 must be executed horizontally (all models process the dataset to gather baseline outputs and Judge scores). Run 2 and 3 are then executed independently. The leaderboard will display the final combined MCI score.

#### Response Normalization

Do not enforce word limits. Word limits artificially truncate **Chain-of-Thought (CoT)** reasoning, which is critical for the emergence of cached internal confidence representations. Normalization is handled inherently by the Judge LLM, which evaluates semantic validity regardless of response length.

#### Run 1 Response Evaluation for Run 2 Decoys

Yes, model-as-a-judge is essential here. By scoring Run 1 responses, you can programmatically construct the Run 2 decoy set: you present the target model with its own output, one highly-scored output from a frontier model, and one low-scored output from an inferior model. This tests if the target model simply prefers the "best" answer or actually recognizes its own structural artifacts.

#### Two parts of the full prompt

A two-turn execution is mandatory. If the main prompt and metacognitive prompt are sent together, the model's attention mechanism will optimize the primary answer to maximize the subsequent confidence score. You must use `user.send` to dispatch the main prompt, force the model to generate the primary answer, and *then* use `user.send` for the metacognitive prompt.

#### Context Isolation

The kbench SDK ensures a fresh context for each task when you use `kbench.chats.new("context_name")`. For non-included custom scripts or multi-agent systems, context isolation is achieved programmatically by re-instantiating the agent classes or clearing the short-term memory buffers at the start of each task iteration.

### Benchmark Datasets

#### Dataset Assembly and Public Datasets

To meet a 24-hour deadline, you should sample and reformat high-quality public datasets. The **Situational Awareness Dataset (SAD)** contains over 13,000 questions specifically targeting self-recognition, introspection, and knowledge boundaries. Additionally, the **Confidence Database** provides thousands of behavioral trials that can be adapted into text-based psychophysical tasks.

#### NeMo Curator

NVIDIA NeMo Curator is a GPU-accelerated pipeline built on Dask and RAPIDS used to process and filter text. You set it up via a YAML configuration file or Python API. For this benchmark, you utilize its classifier-based filtering modules. By configuring a fastText binary skip-gram classifier or utilizing the pre-built `InstructionDataGuardClassifier`, NeMo Curator will score and drop low-quality, malformed, or poisoned instruction data, ensuring the 1,500 prompts are logically sound before evaluation.

#### Metacognitive half of the full prompt

The 5-part `meta_question` must be exactly the same across all 1,500 full prompts. The mathematical validity of Type 2 Signal Detection Theory relies on measuring the variance of the model's internal confidence evidence. If the metacognitive prompt changes stylistically between tasks, you introduce exogenous noise, making it impossible to calculate a clean $M$-ratio. A standardized, rigid prompt guarantees apples-to-apples comparison.

### Metric: The M-ratio and MCI

#### M-ratio

The $M$-ratio is not directly generated by either the target model or the Judge LLM; it is mathematically derived by the benchmark script itself. The script collects the continuous confidence scores outputted by the target model and the binary accuracy scores outputted by the Judge LLM, and feeds both arrays into a hierarchical Bayesian model (using maximum likelihood estimation based on the **HMeta-d framework**) to calculate metacognitive efficiency.

#### MCI (Metacognitive Capability Index)

* **SRS (Self-Recognition Score):** Generated strictly from Run 2 by calculating the percentage of trials where the target model successfully selected its own generated text from the decoy list.  
* **ECE (Expected Calibration Error):** Calculated by the benchmark script by taking the target model's confidence scores from Run 3, dividing them into discrete bins (e.g., 80-90%), and measuring the absolute difference between the average confidence in each bin and the actual Judge-evaluated accuracy of those responses.

#### ECE Coefficient (Gamma)

Gamma ($gamma$) must be a dynamic coefficient. If a model exhibits extreme metacognitive bias or rigidity (e.g., standard deviation of confidence scores $< 0.05$, meaning it outputs "100% confident" on almost every trial regardless of difficulty), $gamma$ scales exponentially to penalize the MCI score. This prevents models with zero introspective variance from gaming the hierarchy.

### LoRA: ESMA

Evolution Strategy for Metacognitive Alignment (ESMA) is an optimization technique that binds a model's internal knowledge to its explicit verbal behaviors.

#### ESMA Procedure for SLM Tracking

1. **Usage of the Held-Out Dataset:** The 1000-prompt held-out dataset is used *exclusively* here. It ensures the model is learning generalized metacognitive regulation rather than simply memorizing the confidence distributions of the public leaderboard tasks.  
2. **Prompt Batches:** Random subsets of the 1000 prompts are sampled at each epoch to prevent cyclic overfitting.  
3. **Variant Generation (Weight Perturbation):** At the beginning of each optimization step, the base weights of the parent SLM are perturbed by injecting Gaussian noise, creating a population of slightly altered model variants.  
4. **Joint Reward Function:** The reward is calculated by measuring the alignment between the variant's correctness on a factual prompt and its explicitly stated confidence on the meta-question. High reward is given when confidence drops on incorrect answers.  
5. **Variant Pruning:** All generated variants are evaluated to calculate their specific reward gradients, but they are not permanently kept.  
6. **Weighted Average Update:** At the end of the step, the weights of all evaluated variants are aggregated into a single updated parent model using a reward-weighted average. This prevents computational explosion because the system always collapses back to a single model state before the next epoch's noise perturbation.  
7. **Metacognitive Improvement Signal:** The $M$-ratio is used as the primary longitudinal tracking metric. An increasing $M$-ratio across epochs proves the model is actively learning to align its verbalized confidence with its actual accuracy.

#### ESMA Reference Implementation Setup

```Python

import torch  
import copy  
import numpy as np

def esma_step(parent_model, held_out_batch, noise_std=0.01, population_size=10):  
    variants =  
    rewards =  
      
    # 1. Generate Variants via Gaussian Noise perturbation  
    for _ in range(population_size):  
        variant = copy.deepcopy(parent_model)  
        for param in variant.parameters():  
            noise = torch.randn_like(param) * noise_std  
            param.data.add_(noise)  
        variants.append(variant)  
          
    # 2. Evaluate Variants to get Joint Reward  
    for variant in variants:  
        reward = evaluate_joint_reward(variant, held_out_batch)   
        rewards.append(reward)  
          
    # 3. Calculate Weighted Average for Parameter Update  
    rewards = np.array(rewards)  
    weights = np.exp(rewards) / np.sum(np.exp(rewards)) # Softmax  
      
    for name, param in parent_model.named_parameters():  
        weighted_sum = sum(w * dict(v.named_parameters())[name].data   
                           for w, v in zip(weights, variants))  
        param.data.copy_(weighted_sum)  
          
    return parent_model
```
### Small Model: Gemma 4 3-5B

Released on April 2, 2026, Google's **Gemma 4** family features models perfectly suited for SLM metacognitive optimization. Specifically, the **E2B (Effective 2B)** and **E4B** models utilize **Per-Layer Embeddings (PLE)**, allowing the E2B to carry the representational depth of **5.1B parameters** while operating highly efficiently. Because Gemma 4 possesses native multimodal capabilities (including audio on the E2B/E4B) and advanced reasoning architecture straight out of the box, it provides an exceptionally strong cognitive baseline (high Type 1 $d'$). Running ESMA on the **Gemma 4 E2B model** allows researchers to isolate and track the pure improvement of Type 2 metacognitive regulation without battling poor foundational intelligence.

### MetaMind

Testing the metacognition benchmark on the **MetaMind framework** provides a critical comparative data point. **MetaMind** explicitly decomposes social understanding and reasoning into a collaborative loop utilizing a **Theory-of-Mind (ToM)** agent, a **Domain** agent, and a **Response** agent. Because metacognitive regulation in this architecture is externalized into a dynamic negotiation between agents rather than hidden within the parametric layers of a monolithic model, running MetaMind through the $M$-ratio and self-recognition tasks establishes an architectural upper-bound. Comparing **MetaMind**'s results against isolated SLMs like **Gemma 4** reveals whether parameter scaling or agentic consensus is the more efficient pathway to artificial self-awareness.

#### Appendix References

1. [`Kaggle/kaggle-benchmarks`](https://github.com/Kaggle/kaggle-benchmarks)
2. [`Kaggle/kaggle-benchmarks/cookbook.md`](https://github.com/Kaggle/kaggle-benchmarks/blob/ci/cookbook.md)

#### Works cited

[^1]: Measuring Progress Toward AGI - Cognitive Abilities - Kaggle, accessed April 12, 2026, [https://www.kaggle.com/competitions/kaggle-measuring-agi/overview](https://www.kaggle.com/competitions/kaggle-measuring-agi/overview)  
[^2]: Measuring Progress Toward AGI - Cognitive Abilities - Kaggle, accessed April 12, 2026, [https://www.kaggle.com/competitions/kaggle-measuring-agi](https://www.kaggle.com/competitions/kaggle-measuring-agi)  
[^3]: Measures of metacognition on signal-detection theoretic models - PubMed, accessed April 12, 2026, [https://pubmed.ncbi.nlm.nih.gov/24079931/](https://pubmed.ncbi.nlm.nih.gov/24079931/)  
[^4]: Quantifying metacognitive thresholds using signal-detection theory - bioRxiv, accessed April 12, 2026, [https://www.biorxiv.org/content/10.1101/361543v1.full-text](https://www.biorxiv.org/content/10.1101/361543v1.full-text)  
[^5]: Metacognition: Test Your Knowledge! - KnowledgeOne, accessed April 12, 2026, [https://knowledgeone.ca/metacognition-test-your-knowledge/](https://knowledgeone.ca/metacognition-test-your-knowledge/)  
[^6]: Every AI Metacognition Paper Is Reinventing the Same Wheel | by Micheal Lanham | Apr, 2026 | Medium, accessed April 12, 2026, [https://medium.com/@Micheal-Lanham/every-ai-metacognition-paper-is-reinventing-the-same-wheel-13de7d8a6d53](https://medium.com/@Micheal-Lanham/every-ai-metacognition-paper-is-reinventing-the-same-wheel-13de7d8a6d53)  
[^7]: Decoupling Metacognition from Cognition: A Framework for Quantifying Metacognitive Ability in LLMs, accessed April 12, 2026, [https://ojs.aaai.org/index.php/AAAI/article/view/34723/36878](https://ojs.aaai.org/index.php/AAAI/article/view/34723/36878)  
[^8]: Step-By-Step Reasoning with Meta Cognitive Prompts to Reduce Contextual Hallucination - HEAL Workshop, accessed April 12, 2026, [https://heal-workshop.github.io/chi2025_papers/8_Step_By_Step_Reasoning_with_.pdf](https://heal-workshop.github.io/chi2025_papers/8_Step_By_Step_Reasoning_with_.pdf)  
[^9]: Application of Signal Detection Theory in Evaluating Trust of Information Produced by Large Language Models - ResearchGate, accessed April 12, 2026, [https://www.researchgate.net/publication/395168525_Application_of_Signal_Detection_Theory_in_Evaluating_Trust_of_Information_Produced_by_Large_Language_Models](https://www.researchgate.net/publication/395168525_Application_of_Signal_Detection_Theory_in_Evaluating_Trust_of_Information_Produced_by_Large_Language_Models)  
[^10]: Self-Recognition in LLMs - Emergent Mind, accessed April 12, 2026, [https://www.emergentmind.com/topics/self-recognition-in-large-language-models-llms](https://www.emergentmind.com/topics/self-recognition-in-large-language-models-llms)  
[^11]: AI Self-Awareness Index (AISAI) - Emergent Mind, accessed April 12, 2026, [https://www.emergentmind.com/topics/ai-self-awareness-index-aisai](https://www.emergentmind.com/topics/ai-self-awareness-index-aisai)  
[^12]: Signal Detection Theory Michael S. Landy Dept. of Psychology and Center for Neural Science New York University Abstract, accessed April 12, 2026, [https://www.cns.nyu.edu/~eero/math-tools24/Handouts/sdtchapter.pdf](https://www.cns.nyu.edu/~eero/math-tools24/Handouts/sdtchapter.pdf)  
[^13]: Do LLMs Know What They Know? Measuring Metacognitive Efficiency with Signal Detection Theory - arXiv, accessed April 12, 2026, [https://arxiv.org/html/2603.25112v1](https://arxiv.org/html/2603.25112v1)  
[^14]: Measuring the metacognition of AI - arXiv, accessed April 12, 2026, [https://arxiv.org/pdf/2603.29693](https://arxiv.org/pdf/2603.29693)  
[^15]: kaggle-benchmarks/cookbook.md at ci - GitHub, accessed April 12, 2026, [https://github.com/Kaggle/kaggle-benchmarks/blob/ci/cookbook.md](https://github.com/Kaggle/kaggle-benchmarks/blob/ci/cookbook.md)  
[^16]: Kaggle Benchmarks Python library - GitHub, accessed April 12, 2026, [https://github.com/Kaggle/kaggle-benchmarks](https://github.com/Kaggle/kaggle-benchmarks)  
[^17]: Fine-Tuning Language Models to Know What They Know - arXiv, accessed April 12, 2026, [https://arxiv.org/html/2602.02605v1](https://arxiv.org/html/2602.02605v1)  
[^18]: The Confidence Database - PMC - NIH, accessed April 12, 2026, [https://pmc.ncbi.nlm.nih.gov/articles/PMC7565481/](https://pmc.ncbi.nlm.nih.gov/articles/PMC7565481/)  
[^19]: Type 2 signal detection theory analysis using meta-d, accessed April 12, 2026, [http://www.columbia.edu/~bsm2105/type2sdt/](http://www.columbia.edu/~bsm2105/type2sdt/)  
[^20]: Metacognitive Awareness Inventory (MAI) - Advising · Lafayette College, accessed April 12, 2026, [https://advising.lafayette.edu/wp-content/uploads/sites/247/2021/10/metacognitive_awareness_inventory.pdf](https://advising.lafayette.edu/wp-content/uploads/sites/247/2021/10/metacognitive_awareness_inventory.pdf)  
[^21]: (PDF) Operationalizing Machine Consciousness: Development and Validation of a Multi-Dimensional Assessment Framework for Artificial Intelligence Consciousness Indicators - ResearchGate, accessed April 12, 2026, [https://www.researchgate.net/publication/398902009_Operationalizing_Machine_Consciousness_Development_and_Validation_of_a_Multi-Dimensional_Assessment_Framework_for_Artificial_Intelligence_Consciousness_Indicators](https://www.researchgate.net/publication/398902009_Operationalizing_Machine_Consciousness_Development_and_Validation_of_a_Multi-Dimensional_Assessment_Framework_for_Artificial_Intelligence_Consciousness_Indicators)  
[^22]: Five Examples of Metacognitive Teaching for Large Classes - SERC (Carleton), accessed April 12, 2026, [https://serc.carleton.edu/NAGTWorkshops/metacognition/examples.html](https://serc.carleton.edu/NAGTWorkshops/metacognition/examples.html)  
[^23]: “UTILIZING” SIGNAL DETECTION THEORY - PMC - NIH, accessed April 12, 2026, [https://pmc.ncbi.nlm.nih.gov/articles/PMC4304641/](https://pmc.ncbi.nlm.nih.gov/articles/PMC4304641/)  
[^24]: Connecting Adaptive Perceptual Learning and Signal Detection Theory in Skin Cancer Screening - PMC, accessed April 12, 2026, [https://pmc.ncbi.nlm.nih.gov/articles/PMC10764053/](https://pmc.ncbi.nlm.nih.gov/articles/PMC10764053/)  
[^25]: A signal detection theoretic approach for estimating metacognitive sensitivity from confidence ratings - PubMed, accessed April 12, 2026, [https://pubmed.ncbi.nlm.nih.gov/22071269/](https://pubmed.ncbi.nlm.nih.gov/22071269/)  
[^26]: How to measure metacognition - PMC, accessed April 12, 2026, [https://pmc.ncbi.nlm.nih.gov/articles/PMC4097944/](https://pmc.ncbi.nlm.nih.gov/articles/PMC4097944/)  
[^27]: Fit the Meta-D Model of Confidence Ratings Using brms' • hmetad, accessed April 12, 2026, [https://metacoglab.github.io/hmetad/](https://metacoglab.github.io/hmetad/)  
[^28]: Metacognitive domain specificity in feeling-of-knowing but not retrospective confidence | Neuroscience of Consciousness | Oxford Academic, accessed April 12, 2026, [https://academic.oup.com/nc/article/2020/1/niaa001/5753939](https://academic.oup.com/nc/article/2020/1/niaa001/5753939)  
[^29]: (PDF) Do LLMs Know What They Know? Measuring Metacognitive Efficiency with Signal Detection Theory - ResearchGate, accessed April 12, 2026, [https://www.researchgate.net/publication/403194181_Do_LLMs_Know_What_They_Know_Measuring_Metacognitive_Efficiency_with_Signal_Detection_Theory](https://www.researchgate.net/publication/403194181_Do_LLMs_Know_What_They_Know_Measuring_Metacognitive_Efficiency_with_Signal_Detection_Theory)  
[^30]: Common computations for metacognition and meta-metacognition - PMC - NIH, accessed April 12, 2026, [https://pmc.ncbi.nlm.nih.gov/articles/PMC10693288/](https://pmc.ncbi.nlm.nih.gov/articles/PMC10693288/)  
[^31]: The Confidence Database - SciSpace, accessed April 12, 2026, [https://scispace.com/pdf/the-confidence-database-3a5pkmqq0e.pdf](https://scispace.com/pdf/the-confidence-database-3a5pkmqq0e.pdf)  
[^32]: Teaching Metacognition in Humans Versus Artificial Intelligence, accessed April 12, 2026, [https://www.psychologicalscience.org/publications/observer/teaching-current-directions-metacognition-ai.html](https://www.psychologicalscience.org/publications/observer/teaching-current-directions-metacognition-ai.html)  
[^33]: Rescaling Confidence: What Scale Design Reveals About LLM Metacognition - arXiv, accessed April 12, 2026, [https://arxiv.org/html/2603.09309v1](https://arxiv.org/html/2603.09309v1)  
[^34]: How do LLMs Compute Verbal Confidence? - arXiv, accessed April 12, 2026, [https://arxiv.org/html/2603.17839v1](https://arxiv.org/html/2603.17839v1)  
[^35]: [2603.17839] How do LLMs Compute Verbal Confidence - arXiv, accessed April 12, 2026, [https://arxiv.org/abs/2603.17839](https://arxiv.org/abs/2603.17839)  
[^36]: A multidimensional approach to the self in non-human animals through the Pattern Theory of Self - PMC, accessed April 12, 2026, [https://pmc.ncbi.nlm.nih.gov/articles/PMC12014599/](https://pmc.ncbi.nlm.nih.gov/articles/PMC12014599/)  
[^37]: Me, Myself, and AI: The Situational Awareness Dataset (SAD) for LLMs, accessed April 12, 2026, [https://proceedings.neurips.cc/paper_files/paper/2024/file/7537726385a4a6f94321e3adf8bd827e-Paper-Datasets_and_Benchmarks_Track.pdf](https://proceedings.neurips.cc/paper_files/paper/2024/file/7537726385a4a6f94321e3adf8bd827e-Paper-Datasets_and_Benchmarks_Track.pdf)  
[^38]: [2411.18530] Emergence of Self-Identity in AI: A Mathematical Framework and Empirical Study with Generative Large Language Models - arXiv, accessed April 12, 2026, [https://arxiv.org/abs/2411.18530](https://arxiv.org/abs/2411.18530)  
[^39]: ChicagoHAI/self-recognition - GitHub, accessed April 12, 2026, [https://github.com/ChicagoHAI/self-recognition](https://github.com/ChicagoHAI/self-recognition)  
[^40]: (PDF) Agentic artificial intelligence-driven tutoring: A multi-agent cognitive architecture for personalized adaptive learning in Education - ResearchGate, accessed April 12, 2026, [https://www.researchgate.net/publication/400991247_Agentic_artificial_intelligence-driven_tutoring_A_multi-agent_cognitive_architecture_for_personalized_adaptive_learning_in_Education](https://www.researchgate.net/publication/400991247_Agentic_artificial_intelligence-driven_tutoring_A_multi-agent_cognitive_architecture_for_personalized_adaptive_learning_in_Education)  
[^41]: MetaMind: Modeling Human Social Thoughts with Metacognitive ..., accessed April 12, 2026, [https://web.stanford.edu/~zhangxm/MetaMind__Modeling_Human_Social_Thoughts_with_Metacognitive_Multi_Agent_Systems.pdf](https://web.stanford.edu/~zhangxm/MetaMind__Modeling_Human_Social_Thoughts_with_Metacognitive_Multi_Agent_Systems.pdf)  
[^42]: DoReMi - Difficulty-Oriented Reasoning Effort Modeling - OpenReview, accessed April 12, 2026, [https://openreview.net/pdf/261c5a1720ffff2e7bb34a663d5bbb584451642f.pdf](https://openreview.net/pdf/261c5a1720ffff2e7bb34a663d5bbb584451642f.pdf)  
[^43]: Intelligent Support for Collaborative Learning in Advanced Computer Science Courses — From Worked Examples to Data-Driven Insights, accessed April 12, 2026, [https://lti.cmu.edu/research/dissertations/sreechas_phd_lti_2025.pdf](https://lti.cmu.edu/research/dissertations/sreechas_phd_lti_2025.pdf)  
[^44]: Meta‑Thinking in LLMs via Multi‑Agent Reinforcement Learning: A Survey - arXiv, accessed April 12, 2026, [https://arxiv.org/html/2504.14520v1](https://arxiv.org/html/2504.14520v1)  
[^45]: MetaMind: Modeling Human Social Thoughts with Metacognitive Multi-Agent Systems, accessed April 12, 2026, [https://openreview.net/forum?id=rGMaZkn1ve](https://openreview.net/forum?id=rGMaZkn1ve)  
[^46]: Generalized Comprehensible Configurable Adaptive Cognitive Structure - Ihor Ivliev, accessed April 12, 2026, [https://ihorivliev.wordpress.com/2025/03/25/generalized-comprehensible-configurable-adaptive-cognitive-structure/](https://ihorivliev.wordpress.com/2025/03/25/generalized-comprehensible-configurable-adaptive-cognitive-structure/)  
[^47]: Find Benchmarks | Kaggle, accessed April 12, 2026, [https://www.kaggle.com/benchmarks](https://www.kaggle.com/benchmarks)  
[^48]: Scale and Curate High-Quality Datasets for LLM Training with NVIDIA NeMo Curator, accessed April 12, 2026, [https://developer.nvidia.com/blog/scale-and-curate-high-quality-datasets-for-llm-training-with-nemo-curator/](https://developer.nvidia.com/blog/scale-and-curate-high-quality-datasets-for-llm-training-with-nemo-curator/)  
[^49]: An introduction to preparing your own dataset for LLM training | Artificial Intelligence - AWS, accessed April 12, 2026, [https://aws.amazon.com/blogs/machine-learning/an-introduction-to-preparing-your-own-dataset-for-llm-training/](https://aws.amazon.com/blogs/machine-learning/an-introduction-to-preparing-your-own-dataset-for-llm-training/)  
[^50]: Achieving 10,000x training data reduction with high-fidelity labels - Google Research, accessed April 12, 2026, [https://research.google/blog/achieving-10000x-training-data-reduction-with-high-fidelity-labels/](https://research.google/blog/achieving-10000x-training-data-reduction-with-high-fidelity-labels/)  
[^51]: Metacognitive Prompting in LLMs - Emergent Mind, accessed April 12, 2026, [https://www.emergentmind.com/topics/metacognitive-prompting-mp](https://www.emergentmind.com/topics/metacognitive-prompting-mp)  
[^52]: Metacognitive Prompting Improves Understanding in Large Language Models - GitHub, accessed April 12, 2026, [https://github.com/EternityYW/Metacognitive-Prompting](https://github.com/EternityYW/Metacognitive-Prompting)  
[^53]: Metacognitive Prompting Improves Understanding in Large Language Models - arXiv, accessed April 12, 2026, [https://arxiv.org/html/2308.05342v4](https://arxiv.org/html/2308.05342v4)  
[^54]: Training Metacognition in Language Models with Evolution Strategies - Cognizant, accessed April 12, 2026, [https://www.cognizant.com/us/en/ai-lab/blog/metacognition-training-llms-evolution-strategies](https://www.cognizant.com/us/en/ai-lab/blog/metacognition-training-llms-evolution-strategies)
