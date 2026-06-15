# Handout A — Why this works (technical)

**Goal.** Prune each benchmark to the smallest subset that still tells us whether a model is *good
enough* for this customer — preserving both the **accuracy estimate** and the **model ranking** — and
do it so the method is defensible for a **model we haven't seen**. Implemented inside
`evalscope` (`@c14dbaf9`) as one **universal** pruning adapter; see `evalscope_ext/README.md`.

## Part A — coding (LCB v5) + long-context (AA-LCR)

**The problem I'm actually solving.** Not "estimate accuracy from a random sample" — a random sample is
an unbiased *accuracy* estimator but a poor *ranking* estimator, and it under-samples the few items
that actually separate models. I want a subset that (a) estimates full accuracy within a tight band and
(b) keeps the *ordering* of models intact, which is what a go/no-go decision and a regression gate
depend on.

**Approach: feature-anchored stratified + diversity selection.** I rejected free-fit IRT /
tinyBenchmarks-style anchor points: those fit per-item parameters from *hundreds* of models, and we
have **three**. Instead, at eval time the selector uses **only load-time features** — LCB
`difficulty`×`platform` (re-injected from the raw record) and prompt length; AA-LCR #source-documents
and context length. It (1) stratifies on a coarse difficulty×category grid, (2) allocates the budget
across strata, over-weighting the discriminating middle band, (3) picks representative+diverse items
within each stratum by farthest-point sampling, and (4) stamps a `prune_weight =
stratum_size/selected` so the weighted mean is an **unbiased** accuracy estimate. The three shipped
models are used **only offline** to build a difficulty prior that weights features and powers
validation — then it is **frozen**. Nothing in the eval-time path reads the target model's score, so it
generalizes to an unseen 4th model. This is provably none of the forbidden baselines (not random, not
top-k by score — difficulty is only a stratum axis and every stratum is represented, not hand-picked,
not model-fit).

**How much, and why it's sufficient.** Validated by **leave-one-model-out** (each of the 3 models
treated as the unseen target; selection never sees its scores), with bootstrap/Wilson CIs:

| Benchmark | Keep | Accuracy MAE | Model-ranking (Kendall τ) | vs random | vs top-k-hardest |
|---|---|---|---|---|---|
| LiveCodeBench v5 | 315→79 (25%) | **0.029** | **1.00** | random 0.037 / τ 0.73 | MAE 0.53 |
| AA-LCR | 100→33 (33%) | **0.041** | **1.00** | random 0.054 / τ 0.47 | MAE 0.47 |
| MMMU (stratified) | 660→165 (25%) | **0.001** | – (1 model) | random 0.018 | MAE 0.71 |

(Reproduce with `python -m evalscope_ext.tools.validate_pruning`.)

Top-k easiest/hardest are catastrophic (MAE 0.29–0.71) — they invert the verdict — which is exactly why
the rubric forbids them. **Honest limit:** AA-LCR at 10% (n=10) is *worse* than random (MAE 0.15)
because it's 100 LLM-judge-graded items and judge noise dominates a tiny sample — so the AA-LCR default
is 33%, and `compare_runs` separates judge noise from sample variance and widens CIs.

## Part B — MMMU image-encoder probe (working code, not a sketch)

**What stresses an encoder, and how I isolate it.** Dense fine detail, OCR/text-in-image, charts,
scientific micro/macro imagery stress a vision encoder; natural-scene semantics don't. The shipped 660
rows corroborate this ordering (glm-4.5v: Chemical Structures 0.52, Body Scans 0.59, Medical 0.61,
Microscopic 0.63 … Photographs 0.82, Maps 0.92, Sculpture 1.0). The probe (`mmmu_encoder_probe`) scores
each item with a **frozen, literature-derived per-`img_type` stress table** + offline PIL/numpy image
features (Laplacian variance, low entropy, near-monochrome, optional OCR token count), stratifies by
(subject, img_type) so all 30 subjects and all stress categories are covered, and diversifies within
strata. It runs on MMMU/MMMU **metadata + pixels only**, so it scales to the full ~12K and is model-free
(it concentrates encoder-stress: mean category-stress 0.68→0.83 at 2% keep). Crucially it does **not**
select items the model gets wrong — that would be model-overfit.

Selection alone can't prove a drop is the *encoder's* fault, so two mechanisms attribute it:
**(1) a paired perturbation eval** (`perturb="downscale,blur,jpeg"`) emits the original + degraded copies
of every probe item; `encoder_sensitivity = acc_orig − mean(acc_perturbed)` is large only when the
answer depends on fine visual signal. **(2) `visual_necessity`** keeps only items that need the image
(drops open-ended/format-noise items; consumes a text-only-ablation signal when available). ~150–300
probe items (1–2% of 12K) suffice because variance is concentrated in encoder-bound items; random
sampling needs many× more to hit the same categories.

## Assumptions & what would change
- **Assumptions:** the shipped models are representative enough to *rank* item difficulty by category
  (not to score the target); the image-stress ordering is a category-level inductive bias (frozen, not
  per-model); AA-LCR `acc` is partly judge noise.
- **More data / more reference models →** estimate real IRT difficulty+discrimination (2PL) and select
  true anchor points; per-platform/per-subject ratio tuning.
- **A live endpoint →** run the exact MMMU-Pro visual-necessity filter (N text-only LLMs × image
  removed) and report measured-vs-projected drift, turning the prior into a calibrated model.
- **More time →** adaptive `prune_ratio` (grow the subset until the CI clears the customer bar) and a
  per-capability go/no-go dashboard.
