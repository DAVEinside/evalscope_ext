# Handout B — Why this matters & how to use it

*For developers, test engineers, product, and the customer team.*

## What this changes for the customer conversation
Today, qualifying a model on a customer's capabilities (coding, long-context, and — next quarter —
multimodal) means running **full** benchmark suites: slow and expensive, so it happens rarely and late.
These pruners let a sales engineer get the **same go/no-go answer from ~10–25% of the items**. In our
offline checks the pruned set reproduced the full-suite accuracy within ~1–4 points **and kept the
model ranking identical**, so a "this model clears the bar" claim made on the small set holds on the
full set. Concretely: a long-context (AA-LCR) qualification that was 100 long-document runs becomes ~33;
a coding (LiveCodeBench) run drops from 315 problems to ~79. That turns a one-off, end-of-cycle eval
into something you can run on **every candidate model and every release** as a regression gate.

## How to run it tomorrow (inside evalscope)
```bash
# 1. full reference (once per model)
evalscope eval --model <model> --datasets live_code_bench --output ./results_full/
# 2. the cheap pruned run you'll actually repeat
evalscope eval --model <model> --datasets live_code_bench_pruned \
  --dataset-args '{"live_code_bench_pruned":{"extra_params":{"pruning_strategy":"stratified_diversity","prune_ratio":0.25}}}' \
  --output ./results_pruned/
# 3. the verdict, with confidence intervals and go/no-go agreement
python -m evalscope_ext.tools.compare_runs --full ./results_full/ --pruned ./results_pruned/
```
Swap `live_code_bench` for `aa_lcr` or `mmmu`; the same flags work. `compare_runs` prints full-vs-pruned
accuracy with 95% CIs, a CI-overlap flag, per-difficulty/per-image-type deltas, and whether the pruned
set lands the model on the **same side of your go/no-go bar**.

## What the multimodal probe gives that random sampling cannot
A customer adding image input needs to know if a model's **image encoder** is good enough — not whether
it can reason. Random sampling spreads thinly across easy photos and hard charts alike and would **miss
a weak encoder** that only shows up on dense, detailed images (tables, chemical structures, medical
scans, plots). Our probe deliberately concentrates on those encoder-stressing images and then
**degrades each one** (downscale / blur / compress) to measure how much accuracy depends on the
picture. A model whose accuracy collapses under mild degradation has a fragile encoder; one that holds
up is genuinely seeing the image. That targeted signal — "fragile vs robust image encoder" — is the
question a multimodal customer is really asking, and it costs ~1–2% of the full MMMU suite.

## Why a customer-facing PM should care
It makes "is this model good enough for *your* workload?" **cheap and repeatable**: a defensible
go/no-go you can re-run for every new model the customer asks about, with explicit confidence intervals
instead of a single number, and a forward-looking multimodal readiness check — without paying for the
full benchmark suite each time.
