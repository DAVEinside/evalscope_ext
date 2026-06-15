# evalscope_ext — universal benchmark pruning

An extension to [`modelscope/evalscope`](https://github.com/modelscope/evalscope) that prunes
any benchmark to the **smallest principled subset** that still answers *"is this model good enough?"* —
for a sales go/no-go signal and engineering regression testing.

**Developed against evalscope commit** `c14dbaf94e9129f7054ad4a184c2ff0cae2e6a5d` (pin this SHA; the
framework API is still evolving).

## Why it's defensible (not a forbidden baseline)
Selection at eval time uses **only load-time features** — difficulty/platform (LCB), #source-docs and
context length (AA-LCR), image-type + offline image features (MMMU) — plus a frozen,
literature-derived image-stress table. The 3 shipped models build an offline difficulty prior that
**only weights features and powers validation, then is frozen** — it never selects by the target
model's score. So the pruner generalizes to an unseen 4th model and is provably **not** uniform-random,
**not** top-k easiest/hardest, **not** hand-picked, **not** overfit to the shipped models.

## Install (from a clean clone of the fork)
```bash
cd task2-evalscope
python -m venv .venv && . .venv/Scripts/activate      # (Linux/Mac: source .venv/bin/activate)
pip install -e .                                       # installs evalscope + evalscope_ext
```

## Run contract
The pruned benchmarks (`live_code_bench_pruned`, `aa_lcr_pruned`, `mmmu_pruned`) are auto-discovered by
evalscope. Custom args flow through evalscope's `extra_params`, so the JSON is **keyed by the dataset
name** with an `extra_params` sub-dict (the brief's flat form is shorthand):

```bash
# full run
evalscope eval --model <m> --datasets live_code_bench --output ./results_full/

# pruned run (keep 25%, universal stratified+diversity strategy)
evalscope eval --model <m> --datasets live_code_bench_pruned \
  --dataset-args '{"live_code_bench_pruned":{"extra_params":{"pruning_strategy":"stratified_diversity","prune_ratio":0.25}}}' \
  --output ./results_pruned/

# compare
python -m evalscope_ext.tools.compare_runs --full ./results_full/ --pruned ./results_pruned/
```
A tiny smoke run: add `--limit 10` and use a cheap endpoint (Cerebras free tier serves gpt-oss-120b;
or OpenAI `gpt-4o-mini`), plus `--judge-model` for the LLM-judged benchmarks.

### Part B — MMMU image-encoder probe (working code)
```bash
# select an encoder-stress probe AND expand each item with degraded copies
evalscope eval --model <vlm> --datasets mmmu_pruned \
  --dataset-args '{"mmmu_pruned":{"extra_params":{"pruning_strategy":"mmmu_encoder_probe","prune_ratio":0.02,"perturb":"downscale,blur,jpeg"}}}' \
  --output ./results_probe/
python -m evalscope_ext.tools.compare_runs --pruned ./results_probe/ --perturb
```

## Verify without a model (offline, free)
```bash
python -m evalscope_ext.tools.build_item_stats        # shipped Evals -> data/item_stats.json
python -m evalscope_ext.tools.validate_pruning        # leave-one-model-out: beats random & top-k
python -m evalscope_ext.tools.make_demo_runs --bench live_code_bench --ratio 0.25   # fake full/pruned runs
python -m evalscope_ext.tools.compare_runs --full ./demo_runs/results_full --pruned ./demo_runs/results_pruned
python -m evalscope_ext.tests.test_pruning            # unit tests
```

## Strategies (`evalscope_ext/pruning/list_strategies()`)
- **`stratified_diversity`** — primary universal: stratify by a difficulty×category grid, allocate over
  the discriminating middle band, diversify within strata by farthest-point sampling; stamps a
  `prune_weight` for unbiased accuracy estimates. Use for LCB / AA-LCR / MMMU go/no-go.
- **`mmmu_encoder_probe`** — Part B: concentrate on encoder-stressing image categories + high-detail
  images, covering all subjects. Pair with the perturbation eval to *measure* encoder sensitivity.
- **`visual_necessity`** — Part B: keep only items that need the image (drops open-ended; consumes an
  optional text-only-ablation signal), then run the probe.

## Layout
```
evalscope_ext/
  pruning/{registry,base,selection,mixin,registration}.py
  pruning/strategies/{stratified_diversity,mmmu_encoder_probe,visual_necessity}.py
  features/image_features.py        # PIL+numpy core; optional pytesseract/CLIP behind find_spec
  perturb/{image_ops,perturb_mixin}.py
  data/{item_stats.json,img_type_stress.json}
  tools/{build_item_stats,validate_pruning,compare_runs,make_demo_runs}.py
  tests/test_pruning.py
evalscope/benchmarks/{live_code_bench,aa_lcr,mmmu}_pruned/   # thin auto-discovered shims
```
Optional extras (`pytesseract`, `open_clip`, `scikit-learn`) are gated behind `importlib.util.find_spec`
with PIL+numpy fallbacks, so the default install stays light and the benchmarks always register.
