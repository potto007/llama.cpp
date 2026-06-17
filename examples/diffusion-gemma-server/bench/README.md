# diffusion-gemma server benchmark

A spec-enforcing benchmark for `llama-diffusion-server`. It drives the server's
OpenAI-compatible `/v1/chat/completions` endpoint and verifies that the model produces
**correct output across a spread of prompt sizes and output sizes**, while reporting the
per-request prefill/denoise timing.

Unlike a plain throughput script, it asserts a size spec and exits non-zero if the spec is
not met, so it can gate changes (e.g. the Q8 KV-store work) in CI or a loop.

## What it enforces

Size buckets are half-open bands from 1000 tokens up: `1k=[1000,2000)`, `2k=[2000,4000)`,
`4k=[4000,8000)`, `8k=[8000,16000)`, ...

* **prompt-size coverage** - one passing case per bucket from 1k up to `--prompt-max`.
* **output-size coverage** - one passing case per bucket from 1k up to `--output-max`.
* **size landing** - each case's measured prompt/completion tokens must fall in its target band.
* **correctness** - every output must be non-degenerate; prompt-axis cases additionally must
  recall a secret planted before a long document (this also checks the long-context KV path
  returns correct information, not just plausible text).

`SPEC RESULT: PASS` (exit 0) iff every requested bucket on both axes is covered by a passing case.

## Timing

The server exposes per-request timing in the response (`timings`: `prefill_ms`, `denoise_ms`,
`ms_per_step`, `n_steps`, `n_blocks`), so the harness reads `denoise ms/step` directly instead of
scraping stderr. `denoise ms/step` is the stable metric to compare KV-store variants - it is
per-step, so it does not depend on how many tokens the model happens to emit.

## How sizes are hit

* **Prompt size** is fully controllable: a real-doc slice (from in-repo docs) is sized to the
  target and the harness re-measures and retries until the prompt lands in-band.
* **Output size** is steered with a single full-rewrite/expand task. Committed output grows
  monotonically with input length and is capped by `max_tokens`; small buckets size the input
  large enough to overfill a low cap (the cap then binds the size exactly), large buckets let the
  natural length land in-band. The model plateaus near ~9k committed tokens, so output buckets
  above that need a context far larger than the output and are intentionally out of the default
  `--output-max` (they are reported, never silently skipped).

## Usage

Start the server (Q8 KV store shown; drop `DG_KV_STORE` for the F16 baseline):

```bash
DG_KV_STORE=q8 ./build/bin/llama-diffusion-server \
  -m <diffusiongemma>.gguf -ngl 99 -fa -c 32768 --port 8088 -a dg
```

Run the benchmark:

```bash
python3 examples/diffusion-gemma-server/bench/diffusion_bench.py \
  --prompt-max 16000 --output-max 8000 --reps 3
```

Flags: `--url` (default `http://127.0.0.1:8088/v1/chat/completions`), `--reps`, `--warmup`,
`--prompt-max`, `--output-max`.

### Output modes

Progress lines go to stderr; the report goes to stdout (or to files). Pick one or more formats
with `--format` (comma-separated: `text`, `json`, `html`, `md`; default `text`) and an optional
`--out` path:

```bash
# text report to stdout (default)
... diffusion_bench.py

# one format to an exact file
... diffusion_bench.py --format html --out report.html

# several formats at once: --out is used as a stem, extensions are appended
# (writes report.txt, report.json, report.html, report.md)
... diffusion_bench.py --format text,json,html,md --out report

# no --out with multiple formats prints each to stdout, separated by markers
... diffusion_bench.py --format md,json
```

`--json PATH` is kept as a backward-compatible alias for `--format json --out PATH`.

### Comparing configurations

To A/B several KV-store configs, save one JSON report per config (run the same command against each
server build), then render them side by side with `--compare`. Each entry is `label=path`:

```bash
# one report per config
... --format json --out data/f16.json        # against the F16 server
DG_KV_STORE=q8 ... --format json --out data/q8-cache.json   # against the Q8 server
# ... etc

# render the comparison (no server run); --note adds facts the harness cannot measure
python3 diffusion_bench.py --compare \
  "Baseline=data/baseline.json" "F16=data/f16.json" \
  "Q8 no-cache=data/q8-nocache.json" "Q8+cache=data/q8-cache.json" \
  --note "Context ceiling: F16 32768 -> Q8 65536" \
  --format html,md --out RESULTS
```

`--compare` emits four tables (denoise ms/step by prompt size and by output size, effective output
tok/s, prefill tok/s), highlighting the best config per row. `RESULTS.html` / `RESULTS.md` in this
directory are generated this way from the saved reports in `data/`.
