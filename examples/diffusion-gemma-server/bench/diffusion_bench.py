#!/usr/bin/env python3
"""Diffusion-gemma server benchmark harness.

Drives the running llama-diffusion-server over its OpenAI-compatible /v1/chat/completions
endpoint and measures per-request prefill/denoise timing (read from the server's `timings`
response field, not scraped from stderr).

It ENFORCES a size spec rather than just reporting numbers:

  * prompt-token sizes must cover the buckets 1000 tok .. context-max
  * completion-token sizes must cover the buckets 1000 tok .. (model-reachable max)
  * each measured size must land inside its target bucket
  * each output must be non-degenerate; prompt-axis cases must recall a planted secret
    (which also verifies the long-context KV path returns correct information)

Buckets are half-open bands [lo, 2*lo): 1k=[1000,2000), 2k=[2000,4000), ... top is [lo, inf).

Levers (empirically calibrated for diffusion-gemma):
  * prompt size  - a real-doc slice is sized to the target bucket; the harness re-measures and
                   retries (auto-calibration) until the prompt lands in-bucket. Fully controllable
                   up to the server context.
  * output size  - committed output scales with input length and is capped by max_tokens. Small
                   buckets use a strong short-seed expansion capped by max_tokens; large buckets
                   expand a sized input document. The model plateaus near ~9k committed tokens, so
                   output buckets above that need a context far larger than the output (and are
                   reported as unreachable rather than silently skipped).

Exit code 0 iff every requested bucket on both axes is covered by a passing case.
See README.md for usage.
"""
import argparse
import json
import os
import statistics
import sys
import time
import urllib.request

# ----------------------------------------------------------------------------- size buckets
BUCKETS = [1000, 2000, 4000, 8000, 16000, 32000]


def bucket_band(lo):
    """Half-open [lo, hi) band for a bucket; the top bucket is open-ended."""
    hi = float("inf") if lo == BUCKETS[-1] else 2 * lo
    return lo, hi


def in_band(n, lo):
    a, b = bucket_band(lo)
    return a <= n < b


# ----------------------------------------------------------------------------- corpus (real repo docs)
REPO = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
CORPUS_DOCS = [
    "tools/server/README.md",
    "docs/build.md",
    "README.md",
    "docs/function-calling.md",
    "docs/development/HOWTO-add-model.md",
]


def load_corpus():
    parts = []
    for rel in CORPUS_DOCS:
        p = os.path.join(REPO, rel)
        if os.path.isfile(p):
            with open(p, encoding="utf-8", errors="replace") as f:
                parts.append(f.read())
    corpus = "\n\n".join(parts) or ("lorem ipsum dolor sit amet " * 20000)
    while len(corpus) < 32000 * 6:  # enough chars for the largest prompt bucket
        corpus += "\n\n" + corpus
    return corpus


CORPUS = load_corpus()


# ----------------------------------------------------------------------------- HTTP
def chat(url, text, max_tokens, seed=42, timeout=1800):
    body = json.dumps(
        {"messages": [{"role": "user", "content": text}], "seed": seed, "max_tokens": max_tokens}
    ).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        j = json.load(r)
    wall = time.time() - t0
    u = j["usage"]
    tm = j.get("timings", {})
    content = (j["choices"][0]["message"].get("content") or "")
    return {
        "content": content,
        "prompt_tok": u["prompt_tokens"],
        "compl_tok": u["completion_tokens"],
        "wall_s": wall,
        "prefill_ms": tm.get("prefill_ms", 0.0),
        "denoise_ms": tm.get("denoise_ms", 0.0),
        "ms_per_step": tm.get("ms_per_step", 0.0),
        "n_steps": tm.get("n_steps", 0),
        "n_blocks": tm.get("n_blocks", 0),
    }


# ----------------------------------------------------------------------------- prompt builders
SECRET = "ACCESS-CODE-7Z4Q"


def doc_slice(approx_tokens, ratio):
    n_chars = max(1, int(approx_tokens * ratio))
    return CORPUS[:n_chars]


def build_prompt_axis(approx_prompt_tok, ratio):
    """Plant a secret, then a long document; ask only to recall the secret.

    Recall is reliable at every prompt length (summarization is not - this model often emits an
    empty canvas on summarize), so this isolates the PROMPT-size axis and doubles as a long-context
    correctness check: the secret sits before the whole document and must be retrieved across it.
    """
    doc = doc_slice(max(1, approx_prompt_tok - 60), ratio)
    return (
        f"The access code is {SECRET}. Remember it.\n\n"
        "--- DOCUMENT ---\n" + doc +
        f"\n--- END DOCUMENT ---\n\nIgnore the document. Repeat the access code exactly, and nothing else."
    )


# Committed output is a monotone function of input length for a full-rewrite/expand task, capped by
# max_tokens. Measured (input_tok -> committed_tok): 3000->4231, 5000->5780, 6000->7498, 10000->8865;
# it plateaus near ~9k. So every output bucket uses the same expand-a-sized-input task: small buckets
# size the input large enough to overfill a low max_tokens cap (the cap then binds the size exactly);
# large buckets let the natural length land in-band under a generous cap. (input_tok, max_tokens):
OUTPUT_PLAN = {
    1000:  (2500, 1536),    # model wants ~3.5k, capped to 1536  -> [1000,2000)
    2000:  (3000, 3072),    # model wants ~4.2k, capped to 3072  -> [2000,4000)
    4000:  (5000, 8000),    # natural ~5.8k under cap            -> [4000,8000)
    8000:  (11000, 16000),  # natural ~8.9k under cap            -> [8000,16000)
    16000: (24000, 32768),  # plateaus ~9k: not reachable at ctx 32k (needs ctx >> output)
    32000: (44000, 32768),  # not reachable
}


def build_output_axis(target, ratio):
    """Return (prompt, max_tokens): expand a sized input document so committed output lands in band."""
    in_tok, cap = OUTPUT_PLAN.get(target, (target, 32768))
    doc = doc_slice(in_tok, ratio)
    prompt = (
        "Rewrite the following documentation in full, expanding EVERY sentence into a detailed "
        "paragraph with concrete examples and rationale. Preserve all facts and the section order. "
        "Do not summarize and do not stop early.\n\n--- DOCUMENT ---\n" + doc
    )
    return prompt, cap


# ----------------------------------------------------------------------------- correctness
def degenerate(content):
    words = content.split()
    if len(words) < 5:
        return True
    uniq = len(set(w.lower() for w in words)) / len(words)
    return uniq < 0.12


def check_correct(spec, content):
    if spec["axis"] == "prompt":
        if SECRET not in content:
            return False, f"secret not recalled (got {content[:40]!r})"
        return True, "secret recalled"
    if degenerate(content):
        return False, "degenerate/empty output"
    return True, "coherent"


# ----------------------------------------------------------------------------- runner
def median_std(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return 0.0, 0.0
    return statistics.median(xs), (statistics.pstdev(xs) if len(xs) > 1 else 0.0)


def measure_ratio(url):
    probe = CORPUS[:8000]
    r = chat(url, "Repeat this word: benchmark\n\n" + probe + "\n\nRepeat: benchmark", max_tokens=16)
    return len(probe) / max(1, r["prompt_tok"])


def run_spec(url, spec, ratio, reps, warmup):
    target = spec["bucket"]
    if spec["axis"] == "prompt":
        approx = int(target * 1.4)
        prompt = build_prompt_axis(approx, ratio)
        max_tokens = 64
        for _ in range(4):  # auto-calibrate prompt length into the band
            probe = chat(url, prompt, max_tokens=max_tokens)
            if in_band(probe["prompt_tok"], target):
                break
            lo, hi = bucket_band(target)
            mid = lo * 1.4 if hi == float("inf") else (lo + hi) / 2
            approx = max(1, int(approx * mid / max(1, probe["prompt_tok"])))
            prompt = build_prompt_axis(approx, ratio)
    else:
        prompt, max_tokens = build_output_axis(target, ratio)

    for _ in range(warmup):
        chat(url, prompt, max_tokens=max_tokens)
    runs = [chat(url, prompt, max_tokens=max_tokens) for _ in range(reps)]

    last = runs[-1]
    ms_med, ms_std = median_std([r["ms_per_step"] for r in runs])
    pf_med, _ = median_std([r["prefill_ms"] for r in runs])
    prompt_tok = int(statistics.median([r["prompt_tok"] for r in runs]))
    compl_tok = int(statistics.median([r["compl_tok"] for r in runs]))

    measured = prompt_tok if spec["axis"] == "prompt" else compl_tok
    correct, why = check_correct(spec, last["content"])
    return {
        "name": spec["name"], "axis": spec["axis"], "bucket": target,
        "prompt_tok": prompt_tok, "compl_tok": compl_tok,
        "size_ok": in_band(measured, target), "correct": correct, "why": why,
        "ms_per_step": ms_med, "ms_std": ms_std, "prefill_ms": pf_med,
        "n_steps": last["n_steps"], "n_blocks": last["n_blocks"],
    }


def make_specs(prompt_max, output_max):
    specs = []
    for lo in BUCKETS:
        if lo <= prompt_max:
            specs.append({"name": f"prompt-{lo // 1000}k", "axis": "prompt", "bucket": lo})
    for lo in BUCKETS:
        if lo <= output_max:
            specs.append({"name": f"output-{lo // 1000}k", "axis": "output", "bucket": lo})
    return specs


def main():
    ap = argparse.ArgumentParser(description="Diffusion-gemma server benchmark (spec-enforcing).")
    ap.add_argument("--url", default="http://127.0.0.1:8088/v1/chat/completions")
    ap.add_argument("--reps", type=int, default=3, help="measured repetitions per spec")
    ap.add_argument("--warmup", type=int, default=1, help="unmeasured warmup runs per spec")
    ap.add_argument("--prompt-max", type=int, default=16000, help="largest prompt bucket (<= server ctx)")
    ap.add_argument("--output-max", type=int, default=8000, help="largest output bucket (model-reachable)")
    ap.add_argument("--json", default="", help="optional path to write the JSON report")
    args = ap.parse_args()

    print(f"calibrating tokenizer ratio against {args.url} ...", flush=True)
    ratio = measure_ratio(args.url)
    print(f"  chars/token ~= {ratio:.2f}\n", flush=True)

    specs = make_specs(args.prompt_max, args.output_max)
    results = []
    for spec in specs:
        print(f"running {spec['name']:<12} ...", end=" ", flush=True)
        try:
            res = run_spec(args.url, spec, ratio, args.reps, args.warmup)
        except Exception as e:  # noqa: BLE001
            print(f"ERROR {e}")
            results.append({"name": spec["name"], "axis": spec["axis"], "bucket": spec["bucket"],
                            "error": str(e), "size_ok": False, "correct": False})
            continue
        print(f"prompt={res['prompt_tok']:>6} compl={res['compl_tok']:>6} "
              f"denoise={res['ms_per_step']:.1f}ms/step  "
              f"size={'OK' if res['size_ok'] else 'MISS'} corr={'OK' if res['correct'] else 'FAIL'}"
              f"{'' if res['correct'] else '  (' + res.get('why', '') + ')'}")
        results.append(res)

    print("\n" + "=" * 94)
    print(f"{'spec':<12}{'axis':<8}{'bucket':>8}{'prompt_tok':>12}{'compl_tok':>11}"
          f"{'ms/step':>11}{'size':>7}{'correct':>9}")
    print("-" * 94)
    for r in results:
        if "error" in r:
            print(f"{r['name']:<12}{r['axis']:<8}{r['bucket']:>8}   ERROR: {r['error']}")
            continue
        print(f"{r['name']:<12}{r['axis']:<8}{r['bucket']:>8}{r['prompt_tok']:>12}{r['compl_tok']:>11}"
              f"{r['ms_per_step']:>8.1f}±{r['ms_std']:<2.0f}{'OK' if r['size_ok'] else 'MISS':>5}"
              f"{'OK' if r['correct'] else 'FAIL':>9}")

    def covered(axis, mx):
        need = [b for b in BUCKETS if b <= mx]
        got = {r["bucket"] for r in results
               if r.get("axis") == axis and r.get("size_ok") and r.get("correct")}
        return [b for b in need if b not in got]

    miss_prompt = covered("prompt", args.prompt_max)
    miss_output = covered("output", args.output_max)
    all_correct = all(r.get("correct") for r in results)
    print("\n" + "=" * 94)
    print(f"prompt-size coverage 1k..{args.prompt_max // 1000}k : "
          f"{'COMPLETE' if not miss_prompt else 'MISSING ' + str([b // 1000 for b in miss_prompt]) + 'k'}")
    print(f"output-size coverage 1k..{args.output_max // 1000}k : "
          f"{'COMPLETE' if not miss_output else 'MISSING ' + str([b // 1000 for b in miss_output]) + 'k'}")
    print(f"output correctness                : {'ALL PASS' if all_correct else 'FAILURES'}")
    ok = not miss_prompt and not miss_output and all_correct
    print(f"\nSPEC RESULT: {'PASS' if ok else 'FAIL'}")

    if args.json:
        with open(args.json, "w") as f:
            json.dump({"ratio": ratio, "results": results, "missing_prompt": miss_prompt,
                       "missing_output": miss_output, "pass": ok}, f, indent=2)
        print(f"wrote {args.json}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
