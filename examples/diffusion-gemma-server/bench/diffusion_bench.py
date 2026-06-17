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
        "total_ms": tm.get("total_ms", 0.0),
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
    dn_med, _ = median_std([r["denoise_ms"] for r in runs])
    tot_med, _ = median_std([r["total_ms"] for r in runs])
    prompt_tok = int(statistics.median([r["prompt_tok"] for r in runs]))
    compl_tok = int(statistics.median([r["compl_tok"] for r in runs]))
    n_steps = int(statistics.median([r["n_steps"] for r in runs]))
    n_blocks = int(statistics.median([r["n_blocks"] for r in runs]))

    # derived metrics
    eff_tok_s = compl_tok / (tot_med / 1000) if tot_med > 0 else 0.0   # effective output tok/s (end-to-end)
    tok_per_step = compl_tok / n_steps if n_steps > 0 else 0.0          # diffusion efficiency
    fill = compl_tok / (n_blocks * 256) if n_blocks > 0 else 0.0        # committed / canvas
    prefill_tok_s = prompt_tok / (pf_med / 1000) if pf_med > 0 else 0.0

    measured = prompt_tok if spec["axis"] == "prompt" else compl_tok
    correct, why = check_correct(spec, last["content"])
    return {
        "name": spec["name"], "axis": spec["axis"], "bucket": target,
        "prompt_tok": prompt_tok, "compl_tok": compl_tok,
        "size_ok": in_band(measured, target), "correct": correct, "why": why,
        "ms_per_step": ms_med, "ms_std": ms_std,
        "prefill_ms": pf_med, "denoise_ms": dn_med, "total_ms": tot_med,
        "n_steps": n_steps, "n_blocks": n_blocks,
        "eff_tok_s": eff_tok_s, "tok_per_step": tok_per_step, "fill": fill,
        "prefill_tok_s": prefill_tok_s,
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


# --------------------------------------------------------------------------- report assembly
def build_coverage(results, prompt_max, output_max):
    def covered(axis, mx):
        need = [b for b in BUCKETS if b <= mx]
        got = {r["bucket"] for r in results
               if r.get("axis") == axis and r.get("size_ok") and r.get("correct")}
        return [b for b in need if b not in got]
    miss_prompt = covered("prompt", prompt_max)
    miss_output = covered("output", output_max)
    all_correct = all(r.get("correct") for r in results)
    return {
        "missing_prompt": miss_prompt, "missing_output": miss_output,
        "all_correct": all_correct,
        "pass": not miss_prompt and not miss_output and all_correct,
    }


def _cov_str(missing, mx):
    return "COMPLETE" if not missing else "MISSING " + str([b // 1000 for b in missing]) + "k"


# --------------------------------------------------------------------------- renderers
def render_json(report):
    return json.dumps(report, indent=2)


def render_text(report):
    m, results, cov = report["meta"], report["results"], report["coverage"]
    out = []
    out.append(f"DiffusionGemma benchmark  ({m['url']}, reps={m['reps']}, chars/tok~={m['ratio']:.2f})")
    out.append("=" * 94)
    out.append(f"{'spec':<12}{'axis':<8}{'bucket':>8}{'prompt_tok':>12}{'compl_tok':>11}"
               f"{'ms/step':>11}{'size':>7}{'correct':>9}")
    out.append("-" * 94)
    for r in results:
        if "error" in r:
            out.append(f"{r['name']:<12}{r['axis']:<8}{r['bucket']:>8}   ERROR: {r['error']}")
            continue
        out.append(f"{r['name']:<12}{r['axis']:<8}{r['bucket']:>8}{r['prompt_tok']:>12}{r['compl_tok']:>11}"
                   f"{r['ms_per_step']:>8.1f}±{r['ms_std']:<2.0f}{'OK' if r['size_ok'] else 'MISS':>5}"
                   f"{'OK' if r['correct'] else 'FAIL':>9}")
    out.append("\n" + "=" * 94)
    out.append("PERFORMANCE METRICS")
    out.append(f"{'spec':<12}{'prompt':>8}{'compl':>7}{'steps':>6}{'fill':>6}"
               f"{'ms/step':>9}{'tok/step':>9}{'prefill t/s':>12}{'eff tok/s':>10}")
    out.append("-" * 94)
    for r in results:
        if "error" in r:
            continue
        out.append(f"{r['name']:<12}{r['prompt_tok']:>8}{r['compl_tok']:>7}{r['n_steps']:>6}"
                   f"{r['fill']:>6.2f}{r['ms_per_step']:>9.1f}{r['tok_per_step']:>9.1f}"
                   f"{r['prefill_tok_s']:>12.0f}{r['eff_tok_s']:>10.1f}")
    out.append("\n" + "=" * 94)
    out.append(f"prompt-size coverage 1k..{m['prompt_max'] // 1000}k : {_cov_str(cov['missing_prompt'], m['prompt_max'])}")
    out.append(f"output-size coverage 1k..{m['output_max'] // 1000}k : {_cov_str(cov['missing_output'], m['output_max'])}")
    out.append(f"output correctness                : {'ALL PASS' if cov['all_correct'] else 'FAILURES'}")
    out.append(f"\nSPEC RESULT: {'PASS' if cov['pass'] else 'FAIL'}")
    return "\n".join(out) + "\n"


def render_md(report):
    m, results, cov = report["meta"], report["results"], report["coverage"]
    out = ["# DiffusionGemma benchmark report", ""]
    out.append(f"`{m['url']}` &middot; reps={m['reps']} &middot; chars/token ~= {m['ratio']:.2f}".replace("&middot;", "|"))
    out += ["", "## Spec", "",
            "| spec | axis | bucket | prompt_tok | compl_tok | ms/step | size | correct |",
            "|---|---|---:|---:|---:|---:|:--:|:--:|"]
    for r in results:
        if "error" in r:
            out.append(f"| {r['name']} | {r['axis']} | {r['bucket']} | | | | | ERROR: {r['error']} |")
            continue
        out.append(f"| {r['name']} | {r['axis']} | {r['bucket']} | {r['prompt_tok']} | {r['compl_tok']} | "
                   f"{r['ms_per_step']:.1f}±{r['ms_std']:.0f} | {'OK' if r['size_ok'] else 'MISS'} | "
                   f"{'OK' if r['correct'] else 'FAIL'} |")
    out += ["", "## Performance metrics", "",
            "| spec | prompt | compl | steps | fill | ms/step | tok/step | prefill t/s | eff tok/s |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in results:
        if "error" in r:
            continue
        out.append(f"| {r['name']} | {r['prompt_tok']} | {r['compl_tok']} | {r['n_steps']} | {r['fill']:.2f} | "
                   f"{r['ms_per_step']:.1f} | {r['tok_per_step']:.1f} | {r['prefill_tok_s']:.0f} | "
                   f"{r['eff_tok_s']:.1f} |")
    out += ["", "## Coverage", "",
            f"- prompt-size 1k..{m['prompt_max'] // 1000}k: {_cov_str(cov['missing_prompt'], m['prompt_max'])}",
            f"- output-size 1k..{m['output_max'] // 1000}k: {_cov_str(cov['missing_output'], m['output_max'])}",
            f"- output correctness: {'ALL PASS' if cov['all_correct'] else 'FAILURES'}",
            "", f"**SPEC RESULT: {'PASS' if cov['pass'] else 'FAIL'}**", ""]
    return "\n".join(out)


def _h(s):  # minimal HTML escape
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def render_html(report):
    m, results, cov = report["meta"], report["results"], report["coverage"]
    spec_rows = []
    for r in results:
        if "error" in r:
            spec_rows.append(f'<tr><td>{_h(r["name"])}</td><td>{_h(r["axis"])}</td><td>{r["bucket"]}</td>'
                             f'<td colspan="5" class="bad">ERROR: {_h(r["error"])}</td></tr>')
            continue
        sz = "OK" if r["size_ok"] else "MISS"
        co = "OK" if r["correct"] else "FAIL"
        spec_rows.append(
            f'<tr><td>{_h(r["name"])}</td><td>{_h(r["axis"])}</td><td>{r["bucket"]}</td>'
            f'<td>{r["prompt_tok"]}</td><td>{r["compl_tok"]}</td>'
            f'<td>{r["ms_per_step"]:.1f}&plusmn;{r["ms_std"]:.0f}</td>'
            f'<td class="{"" if r["size_ok"] else "bad"}">{sz}</td>'
            f'<td class="{"" if r["correct"] else "bad"}">{co}</td></tr>')
    perf_rows = []
    for r in results:
        if "error" in r:
            continue
        perf_rows.append(
            f'<tr><td>{_h(r["name"])}</td><td>{r["prompt_tok"]}</td><td>{r["compl_tok"]}</td>'
            f'<td>{r["n_steps"]}</td><td>{r["fill"]:.2f}</td><td>{r["ms_per_step"]:.1f}</td>'
            f'<td>{r["tok_per_step"]:.1f}</td><td>{r["prefill_tok_s"]:.0f}</td>'
            f'<td>{r["eff_tok_s"]:.1f}</td></tr>')
    verdict = "PASS" if cov["pass"] else "FAIL"
    vclass = "best" if cov["pass"] else "bad"
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>DiffusionGemma benchmark</title>
<style>
  :root {{ --fg:#1a1a1a; --mut:#666; --line:#e2e2e2; --best:#2a6; --bad:#c33; --hl:#f6f8fa; }}
  body {{ font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
         color:var(--fg); max-width:920px; margin:2.5rem auto; padding:0 1.2rem; }}
  h1 {{ font-size:1.5rem; margin:0 0 .2rem; }}
  h2 {{ font-size:1.1rem; margin:1.8rem 0 .5rem; border-bottom:2px solid var(--line); padding-bottom:.3rem; }}
  .sub {{ color:var(--mut); margin:0 0 1rem; font-size:.9rem; }}
  table {{ border-collapse:collapse; width:100%; margin:.3rem 0 1rem; font-variant-numeric:tabular-nums; }}
  th,td {{ padding:.35rem .55rem; text-align:right; border-bottom:1px solid var(--line); }}
  th:first-child, td:first-child {{ text-align:left; }}
  thead th {{ border-bottom:2px solid #ccc; }}
  tbody tr:hover {{ background:var(--hl); }}
  .best {{ color:var(--best); font-weight:600; }}
  .bad {{ color:var(--bad); font-weight:600; }}
  .verdict {{ font-size:1.1rem; font-weight:700; margin-top:1rem; }}
</style></head><body>
<h1>DiffusionGemma benchmark</h1>
<p class="sub">{_h(m['url'])} &middot; reps={m['reps']} &middot; chars/token ~= {m['ratio']:.2f}</p>
<h2>Spec</h2>
<table><thead><tr><th>spec</th><th>axis</th><th>bucket</th><th>prompt_tok</th><th>compl_tok</th>
<th>ms/step</th><th>size</th><th>correct</th></tr></thead>
<tbody>{''.join(spec_rows)}</tbody></table>
<h2>Performance metrics</h2>
<table><thead><tr><th>spec</th><th>prompt</th><th>compl</th><th>steps</th><th>fill</th>
<th>ms/step</th><th>tok/step</th><th>prefill t/s</th><th>eff tok/s</th></tr></thead>
<tbody>{''.join(perf_rows)}</tbody></table>
<h2>Coverage</h2>
<ul>
<li>prompt-size 1k..{m['prompt_max'] // 1000}k: {_cov_str(cov['missing_prompt'], m['prompt_max'])}</li>
<li>output-size 1k..{m['output_max'] // 1000}k: {_cov_str(cov['missing_output'], m['output_max'])}</li>
<li>output correctness: {'ALL PASS' if cov['all_correct'] else 'FAILURES'}</li>
</ul>
<p class="verdict {vclass}">SPEC RESULT: {verdict}</p>
</body></html>
"""


RENDERERS = {"text": render_text, "json": render_json, "html": render_html, "md": render_md}
EXT = {"text": ".txt", "json": ".json", "html": ".html", "md": ".md"}


def write_outputs(report, formats, out, renderers=RENDERERS):
    """Render `report` in each format. With --out: single format -> that path, multiple -> stem+ext.
    Without --out: print each format to stdout."""
    import os.path as _p
    stem = out
    if out and len(formats) > 1:
        root, ext = _p.splitext(out)
        stem = root if ext.lower() in (".txt", ".json", ".html", ".md") else out
    for fmt in formats:
        body = renderers[fmt](report)
        if not out:
            if len(formats) > 1:
                print(f"\n===== {fmt} =====")
            print(body)
        else:
            path = out if len(formats) == 1 else stem + EXT[fmt]
            with open(path, "w", encoding="utf-8") as f:
                f.write(body if body.endswith("\n") else body + "\n")
            print(f"wrote {fmt} -> {path}", file=sys.stderr)


# --------------------------------------------------------------------------- comparison mode
# Render a side-by-side comparison of several saved JSON reports (one column per config). The harness
# measures per-step latency / throughput; it does NOT measure VRAM or the auto-sized context ceiling, so
# those are out of scope here (annotate them separately if needed).
def parse_compare_entry(s):
    """`label=path` or just `path` (label derived from the filename, stripping a leading report_)."""
    if "=" in s and not os.path.exists(s):
        label, path = s.split("=", 1)
        return label.strip(), path.strip()
    base = os.path.basename(s)
    for pre in ("report_", "report-"):
        if base.startswith(pre):
            base = base[len(pre):]
    return os.path.splitext(base)[0], s


def build_comparison(entries, title="DiffusionGemma benchmark comparison", notes=None):
    configs, data, seen = [], {}, {}
    for ent in entries:
        label, path = parse_compare_entry(ent)
        with open(path, encoding="utf-8") as f:
            results = json.load(f).get("results", [])
        by_name = {}
        for r in results:
            if "error" in r or "ms_per_step" not in r:
                continue
            by_name[r["name"]] = r
            size = r["prompt_tok"] if r["axis"] == "prompt" else r["compl_tok"]
            seen.setdefault(r["name"], {"name": r["name"], "bucket": r["bucket"], "axis": r["axis"], "size": size})
        configs.append(label)
        data[label] = by_name
    prompt_specs = sorted([v for v in seen.values() if v["axis"] == "prompt"], key=lambda x: x["bucket"])
    output_specs = sorted([v for v in seen.values() if v["axis"] == "output"], key=lambda x: x["bucket"])
    return {"title": title, "configs": configs, "data": data,
            "prompt_specs": prompt_specs, "output_specs": output_specs, "notes": notes or []}


def _ccell(comp, label, spec_name, metric, fmt):
    r = comp["data"][label].get(spec_name)
    if not r or r.get(metric) is None:
        return "-"
    return fmt.format(r[metric])


def _row_label(sp):
    # prompt size is ~identical across configs (deterministic prompts); committed output is not, so label
    # output rows by their bucket target instead of one config's measured length.
    return f"{sp['size']:,}" if sp["axis"] == "prompt" else f"{sp['bucket'] // 1000}k"


def _col0(specs_key):
    return "prompt tok" if "prompt" in specs_key else "out bkt"


# each table: (heading, spec-list-key, metric, value-format, lower_is_better)
_CMP_TABLES = [
    ("Denoise ms/step - prompt axis (latency vs context)", "prompt_specs", "ms_per_step", "{:.1f}", True),
    ("Denoise ms/step - output axis (generation-bound)", "output_specs", "ms_per_step", "{:.1f}", True),
    ("Effective output tok/s - output axis", "output_specs", "eff_tok_s", "{:.1f}", False),
    ("Prefill tok/s - prompt axis", "prompt_specs", "prefill_tok_s", "{:.0f}", True),
]


def render_compare_json(comp):
    return json.dumps(comp, indent=2)


def render_compare_text(comp):
    out = [comp["title"], f"configs: {', '.join(comp['configs'])}", ""]
    w = max(12, *(len(c) for c in comp["configs"]))
    for heading, specs_key, metric, fmt, _ in _CMP_TABLES:
        specs = comp[specs_key]
        if not specs:
            continue
        out.append(heading)
        out.append(f"{_col0(specs_key):>9}" + "".join(f"{c:>{w + 2}}" for c in comp["configs"]))
        for sp in specs:
            row = f"{_row_label(sp):>9}" + "".join(
                f"{_ccell(comp, c, sp['name'], metric, fmt):>{w + 2}}" for c in comp["configs"])
            out.append(row)
        out.append("")
    if comp.get("notes"):
        out += ["Notes:"] + [f"  - {n}" for n in comp["notes"]] + [""]
    return "\n".join(out)


def render_compare_md(comp):
    out = [f"# {comp['title']}", "", f"Configs: {', '.join('`' + c + '`' for c in comp['configs'])}", ""]
    for heading, specs_key, metric, fmt, _ in _CMP_TABLES:
        specs = comp[specs_key]
        if not specs:
            continue
        out += [f"## {heading}", "", f"| {_col0(specs_key)} | " + " | ".join(comp["configs"]) + " |",
                "|---:|" + "|".join(["---:"] * len(comp["configs"])) + "|"]
        for sp in specs:
            cells = " | ".join(_ccell(comp, c, sp["name"], metric, fmt) for c in comp["configs"])
            out.append(f"| {_row_label(sp)} | {cells} |")
        out.append("")
    if comp.get("notes"):
        out += ["## Notes", ""] + [f"- {n}" for n in comp["notes"]] + [""]
    return "\n".join(out)


def render_compare_html(comp):
    def best_idx(sp, metric, lower):
        vals = []
        for c in comp["configs"]:
            r = comp["data"][c].get(sp["name"])
            vals.append(r[metric] if r and r.get(metric) is not None else None)
        present = [(i, v) for i, v in enumerate(vals) if v is not None]
        if not present:
            return -1
        return (min if lower else max)(present, key=lambda t: t[1])[0]

    sections = []
    for heading, specs_key, metric, fmt, lower in _CMP_TABLES:
        specs = comp[specs_key]
        if not specs:
            continue
        head = "".join(f"<th>{_h(c)}</th>" for c in comp["configs"])
        rows = []
        for sp in specs:
            bi = best_idx(sp, metric, lower)
            cells = []
            for i, c in enumerate(comp["configs"]):
                val = _ccell(comp, c, sp["name"], metric, fmt)
                cells.append(f'<td class="{"best" if i == bi and val != "-" else ""}">{val}</td>')
            rows.append(f'<tr><td>{_row_label(sp)}</td>{"".join(cells)}</tr>')
        sections.append(f'<h2>{_h(heading)}</h2><table><thead><tr><th>{_col0(specs_key)}</th>{head}</tr></thead>'
                        f'<tbody>{"".join(rows)}</tbody></table>')
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_h(comp['title'])}</title>
<style>
  :root {{ --fg:#1a1a1a; --mut:#666; --line:#e2e2e2; --best:#2a6; --hl:#f6f8fa; }}
  body {{ font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
         color:var(--fg); max-width:920px; margin:2.5rem auto; padding:0 1.2rem; }}
  h1 {{ font-size:1.5rem; margin:0 0 .2rem; }}
  h2 {{ font-size:1.1rem; margin:1.8rem 0 .5rem; border-bottom:2px solid var(--line); padding-bottom:.3rem; }}
  .sub {{ color:var(--mut); margin:0 0 1rem; font-size:.9rem; }}
  table {{ border-collapse:collapse; width:100%; margin:.3rem 0 1rem; font-variant-numeric:tabular-nums; }}
  th,td {{ padding:.35rem .6rem; text-align:right; border-bottom:1px solid var(--line); }}
  th:first-child, td:first-child {{ text-align:left; }}
  thead th {{ border-bottom:2px solid #ccc; }}
  tbody tr:hover {{ background:var(--hl); }}
  .best {{ color:var(--best); font-weight:600; }}
  .note {{ color:var(--mut); font-size:.85rem; }}
</style></head><body>
<h1>{_h(comp['title'])}</h1>
<p class="sub">Configs: {' &middot; '.join('<b>' + _h(c) + '</b>' for c in comp['configs'])}.
Lower ms/step and higher tok/s are better (best in each row highlighted).</p>
{''.join(sections)}
{('<h2>Notes</h2><ul>' + ''.join('<li>' + _h(n) + '</li>' for n in comp['notes']) + '</ul>') if comp.get('notes') else ''}
<p class="note">Generated by diffusion_bench.py --compare. Latency and throughput are measured by the harness;
any VRAM or context-ceiling figures appear under Notes and are supplied manually.</p>
</body></html>
"""


COMPARE_RENDERERS = {"text": render_compare_text, "json": render_compare_json,
                     "html": render_compare_html, "md": render_compare_md}


def main():
    ap = argparse.ArgumentParser(description="Diffusion-gemma server benchmark (spec-enforcing).")
    ap.add_argument("--url", default="http://127.0.0.1:8088/v1/chat/completions")
    ap.add_argument("--reps", type=int, default=3, help="measured repetitions per spec")
    ap.add_argument("--warmup", type=int, default=1, help="unmeasured warmup runs per spec")
    ap.add_argument("--prompt-max", type=int, default=16000, help="largest prompt bucket (<= server ctx)")
    ap.add_argument("--output-max", type=int, default=8000, help="largest output bucket (model-reachable)")
    ap.add_argument("--format", default="text",
                    help="comma-separated output formats: text,json,html,md (default text)")
    ap.add_argument("--out", default="",
                    help="output path; single format -> this file, multiple -> used as a stem (ext appended); "
                         "omit to print to stdout")
    ap.add_argument("--json", default="", help="(compat) write JSON to this path; adds json to --format")
    ap.add_argument("--compare", nargs="+", default=None, metavar="LABEL=REPORT.json",
                    help="render a side-by-side comparison of saved JSON reports (no server run); each entry "
                         "is `label=path` or just `path`")
    ap.add_argument("--title", default="DiffusionGemma benchmark comparison", help="title for --compare output")
    ap.add_argument("--note", action="append", default=None, metavar="TEXT",
                    help="add an annotation line to --compare output (repeatable; e.g. VRAM/ceiling facts)")
    args = ap.parse_args()

    formats = [f.strip() for f in args.format.split(",") if f.strip()]
    out = args.out
    if args.json:  # back-compat: --json PATH behaves like --format json --out PATH
        if "json" not in formats:
            formats = ["json"] if formats == ["text"] else formats + ["json"]
        out = out or args.json
    bad = [f for f in formats if f not in RENDERERS]
    if bad:
        ap.error(f"unknown format(s): {bad}; choose from {list(RENDERERS)}")

    # --compare: render saved reports side by side, no server run.
    if args.compare:
        comp = build_comparison(args.compare, title=args.title, notes=args.note)
        write_outputs(comp, formats, out, renderers=COMPARE_RENDERERS)
        sys.exit(0)

    print(f"calibrating tokenizer ratio against {args.url} ...", file=sys.stderr, flush=True)
    ratio = measure_ratio(args.url)
    print(f"  chars/token ~= {ratio:.2f}\n", file=sys.stderr, flush=True)

    specs = make_specs(args.prompt_max, args.output_max)
    results = []
    for spec in specs:
        print(f"running {spec['name']:<12} ...", end=" ", file=sys.stderr, flush=True)
        try:
            res = run_spec(args.url, spec, ratio, args.reps, args.warmup)
        except Exception as e:  # noqa: BLE001
            print(f"ERROR {e}", file=sys.stderr)
            results.append({"name": spec["name"], "axis": spec["axis"], "bucket": spec["bucket"],
                            "error": str(e), "size_ok": False, "correct": False})
            continue
        print(f"prompt={res['prompt_tok']:>6} compl={res['compl_tok']:>6} "
              f"denoise={res['ms_per_step']:.1f}ms/step  "
              f"size={'OK' if res['size_ok'] else 'MISS'} corr={'OK' if res['correct'] else 'FAIL'}"
              f"{'' if res['correct'] else '  (' + res.get('why', '') + ')'}", file=sys.stderr)
        results.append(res)

    cov = build_coverage(results, args.prompt_max, args.output_max)
    report = {
        "meta": {"url": args.url, "ratio": ratio, "reps": args.reps,
                 "prompt_max": args.prompt_max, "output_max": args.output_max},
        "results": results,
        "coverage": cov,
        # flat back-compat keys for older JSON consumers
        "missing_prompt": cov["missing_prompt"], "missing_output": cov["missing_output"], "pass": cov["pass"],
    }
    write_outputs(report, formats, out)
    sys.exit(0 if cov["pass"] else 1)


if __name__ == "__main__":
    main()
