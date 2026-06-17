# DiffusionGemma prompt-KV store: benchmark comparison

Configs: `Baseline`, `F16`, `Q8 no-cache`, `Q8+cache`

## Denoise ms/step - prompt axis (latency vs context)

| prompt tok | Baseline | F16 | Q8 no-cache | Q8+cache |
|---:|---:|---:|---:|---:|
| 1,309 | 292.5 | 58.3 | 109.1 | 65.3 |
| 2,768 | 500.2 | 54.3 | 110.1 | 66.3 |
| 5,568 | 909.9 | 59.4 | 113.7 | 69.0 |
| 10,897 | 1742.2 | 58.2 | 112.7 | 68.4 |
| 20,785 | - | 61.0 | 115.1 | 70.8 |

## Denoise ms/step - output axis (generation-bound)

| out bkt | Baseline | F16 | Q8 no-cache | Q8+cache |
|---:|---:|---:|---:|---:|
| 1k | - | 56.1 | 108.6 | 61.7 |
| 2k | - | 56.6 | 109.3 | 62.2 |
| 4k | - | 57.7 | 110.7 | 63.0 |
| 8k | - | 59.7 | 112.2 | 65.0 |

## Effective output tok/s - output axis

| out bkt | Baseline | F16 | Q8 no-cache | Q8+cache |
|---:|---:|---:|---:|---:|
| 1k | - | 183.2 | 102.1 | 169.0 |
| 2k | - | 160.7 | 97.8 | 156.7 |
| 4k | - | 130.8 | 82.0 | 122.1 |
| 8k | - | 93.0 | 58.5 | 78.5 |

## Prefill tok/s - prompt axis

| prompt tok | Baseline | F16 | Q8 no-cache | Q8+cache |
|---:|---:|---:|---:|---:|
| 1,309 | 2068 | 29634 | 29784 | 29824 |
| 2,768 | 2217 | 16304 | 14166 | 13978 |
| 5,568 | 1252 | 15467 | 12654 | 12614 |
| 10,897 | 2099 | 12457 | 9811 | 9787 |
| 20,785 | - | 10454 | 8407 | 8430 |

## Notes

- Hardware: RTX 5090 (32 GB), DiffusionGemma 26B-A4B Q4_K_M, flash-attn, seed 42.
- Baseline = DG_FATTN_KV_PAD=1 (CPU-bound global FA); F16 = pad-fixed default; Q8 = SWA-only Q8_0 store; Q8+cache = with per-block dequant cache (shipped).
- Context ceiling (auto-sized): F16 32768 -> Q8 65536 (doubled).
- VRAM saved @30k prompt vs F16: Q8 no-cache -2.8 GB, Q8+cache -2.3 GB (cache ~0.5 GB).
- Baseline output-axis and 16k rows omitted: CPU-bound runs are prohibitively slow.
