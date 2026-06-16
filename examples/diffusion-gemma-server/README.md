# DiffusionGemma servers

DiffusionGemma decodes **non-autoregressively** (bidirectional attention over a fixed-length canvas, no
per-token KV reuse), so it cannot be served by the `llama-server` router. These standalone binaries wrap the
entropy-bound denoiser in `examples/diffusion/diffusion.cpp` instead.

| binary | protocol | use |
|---|---|---|
| `llama-diffusion-server` | **HTTP, OpenAI-compatible** | drop-in `/v1/chat/completions` endpoint |
| `llama-diffusion-gemma-visual-server` | stdin/stdout pipe | per-step canvas frames for a UI |
| `llama-diffusion-gemma-server` | stdin/stdout pipe | raw forward logits for a Python driver |

## `llama-diffusion-server` (HTTP)

OpenAI-compatible server. Tokenization, chat templating and detokenization all use the GGUF's own embedded
tokenizer + chat template, so clients need no tokenizer files.

### Build

```bash
cmake -B build -DGGML_CUDA=ON
cmake --build build -j --target llama-diffusion-server
```

### Run

```bash
./build/bin/llama-diffusion-server -m model.gguf -ngl 99 --host 127.0.0.1 --port 8088
```

| flag | default | meaning |
|---|---|---|
| `-m, --model` | (required) | path to the DiffusionGemma GGUF |
| `--host` / `--port` | `127.0.0.1` / `8080` | bind address |
| `-ngl, --n-gpu-layers` | `0` (env `NGL`) | layers to offload |
| `-c, --ctx-size` | `0` = auto | `<=0` auto-sizes the largest non-causal context that fits VRAM, else RAM |
| `-fa, --flash-attn` | off (env `FA`) | enable flash attention |
| `-a, --alias` | GGUF name / filename | id reported by `/v1/models` |

### Endpoints

- `GET /health` -> `{"status":"ok"}`
- `GET /v1/models` -> the single loaded model
- `POST /v1/chat/completions` -> OpenAI chat completion; `stream: true` gives SSE chunks

Extra request fields beyond the OpenAI schema: `seed` (int) and `n_blocks` (denoise extra canvas blocks
autoregressively). If `max_tokens` is given, `n_blocks` defaults to `ceil(max_tokens / canvas_length)`.

```bash
curl http://127.0.0.1:8088/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "messages": [{"role":"user","content":"What is the capital of France?"}],
  "seed": 42
}'
```

### Reasoning channel

DiffusionGemma wraps its chain-of-thought in `<|channel>thought ... <channel|>answer` markers (model-specific
text, not standard special tokens). By default the server strips them and returns only the final answer in
`content`. Flags:

- `--show-reasoning` — also return the thought as an OpenAI `reasoning_content` field instead of dropping it
- `--raw` — disable stripping; return the markers verbatim in `content`

### Use with opencode (or any OpenAI-compatible client)

Add a provider pointing at the server's port:

```json
{
  "provider": {
    "diffusiongemma": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "DiffusionGemma (local)",
      "options": { "baseURL": "http://127.0.0.1:8088/v1" },
      "models": { "diffusiongemma-26b-a4b": { "name": "DiffusionGemma 26B-A4B" } }
    }
  }
}
```

The model key must match the server's reported id (`-a diffusiongemma-26b-a4b`). The server ignores `tools`,
`temperature`, etc., so it works for chat but not tool-calling/agentic flows (the model isn't tool-trained).

### Notes / limitations

- One model per process; generation is serialized (single non-thread-safe context). Run multiple processes on
  different ports for concurrency.
- Streaming emits one chunk per committed block, not per denoising step.
- This server does **not** join the `--models-preset` router; point clients directly at its port.
