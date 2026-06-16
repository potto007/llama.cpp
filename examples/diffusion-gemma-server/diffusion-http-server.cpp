// OpenAI-compatible HTTP server for DiffusionGemma. Loads the GGUF once, then serves
// /v1/chat/completions (and /v1/models, /health) by running the same entropy-bound denoiser the CLI's
// --diffusion-visual mode uses (diffusion_generate_entropy_bound). Tokenization, chat templating and
// detokenization all happen here from the GGUF's own embedded tokenizer + chat template, so clients need
// no tokenizer files.
//
// This is a STANDALONE server, NOT the llama-server router: DiffusionGemma decodes non-autoregressively
// (bidirectional attention over a fixed canvas, no KV reuse across tokens), which the router's slot/decode
// machinery cannot drive. See examples/diffusion/diffusion.cpp for the decode loop.
//
// Usage: llama-diffusion-server -m <model.gguf> [--host H] [--port P] [-ngl N] [-c CTX] [--flash-attn] [-a ALIAS]
//   ctx (-c) <= 0 auto-sizes the largest non-causal context that fits VRAM (else RAM), like the visual server.
//
// Request extensions beyond the OpenAI schema: "seed" (int, default 0) and "n_blocks" (int) which denoises
// extra canvas blocks autoregressively. If "max_tokens" is given, n_blocks defaults to ceil(max_tokens/canvas).

#include "llama.h"
#include "ggml-backend.h"
#include "common.h"
#include "chat.h"
#include "../diffusion/diffusion.h"

#include <nlohmann/json.hpp>
#include <cpp-httplib/httplib.h>

#include <atomic>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <functional>
#include <mutex>
#include <string>
#include <vector>

using json = nlohmann::ordered_json;

// ----------------------------------------------------------------------------- model metadata helpers
static float meta_f(llama_model * m, const char * key, float def) {
    char buf[32];
    return llama_model_meta_val_str(m, key, buf, sizeof(buf)) >= 0 ? strtof(buf, nullptr) : def;
}
static int32_t meta_i(llama_model * m, const char * key, int32_t def) {
    char buf[32];
    return llama_model_meta_val_str(m, key, buf, sizeof(buf)) >= 0 ? (int32_t) strtol(buf, nullptr, 10) : def;
}

// Trim a denoised canvas like the CLI: cut at the first end-of-generation token, else at the onset of a
// repetition loop (a token recurring at stride 1-2 for >= 6 steps).
static size_t trim_canvas(const llama_vocab * vocab, const llama_token * canvas, size_t n) {
    size_t cut = n;
    for (size_t i = 0; i < n; i++) {
        if (llama_vocab_is_eog(vocab, canvas[i])) { cut = i; break; }
    }
    for (size_t i = 0; i + 1 < cut; i++) {
        bool loop = false;
        for (size_t stride = 1; stride <= 2 && !loop; stride++) {
            size_t reps = 0;
            for (size_t j = i; j + stride < n && canvas[j] == canvas[j + stride]; j += stride) { reps++; }
            loop = reps >= 6;
        }
        if (loop) { cut = i; break; }
    }
    return cut;
}

// DiffusionGemma wraps its chain-of-thought in channel markers and puts the user-facing answer after the
// final one, e.g.  "<|channel>thought\n...reasoning...<channel|>The answer."  These render as plain text
// (they are model-specific tokens, not standard special tokens), so a chat client sees the reasoning inline.
// Split the raw text into {reasoning, content}: content is everything after the LAST close marker; reasoning
// is what precedes it with the leading open marker stripped. If no marker is present, it's all content.
struct split_text {
    std::string reasoning;
    std::string content;
};

static std::string strip_ws(const std::string & s) {
    const size_t b = s.find_first_not_of(" \t\r\n");
    if (b == std::string::npos) return "";
    const size_t e = s.find_last_not_of(" \t\r\n");
    return s.substr(b, e - b + 1);
}

static split_text split_channels(const std::string & raw) {
    static const std::string close = "<channel|>";
    static const std::string open  = "<|channel>";
    split_text r;
    const size_t pos = raw.rfind(close);
    if (pos == std::string::npos) {
        // no close marker: if a lone open marker leaked, drop everything up to it; else it's all content
        const size_t op = raw.find(open);
        r.content = strip_ws(op == std::string::npos ? raw : raw.substr(0, op));
        return r;
    }
    r.content = strip_ws(raw.substr(pos + close.size()));
    std::string reasoning = raw.substr(0, pos);
    const size_t op = reasoning.find(open);
    if (op != std::string::npos) reasoning = reasoning.substr(op + open.size());
    // the open marker is usually followed by a channel name ("thought"); drop a leading bare word
    reasoning = strip_ws(reasoning);
    r.reasoning = reasoning;
    return r;
}

// ----------------------------------------------------------------------------- server state (one model)
struct server_state {
    llama_model *             model        = nullptr;
    llama_context *           ctx          = nullptr;
    const llama_vocab *       vocab        = nullptr;
    common_chat_templates_ptr chat_templates;
    int64_t                   canvas_length = 0;
    int                       maxtok        = 0;
    diffusion_eb_params       base;
    std::string               model_id;
    bool                      strip_channels = true;   // split off the <|channel>thought...<channel|> reasoning
    bool                      show_reasoning = false;  // surface it as reasoning_content instead of dropping it
    std::vector<llama_token>  output_tokens;  // reused under gen_mtx
    std::mutex                gen_mtx;         // ctx is single + not thread-safe: serialize generation
};

struct gen_result {
    bool        ok          = false;
    int         http_status = 200;
    std::string err;
    std::string text;
    int         prompt_n     = 0;
    int         completion_n = 0;
};

// Run the block-diffusion loop for one chat request. on_progress (optional) is called with the full raw
// answer text after each committed block, so a streaming caller can split + diff it. gen_mtx must be held.
static gen_result run_generation(server_state & st, const json & messages, int seed, int n_blocks,
                                 const std::function<void(const std::string &)> & on_progress) {
    gen_result r;

    std::vector<llama_token> prefix;
    try {
        std::vector<common_chat_msg> msgs = common_chat_msgs_parse_oaicompat(messages);
        common_chat_templates_inputs inputs;
        inputs.messages              = msgs;
        inputs.add_generation_prompt = true;
        const std::string prompt = common_chat_templates_apply(st.chat_templates.get(), inputs).prompt;
        prefix = common_tokenize(st.vocab, prompt, /*add special*/ true, /*parse special*/ true);
    } catch (const std::exception & e) {
        r.http_status = 400;
        r.err = std::string("failed to apply chat template: ") + e.what();
        return r;
    }
    if (prefix.empty()) {
        r.http_status = 400;
        r.err = "empty prompt after templating";
        return r;
    }

    const int P = (int) prefix.size();
    std::vector<llama_token> response;  // cumulative committed canvas tokens across blocks

    for (int b = 0; b < std::max(1, n_blocks); b++) {
        const int32_t prefix_len = (int32_t) prefix.size();
        const int32_t max_length = prefix_len + (int32_t) st.canvas_length;
        if (max_length > st.maxtok) {
            if (b == 0) {
                r.http_status = 400;
                r.err = string_format("conversation too long: needs %d tokens (prompt %d + canvas %d) but the "
                                      "server context is %d. Shorten the prompt or restart with a larger -c.",
                                      (int) max_length, prefix_len, (int) st.canvas_length, st.maxtok);
                return r;
            }
            break;  // out of room mid-generation: keep what we have
        }

        diffusion_eb_params eb = st.base;
        eb.max_length = max_length;
        eb.seed       = seed + b;  // deterministic, distinct per block

        int32_t n_generated = 0;
        diffusion_generate_entropy_bound(st.ctx, prefix.data(), st.output_tokens.data(), prefix_len, eb, n_generated);
        if (n_generated <= prefix_len) {
            if (b == 0) {
                r.http_status = 500;
                r.err = "diffusion generation failed";
                return r;
            }
            break;
        }

        const llama_token * canvas = st.output_tokens.data() + prefix_len;
        const size_t        cut    = trim_canvas(st.vocab, canvas, (size_t) st.canvas_length);
        response.insert(response.end(), canvas, canvas + cut);

        if (on_progress) {
            on_progress(common_detokenize(st.vocab, response, /*special*/ false));
        }

        if (cut < (size_t) st.canvas_length) { break; }       // eog / repetition loop: answer complete
        prefix.insert(prefix.end(), canvas, canvas + cut);    // commit the block, denoise the next
    }

    r.ok           = true;
    r.text         = common_detokenize(st.vocab, response, /*special*/ false);
    r.prompt_n     = P;
    r.completion_n = (int) response.size();
    return r;
}

// ----------------------------------------------------------------------------- response helpers
static std::atomic<uint64_t> g_id_counter{0};
static std::string new_id(const char * prefix) {
    return std::string(prefix) + std::to_string((long) std::time(nullptr)) + "-" +
           std::to_string(g_id_counter.fetch_add(1));
}

static json error_json(const std::string & msg, const char * type) {
    return json{ { "error", { { "message", msg }, { "type", type }, { "code", nullptr } } } };
}

// ----------------------------------------------------------------------------- arg parsing
struct cli_args {
    std::string model;
    std::string host  = "127.0.0.1";
    int         port  = 8080;
    int         ngl   = 0;
    int         ctx   = 0;       // 0 = auto-size
    bool        fa    = false;
    bool        raw   = false;            // keep the <|channel> reasoning markers in content
    bool        show_reasoning = false;   // surface reasoning as reasoning_content
    std::string alias;
};

static bool parse_args(int argc, char ** argv, cli_args & a) {
    if (const char * e = getenv("NGL")) a.ngl = atoi(e);
    if (const char * e = getenv("FA"))  a.fa  = atoi(e) != 0;
    for (int i = 1; i < argc; i++) {
        std::string s = argv[i];
        auto next = [&](const char * name) -> const char * {
            if (i + 1 >= argc) { fprintf(stderr, "%s requires a value\n", name); return nullptr; }
            return argv[++i];
        };
        if (s == "-m" || s == "--model") { const char * v = next("--model"); if (!v) return false; a.model = v; }
        else if (s == "--host")                       { const char * v = next("--host"); if (!v) return false; a.host = v; }
        else if (s == "--port")                       { const char * v = next("--port"); if (!v) return false; a.port = atoi(v); }
        else if (s == "-ngl" || s == "--n-gpu-layers"){ const char * v = next("-ngl"); if (!v) return false; a.ngl = atoi(v); }
        else if (s == "-c" || s == "--ctx-size")      { const char * v = next("-c"); if (!v) return false; a.ctx = atoi(v); }
        else if (s == "-a" || s == "--alias")         { const char * v = next("-a"); if (!v) return false; a.alias = v; }
        else if (s == "-fa" || s == "--flash-attn")   { a.fa = true; }
        else if (s == "--raw")                        { a.raw = true; }
        else if (s == "--show-reasoning")             { a.show_reasoning = true; }
        else if (s == "-h" || s == "--help")          { return false; }
        else if (a.model.empty() && s[0] != '-')      { a.model = s; }  // positional model path
        else { fprintf(stderr, "unknown argument: %s\n", s.c_str()); return false; }
    }
    return !a.model.empty();
}

// ----------------------------------------------------------------------------- main
int main(int argc, char ** argv) {
    cli_args args;
    if (!parse_args(argc, argv, args)) {
        fprintf(stderr,
                "usage: %s -m <model.gguf> [--host H] [--port P] [-ngl N] [-c CTX] [--flash-attn] [-a ALIAS]\n"
                "          [--raw] [--show-reasoning]\n"
                "  --raw             keep the model's <|channel> reasoning markers in the response content\n"
                "  --show-reasoning  return the reasoning as an OpenAI reasoning_content field (default: drop)\n",
                argv[0]);
        return 1;
    }

    llama_backend_init();
    ggml_backend_load_all();  // load dynamic backends so -ngl can offload to GPU

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = args.ngl;
    llama_model * model = llama_model_load_from_file(args.model.c_str(), mparams);
    if (!model) { fprintf(stderr, "failed to load model: %s\n", args.model.c_str()); return 1; }
    if (!llama_model_is_diffusion(model)) {
        fprintf(stderr, "%s is not a diffusion model; use llama-server for autoregressive models\n",
                args.model.c_str());
        llama_model_free(model);
        return 1;
    }

    server_state st;
    st.model = model;
    st.strip_channels = !args.raw;
    st.show_reasoning = args.show_reasoning;
    st.vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(st.vocab);

    st.chat_templates = common_chat_templates_init(model, "");

    {
        char buf[32];
        if (llama_model_meta_val_str(model, "diffusion.canvas_length", buf, sizeof(buf)) >= 0) {
            st.canvas_length = strtol(buf, nullptr, 10);
        }
    }
    if (st.canvas_length <= 0) {
        fprintf(stderr, "model has no diffusion.canvas_length metadata\n");
        llama_model_free(model);
        return 1;
    }

    // model id: explicit alias, else GGUF general.name, else the file basename
    if (!args.alias.empty()) {
        st.model_id = args.alias;
    } else {
        char buf[256];
        if (llama_model_meta_val_str(model, "general.name", buf, sizeof(buf)) >= 0 && buf[0]) {
            st.model_id = buf;
        } else {
            const size_t slash = args.model.find_last_of("/\\");
            st.model_id = slash == std::string::npos ? args.model : args.model.substr(slash + 1);
        }
    }

    // Enable the self-conditioning graph before context creation so the reserve sizes the compute buffer
    // (matches the CLI / visual server). The entropy-bound decoder supplies the real SC state per step.
    llama_diffusion_set_sc(model, nullptr, 0.0f, 1.0f, true);

    // GPU enumeration: Stage 1+2 (device SC / prompt-KV) are single-device features; grab a GPU handle so the
    // auto-sizer can read free VRAM.
    int gpu_devs = 0;
    ggml_backend_dev_t gpu_dev = nullptr;
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        ggml_backend_dev_t d = ggml_backend_dev_get(i);
        const auto dt = ggml_backend_dev_type(d);
        if (dt == GGML_BACKEND_DEVICE_TYPE_GPU || dt == GGML_BACKEND_DEVICE_TYPE_IGPU) {
            gpu_devs++;
            if (!gpu_dev) gpu_dev = d;
        }
    }
    const bool one_gpu = (gpu_devs <= 1);

    // Causal prefill runs in chunks of this many tokens (one llama_decode each). The compute buffer is sized
    // by the worst-case reserve at n_tokens = min(n_ctx, n_ubatch), so capping n_ubatch here - not n_ctx -
    // decouples the activation buffer from the prompt length. Must be >= canvas_length (the DECODE batch).
    const int prefill_chunk = 2048;
    auto make_cparams = [&](int n) {
        llama_context_params c = llama_context_default_params();
        c.n_ctx    = (uint32_t) n;
        c.n_batch  = (uint32_t) n;
        c.n_ubatch = (uint32_t) std::min(n, prefill_chunk);  // chunked causal prefill: one chunk per ubatch
        // Cap outputs to the canvas: the decoder only reads canvas-row logits, so encode() need not reserve an
        // [n_ctx, n_vocab] buffer (~1 MB/token at a 262k vocab). Lets context scale past the old ~8-12k ceiling.
        c.n_outputs_max = (uint32_t) st.canvas_length;
        c.no_perf  = true;
        c.flash_attn_type = args.fa ? LLAMA_FLASH_ATTN_TYPE_ENABLED : LLAMA_FLASH_ATTN_TYPE_DISABLED;
        return c;
    };

    // Resolve the context size. Descending probe keeps the largest context that actually allocates (the model
    // can spill to RAM when -ngl exceeds VRAM). Mirrors the visual server's auto-sizer.
    const int n_ctx_train = (int) llama_model_n_ctx_train(model);
    const int n_head      = std::max(1, (int) llama_model_n_head(model));
    const int floor_ctx   = std::max((int) st.canvas_length * 4, 2048);
    const int auto_ceil   = n_ctx_train > 0 ? std::min(n_ctx_train, 65536) : 65536;
    const int cands[]     = {65536, 49152, 40960, 32768, 24576, 20480, 16384, 12288, 8192, 6144, 4096, 2048};

    size_t v_free = 0, v_total = 0;
    if (gpu_dev) ggml_backend_dev_memory(gpu_dev, &v_free, &v_total);
    size_t r_free = 0, r_total = 0;
    if (ggml_backend_dev_t cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU))
        ggml_backend_dev_memory(cpu_dev, &r_free, &r_total);
    const size_t weights    = llama_model_size(model);
    const size_t ram_budget = r_free > weights ? (size_t) ((r_free - weights) * 0.7) : 0;

    // The prompt-KV store (per_tok x N, device-resident, F16 under FA) is allocated lazily at first prefill -
    // AFTER llama_init_from_model - so a context that loads can still OOM on a full-length prompt. It is now
    // the dominant runtime allocation (the activation buffer is chunked), so size context against it: require
    // the store for an N-token prompt to fit in the GPU free space measured after the context is created.
    const size_t pkv_per_tok  = llama_diffusion_pkv_bytes_per_token(model, args.fa);
    // Headroom beyond weights + store(N) for allocations the probe can't see at init: the F32 concat working
    // set (prefix cast + Kfull/Vfull at sequence length) and the self-conditioning buffers (sc_embT ~n_vocab x
    // n_embd F16, sc_dev ~n_vocab x canvas F32) - empirically ~3.5 GB near the ceiling. Generous so the
    // advertised ctx survives a full-length prompt rather than OOMing on it.
    const size_t vram_headroom = std::max(v_total ? (size_t) (v_total * 0.15) : 0, (size_t) 4096 * 1024 * 1024);

    auto probe = [&](int ceil_ctx, size_t budget, size_t gpu_headroom, int * out_n) -> llama_context * {
        for (int raw : cands) {
            if (raw > ceil_ctx) continue;
            int N = (int) ((raw / st.canvas_length) * st.canvas_length);  // whole canvases only
            if (N < floor_ctx) break;
            if (budget && !args.fa) {  // FA off: an fp32 [n_head, N, N] scores buffer is unavoidable. FA on
                                       // eliminates it, so don't pessimistically cap N -- let alloc decide.
                const double min_scores = (double) n_head * (double) N * (double) N * 4.0;
                if (min_scores > (double) budget * 0.9) continue;
            }
            llama_context * c = llama_init_from_model(model, make_cparams(N));
            if (!c) continue;
            if (gpu_dev) {  // reject if the lazily-grown store + headroom won't fit GPU free space
                size_t f = 0, t = 0; ggml_backend_dev_memory(gpu_dev, &f, &t);
                const size_t need = pkv_per_tok * (size_t) N + std::max(gpu_headroom, vram_headroom);
                if (f < need) { llama_free(c); continue; }
            }
            *out_n = N;
            return c;
        }
        return nullptr;
    };

    llama_context * ctx = nullptr;
    const char * reason = "auto";
    if (args.ctx > 0) {  // explicit budget: honour exactly if it fits (store included), else degrade via probe
        const double sc     = (double) n_head * (double) args.ctx * (double) args.ctx * 4.0;
        const size_t budget = std::max(v_free, ram_budget);
        if (args.fa || !budget || sc <= (double) budget * 0.9) {  // FA off: gate on the fp32 scores estimate
            ctx = llama_init_from_model(model, make_cparams(args.ctx));
            if (ctx && gpu_dev) {  // ensure the prompt-KV store for args.ctx tokens also fits
                size_t f = 0, t = 0; ggml_backend_dev_memory(gpu_dev, &f, &t);
                if (f < pkv_per_tok * (size_t) args.ctx + vram_headroom) { llama_free(ctx); ctx = nullptr; }
            }
            if (ctx) { st.maxtok = args.ctx; reason = "requested"; }
        }
    }
    if (!ctx) {
        const int ceil_ctx = args.ctx > 0 ? std::min(auto_ceil, args.ctx) : auto_ceil;
        int n1 = 0;
        ctx = probe(ceil_ctx, v_free, vram_headroom, &n1);
        if (ctx) { st.maxtok = n1; reason = "vram"; }
        if ((!ctx || n1 < ceil_ctx) && ram_budget > v_free) {
            int n2 = 0;
            llama_context * c = probe(ceil_ctx, ram_budget, vram_headroom, &n2);
            if (c && n2 > st.maxtok) { if (ctx) llama_free(ctx); ctx = c; st.maxtok = n2; reason = "ram"; }
            else if (c) { llama_free(c); }
        }
    }
    if (!ctx) {  // last resort: the floor so a very tight machine still loads
        int N = std::max((int) st.canvas_length, (int) ((floor_ctx / st.canvas_length) * st.canvas_length));
        ctx = llama_init_from_model(model, make_cparams(N));
        if (ctx) { st.maxtok = N; reason = "floor"; }
    }
    if (!ctx) {
        fprintf(stderr, "failed to size a context that fits (VRAM free=%zu MiB, RAM budget=%zu MiB)\n",
                v_free / (1024 * 1024), ram_budget / (1024 * 1024));
        llama_model_free(model);
        return 1;
    }
    st.ctx = ctx;
    llama_set_causal_attn(ctx, false);

    // entropy-bound params from GGUF metadata + reference defaults (kept in sync with the CLI / visual server)
    st.base.max_denoising_steps  = meta_i(model, "diffusion.eb_max_steps", 48);
    st.base.t_min                = meta_f(model, "diffusion.eb_t_min", 0.4f);
    st.base.t_max                = meta_f(model, "diffusion.eb_t_max", 0.8f);
    st.base.entropy_bound        = meta_f(model, "diffusion.eb_entropy_bound", 0.1f);
    st.base.stability_threshold  = meta_i(model, "diffusion.eb_stability_threshold", 1);
    st.base.confidence_threshold = meta_f(model, "diffusion.eb_confidence_threshold", 0.005f);
    st.base.kv_cache          = one_gpu;  // Stage 1+2 are single-device features (auto-enable, like the CLI)
    st.base.gpu_sampling      = one_gpu;
    st.base.gpu_sample_reduce = one_gpu;

    st.output_tokens.resize(st.maxtok);

    fprintf(stderr,
            "llama-diffusion-server: model=%s n_vocab=%d canvas=%d ctx=%d (%s) ngl=%d fa=%s "
            "gpu_sampling=%s kv_cache=%s\n",
            st.model_id.c_str(), n_vocab, (int) st.canvas_length, st.maxtok, reason, args.ngl,
            args.fa ? "on" : "off", st.base.gpu_sampling ? "on" : "off", st.base.kv_cache ? "on" : "off");

    // ------------------------------------------------------------------------- HTTP
    httplib::Server svr;
    svr.set_payload_max_length(1024 * 1024 * 32);

    svr.Get("/health", [](const httplib::Request &, httplib::Response & res) {
        res.set_content(json{ { "status", "ok" } }.dump(), "application/json");
    });

    svr.Get("/v1/models", [&st](const httplib::Request &, httplib::Response & res) {
        json data = json::array();
        data.push_back(json{ { "id", st.model_id }, { "object", "model" }, { "owned_by", "llamacpp" } });
        res.set_content(json{ { "object", "list" }, { "data", data } }.dump(), "application/json");
    });

    svr.Post("/v1/chat/completions", [&st](const httplib::Request & req, httplib::Response & res) {
        json body;
        try {
            body = json::parse(req.body);
        } catch (const std::exception & e) {
            res.status = 400;
            res.set_content(error_json(std::string("invalid JSON: ") + e.what(), "invalid_request_error").dump(),
                            "application/json");
            return;
        }
        if (!body.contains("messages") || !body.at("messages").is_array()) {
            res.status = 400;
            res.set_content(error_json("'messages' array is required", "invalid_request_error").dump(),
                            "application/json");
            return;
        }

        const json & messages = body.at("messages");
        const int  seed   = body.value("seed", 0);
        const bool stream = body.value("stream", false);

        // n_blocks: explicit field wins; else derive from max_tokens; else 1.
        int n_blocks = body.value("n_blocks", 0);
        if (n_blocks <= 0) {
            const int max_tokens = body.value("max_tokens", 0);
            n_blocks = max_tokens > 0
                           ? std::max(1, (int) ((max_tokens + st.canvas_length - 1) / st.canvas_length))
                           : 1;
        }

        const std::string id      = new_id("chatcmpl-");
        const long        created = (long) std::time(nullptr);

        if (!stream) {
            std::lock_guard<std::mutex> lock(st.gen_mtx);
            gen_result r = run_generation(st, messages, seed, n_blocks, nullptr);
            if (!r.ok) {
                res.status = r.http_status;
                res.set_content(error_json(r.err, r.http_status == 400 ? "invalid_request_error" : "server_error").dump(),
                                "application/json");
                return;
            }
            const split_text sp = st.strip_channels ? split_channels(r.text) : split_text{ "", r.text };
            json message = { { "role", "assistant" }, { "content", sp.content } };
            if (st.show_reasoning && !sp.reasoning.empty()) {
                message["reasoning_content"] = sp.reasoning;
            }
            json resp = {
                { "id", id },
                { "object", "chat.completion" },
                { "created", created },
                { "model", st.model_id },
                { "choices", json::array({ json{
                      { "index", 0 },
                      { "message", message },
                      { "finish_reason", "stop" } } }) },
                { "usage", { { "prompt_tokens", r.prompt_n },
                             { "completion_tokens", r.completion_n },
                             { "total_tokens", r.prompt_n + r.completion_n } } },
            };
            res.set_content(resp.dump(), "application/json");
            return;
        }

        // streaming: SSE, one chat.completion.chunk per committed block, then [DONE]
        res.set_chunked_content_provider(
            "text/event-stream",
            [&st, messages, seed, n_blocks, id, created](size_t, httplib::DataSink & sink) {
                auto send_chunk = [&](const json & delta, const char * finish) {
                    json chunk = {
                        { "id", id },
                        { "object", "chat.completion.chunk" },
                        { "created", created },
                        { "model", st.model_id },
                        { "choices", json::array({ json{
                              { "index", 0 },
                              { "delta", delta },
                              { "finish_reason", finish ? json(finish) : json(nullptr) } } }) },
                    };
                    const std::string line = "data: " + chunk.dump() + "\n\n";
                    return sink.write(line.data(), line.size());
                };

                std::lock_guard<std::mutex> lock(st.gen_mtx);
                send_chunk(json{ { "role", "assistant" } }, nullptr);  // OpenAI's first delta carries the role

                // emit only the new suffix of each field as the answer grows across blocks
                std::string sent_content, sent_reasoning;
                auto emit_field = [&](const char * field, const std::string & full, std::string & sent) {
                    if (full.size() <= sent.size() || full.compare(0, sent.size(), sent) != 0) {
                        if (full == sent) return;
                        sent.clear();  // split point shifted: re-emit from scratch (rare; multi-block)
                    }
                    send_chunk(json{ { field, full.substr(sent.size()) } }, nullptr);
                    sent = full;
                };
                gen_result r = run_generation(st, messages, seed, n_blocks,
                    [&](const std::string & full_raw) {
                        const split_text sp = st.strip_channels ? split_channels(full_raw)
                                                                : split_text{ "", full_raw };
                        if (st.show_reasoning && !sp.reasoning.empty()) {
                            emit_field("reasoning_content", sp.reasoning, sent_reasoning);
                        }
                        emit_field("content", sp.content, sent_content);
                    });

                if (!r.ok) {
                    // mid-stream failure: emit an SSE error event (status line is already 200)
                    const std::string err = "data: " +
                        error_json(r.err, r.http_status == 400 ? "invalid_request_error" : "server_error").dump() +
                        "\n\n";
                    sink.write(err.data(), err.size());
                } else {
                    send_chunk(json::object(), "stop");
                }
                const std::string done = "data: [DONE]\n\n";
                sink.write(done.data(), done.size());
                sink.done();
                return true;
            });
    });

    fprintf(stderr, "listening on http://%s:%d\n", args.host.c_str(), args.port);
    if (!svr.listen(args.host, args.port)) {
        fprintf(stderr, "failed to bind %s:%d\n", args.host.c_str(), args.port);
        llama_free(st.ctx);
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }

    llama_free(st.ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
