// dual-context-server: single-process, two-context OpenAI-compatible server.
//
// Loads the model weights ONCE (one llama_model) and serves two roles that each
// own a PRIVATE llama_context (KV cache) and prompt cache:
//   - orchestrator (default :8080) - one big KV slot, long-context driver
//   - sub-call worker (default :8081) - many small slots, bursty map-reduce calls
//
// This replaces the WSL2-impossible two-process CUDA-IPC weight share (ADR-0013)
// with intra-process multi-context sharing (ADR-0014). Chain-of-thought is a
// per-request concern handled by the client (prehend's subcall_enable_thinking),
// so both roles run the same server core; they differ only in ctx / parallel /
// port. Verified prereq: cuda-llm-weight-share/wsl-experiments/multi_ctx_proof.c
// proved one model backs two private KV caches with no weight duplication.
//
// GPU runs are gated elsewhere; this file is compile-and-wire only.

#include "server-context.h"
#include "server-http.h"
#include "server-tools.h"

#include "arg.h"
#include "common.h"
#include "llama.h"
#include "log.h"

#include <atomic>
#include <clocale>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <thread>
#include <vector>

#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__))
#include <signal.h>
#elif defined (_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

static std::function<void(int)> shutdown_handler;
static std::atomic_flag is_terminating = ATOMIC_FLAG_INIT;

static inline void signal_handler(int signal) {
    if (is_terminating.test_and_set()) {
        fprintf(stderr, "Received second interrupt, terminating immediately.\n");
        exit(1);
    }
    shutdown_handler(signal);
}

// wrapper function that handles exceptions and logs errors so a handler never throws
static server_http_context::handler_t ex_wrapper(server_http_context::handler_t func) {
    return [func = std::move(func)](const server_http_req & req) -> server_http_res_ptr {
        std::string message;
        error_type error;
        try {
            return func(req);
        } catch (const std::invalid_argument & e) {
            error = ERROR_TYPE_INVALID_REQUEST;
            message = e.what();
        } catch (const std::exception & e) {
            error = ERROR_TYPE_SERVER;
            message = e.what();
        } catch (...) {
            error = ERROR_TYPE_SERVER;
            message = "unknown error";
        }

        auto res = std::make_unique<server_http_res>();
        res->status = 500;
        try {
            json error_data = format_error_response(message, error);
            res->status = json_value(error_data, "code", 500);
            res->data = safe_json_to_str({{ "error", error_data }});
            SRV_WRN("got exception: %s\n", res->data.c_str());
        } catch (const std::exception & e) {
            SRV_ERR("got another exception: %s | while handling exception: %s\n", e.what(), message.c_str());
            res->data = "Internal Server Error";
        }
        return res;
    };
}

// register the (non-router) OpenAI-compatible route set on one http context / routes pair
static void register_routes(server_http_context & ctx_http, server_routes & routes, server_tools & tools, const common_params & params) {
    ctx_http.get ("/health",                   ex_wrapper(routes.get_health)); // public endpoint (no API key check)
    ctx_http.get ("/v1/health",                ex_wrapper(routes.get_health)); // public endpoint (no API key check)
    ctx_http.get ("/metrics",                  ex_wrapper(routes.get_metrics));
    ctx_http.get ("/props",                    ex_wrapper(routes.get_props));
    ctx_http.post("/props",                    ex_wrapper(routes.post_props));
    ctx_http.get ("/models",                   ex_wrapper(routes.get_models)); // public endpoint (no API key check)
    ctx_http.get ("/v1/models",                ex_wrapper(routes.get_models)); // public endpoint (no API key check)
    ctx_http.post("/completion",               ex_wrapper(routes.post_completions)); // legacy
    ctx_http.post("/completions",              ex_wrapper(routes.post_completions));
    ctx_http.post("/v1/completions",           ex_wrapper(routes.post_completions_oai));
    ctx_http.post("/chat/completions",         ex_wrapper(routes.post_chat_completions));
    ctx_http.post("/v1/chat/completions",      ex_wrapper(routes.post_chat_completions));
    ctx_http.post("/v1/chat/completions/control", ex_wrapper(routes.post_control));
    ctx_http.post("/v1/responses",             ex_wrapper(routes.post_responses_oai));
    ctx_http.post("/responses",                ex_wrapper(routes.post_responses_oai));
    ctx_http.post("/v1/audio/transcriptions",  ex_wrapper(routes.post_transcriptions_oai));
    ctx_http.post("/audio/transcriptions",     ex_wrapper(routes.post_transcriptions_oai));
    ctx_http.post("/v1/messages",              ex_wrapper(routes.post_anthropic_messages)); // anthropic messages API
    ctx_http.post("/infill",                   ex_wrapper(routes.post_infill));
    ctx_http.post("/embedding",                ex_wrapper(routes.post_embeddings)); // legacy
    ctx_http.post("/embeddings",               ex_wrapper(routes.post_embeddings));
    ctx_http.post("/v1/embeddings",            ex_wrapper(routes.post_embeddings_oai));
    ctx_http.post("/rerank",                   ex_wrapper(routes.post_rerank));
    ctx_http.post("/reranking",                ex_wrapper(routes.post_rerank));
    ctx_http.post("/v1/rerank",                ex_wrapper(routes.post_rerank));
    ctx_http.post("/v1/reranking",             ex_wrapper(routes.post_rerank));
    ctx_http.post("/tokenize",                 ex_wrapper(routes.post_tokenize));
    ctx_http.post("/detokenize",               ex_wrapper(routes.post_detokenize));
    ctx_http.post("/apply-template",           ex_wrapper(routes.post_apply_template));
    ctx_http.post("/chat/completions/input_tokens",    ex_wrapper(routes.post_chat_completions_tok));
    ctx_http.post("/v1/chat/completions/input_tokens", ex_wrapper(routes.post_chat_completions_tok));
    ctx_http.post("/responses/input_tokens",           ex_wrapper(routes.post_responses_tok_oai));
    ctx_http.post("/v1/responses/input_tokens",        ex_wrapper(routes.post_responses_tok_oai));
    ctx_http.post("/v1/messages/count_tokens",         ex_wrapper(routes.post_anthropic_count_tokens));
    ctx_http.get ("/lora-adapters",            ex_wrapper(routes.get_lora_adapters));
    ctx_http.post("/lora-adapters",            ex_wrapper(routes.post_lora_adapters));
    ctx_http.get ("/slots",                    ex_wrapper(routes.get_slots));
    ctx_http.post("/slots/:id_slot",           ex_wrapper(routes.post_slots));

    ctx_http.register_gcp_compat();

    if (!params.server_tools.empty()) {
        tools.setup(params.server_tools);
        ctx_http.get ("/tools",           ex_wrapper(tools.handle_get));
        ctx_http.post("/tools",           ex_wrapper(tools.handle_post));
    }
}

// per-role overrides parsed out of argv before handing the rest to common_params_parse
struct role_overrides {
    int  orch_ctx       = -1;
    int  orch_parallel  = -1;
    int  orch_port      = -1;
    int  worker_ctx     = -1;
    int  worker_parallel= -1;
    int  worker_port    = -1;
};

static void print_extra_help(const char * argv0) {
    fprintf(stderr,
        "\n"
        "dual-context-server: one model, two private-KV contexts (orchestrator + worker).\n"
        "usage: %s --model <path> [common llama-server args] [dual-server args]\n"
        "\n"
        "dual-server args (override the shared base for each role):\n"
        "  --orch-ctx N         orchestrator context size   (default: base --ctx-size, or 98304)\n"
        "  --orch-parallel N    orchestrator slots           (default: base --parallel, or 4)\n"
        "  --orch-port P        orchestrator listen port      (default: base --port, or 8080)\n"
        "  --worker-ctx N       worker context size           (default: 65536)\n"
        "  --worker-parallel N  worker slots                  (default: 4)\n"
        "  --worker-port P      worker listen port            (default: 8081)\n"
        "\n"
        "all other flags (--model, --flash-attn, --cache-type-k/v, --jinja, ...) are shared by both roles.\n"
        "chain-of-thought is a per-request client concern; both roles run the same server core.\n",
        argv0);
}

// strip the dual-server-only flags out of argv (common_params_parse would reject them);
// returns the filtered argv for common_params_parse via out_args.
static role_overrides extract_overrides(int argc, char ** argv, std::vector<char *> & out_args, bool & want_help) {
    role_overrides ov;
    auto take_int = [&](const char * flag, int & i, int & dst) {
        if (i + 1 >= argc) {
            fprintf(stderr, "error: %s requires an integer value\n", flag);
            exit(1);
        }
        const char * val = argv[++i];
        char * end = nullptr;
        long v = std::strtol(val, &end, 10);
        if (end == val || *end != '\0') {
            fprintf(stderr, "error: %s expects an integer, got '%s'\n", flag, val);
            exit(1);
        }
        dst = (int) v;
    };
    for (int i = 0; i < argc; ++i) {
        std::string a = argv[i];
        if      (a == "--orch-ctx")        take_int("--orch-ctx",        i, ov.orch_ctx);
        else if (a == "--orch-parallel")   take_int("--orch-parallel",   i, ov.orch_parallel);
        else if (a == "--orch-port")       take_int("--orch-port",       i, ov.orch_port);
        else if (a == "--worker-ctx")      take_int("--worker-ctx",      i, ov.worker_ctx);
        else if (a == "--worker-parallel") take_int("--worker-parallel", i, ov.worker_parallel);
        else if (a == "--worker-port")     take_int("--worker-port",     i, ov.worker_port);
        else {
            if (a == "--help" || a == "-h") { want_help = true; }
            out_args.push_back(argv[i]);
        }
    }
    return ov;
}

// Bring up both roles against the already-loaded, borrowed `model` and block until shutdown.
//
// LIFETIME: every server_context here borrows `model` and owns its OWN llama_context (private KV
// cache). ~server_context -> destroy() -> llama_init.reset() -> llama_free(ctx), and llama_free
// reads from the model during teardown. So the model MUST outlive every server_context. By keeping
// all of them as locals of this function, they are destroyed when it returns - the caller then
// frees the model exactly once, strictly after. The borrowed model is never freed by a context
// (T1 owns_model=false contract). Returns a process exit code.
static int run_dual_roles(llama_model * model, common_params & p_orch, common_params & p_worker) {
    // two server contexts, each borrowing the shared model (private KV per context)
    server_context ctx_orch;
    server_context ctx_worker;

    server_http_context http_orch;
    server_http_context http_worker;
    if (!http_orch.init(p_orch) || !http_worker.init(p_worker)) {
        SRV_ERR("%s", "failed to initialize HTTP server(s)\n");
        return 1;
    }

    server_routes routes_orch(p_orch, ctx_orch);
    server_routes routes_worker(p_worker, ctx_worker);
    server_tools  tools_orch;
    server_tools  tools_worker;

    register_routes(http_orch,   routes_orch,   tools_orch,   p_orch);
    register_routes(http_worker, routes_worker, tools_worker, p_worker);

    // stop HTTP listeners + task loops. Does NOT free the model (the caller does, after we return).
    auto clean_up = [&]() {
        SRV_INF("%s", "cleaning up before exit...\n");
        http_orch.stop();
        http_worker.stop();
        ctx_orch.terminate();
        ctx_worker.terminate();
        if (http_orch.thread.joinable())   { http_orch.thread.join(); }
        if (http_worker.thread.joinable()) { http_worker.thread.join(); }
    };

    // start both HTTP listeners before loading contexts so /health is reachable
    if (!http_orch.start() || !http_worker.start()) {
        clean_up();
        SRV_ERR("%s", "exiting due to HTTP server error\n");
        return 1;
    }

    // build the contexts against the shared model (each creates its own KV cache)
    if (!ctx_orch.load_model(p_orch, model)) {
        clean_up();
        SRV_ERR("%s", "exiting due to orchestrator model init error\n");
        return 1;
    }
    if (!ctx_worker.load_model(p_worker, model)) {
        clean_up();
        SRV_ERR("%s", "exiting due to worker model init error\n");
        return 1;
    }

    routes_orch.update_meta(ctx_orch);
    routes_worker.update_meta(ctx_worker);
    http_orch.is_ready.store(true);
    http_worker.is_ready.store(true);

    SRV_INF("orchestrator listening on %s (ctx=%d, slots=%d)\n",
            http_orch.listening_address.c_str(), p_orch.n_ctx, p_orch.n_parallel);
    SRV_INF("sub-call worker listening on %s (ctx=%d, slots=%d)\n",
            http_worker.listening_address.c_str(), p_worker.n_ctx, p_worker.n_parallel);

    shutdown_handler = [&](int) {
        // unblock both start_loop() calls
        ctx_orch.terminate();
        ctx_worker.terminate();
    };

#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__))
    struct sigaction sigint_action;
    sigint_action.sa_handler = signal_handler;
    sigemptyset (&sigint_action.sa_mask);
    sigint_action.sa_flags = 0;
    sigaction(SIGINT, &sigint_action, NULL);
    sigaction(SIGTERM, &sigint_action, NULL);
#elif defined (_WIN32)
    auto console_ctrl_handler = +[](DWORD ctrl_type) -> BOOL {
        return (ctrl_type == CTRL_C_EVENT) ? (signal_handler(SIGINT), true) : false;
    };
    SetConsoleCtrlHandler(reinterpret_cast<PHANDLER_ROUTINE>(console_ctrl_handler), true);
#endif

    // run the worker's task loop on a thread; the orchestrator's blocks this function.
    std::thread worker_thread([&]() {
        ctx_worker.start_loop();
    });

    // blocks until ctx_orch.terminate() is called (by the signal handler)
    ctx_orch.start_loop();

    if (worker_thread.joinable()) {
        worker_thread.join();
    }

    clean_up();
    // drop the handler before the captured locals (ctx_orch/ctx_worker) go out of scope, so a
    // late signal can't dereference dangling captures.
    shutdown_handler = nullptr;
    // ctx_orch / ctx_worker (and their llama_context) destruct on return, BEFORE the caller's
    // llama_model_free(model). Nothing after this returns may touch a server_context.
    return 0;
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_init();

    // pull our extra flags out so common_params_parse sees only flags it knows
    std::vector<char *> base_argv;
    bool want_help = false;
    role_overrides ov = extract_overrides(argc, argv, base_argv, want_help);

    if (want_help) {
        // print our extra section first; common_params_parse calls exit() on --help,
        // so anything after it would never run.
        print_extra_help(argv[0]);
        common_params tmp;
        common_params_parse((int) base_argv.size(), base_argv.data(), tmp, LLAMA_EXAMPLE_SERVER);
        return 0; // not reached: the parser exits on --help
    }

    // base params shared by both roles
    common_params base;
    if (!common_params_parse((int) base_argv.size(), base_argv.data(), base, LLAMA_EXAMPLE_SERVER)) {
        return 1;
    }

    if (base.model.path.empty()) {
        SRV_ERR("%s", "no model specified; pass --model <path>\n");
        print_extra_help(argv[0]);
        return 1;
    }

    llama_backend_init();
    llama_numa_init(base.numa);

    // resolve per-role params from the shared base + overrides
    common_params p_orch = base;
    p_orch.n_ctx      = ov.orch_ctx      > 0 ? ov.orch_ctx      : (base.n_ctx      > 0 ? base.n_ctx      : 98304);
    p_orch.n_parallel = ov.orch_parallel > 0 ? ov.orch_parallel : (base.n_parallel > 0 ? base.n_parallel : 4);
    p_orch.port       = ov.orch_port     > 0 ? ov.orch_port     : (base.port       > 0 ? base.port       : 8080);

    common_params p_worker = base;
    p_worker.n_ctx      = ov.worker_ctx      > 0 ? ov.worker_ctx      : 65536;
    p_worker.n_parallel = ov.worker_parallel > 0 ? ov.worker_parallel : 4;
    p_worker.port       = ov.worker_port     > 0 ? ov.worker_port     : 8081;

    if (p_orch.port == p_worker.port) {
        SRV_ERR("orchestrator and worker ports must differ (both %d)\n", p_orch.port);
        llama_backend_free();
        return 1;
    }

    // model name aliases (parity with single-role server)
    for (common_params * p : { &p_orch, &p_worker }) {
        if (p->model_alias.empty() && !p->model.name.empty()) {
            p->model_alias.insert(p->model.name);
        }
    }

    SRV_INF("dual-context-server: loading weights once from '%s'\n", base.model.path.c_str());
    common_params_print_info(base, /*has_model*/ true);

    // load the weights ONCE; both roles borrow this model
    auto mparams = common_model_params_to_llama(base);
    llama_model * model = llama_model_load_from_file(base.model.path.c_str(), mparams);
    if (model == nullptr) {
        SRV_ERR("failed to load model '%s'\n", base.model.path.c_str());
        llama_backend_free();
        return 1;
    }

    // Run both roles against the borrowed model. All server_context / server_http_context /
    // server_routes live INSIDE run_dual_roles and are destroyed when it returns, BEFORE we free
    // the model below - see the lifetime note on run_dual_roles.
    const int rc = run_dual_roles(model, p_orch, p_worker);

    // Free the shared model exactly once, after both server_context have been destroyed (they
    // borrowed it and never free it - the T1 owns_model=false contract).
    llama_model_free(model);
    llama_backend_free();

    return rc;
}
