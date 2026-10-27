#include "llm.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <thread>
#include <vector>

#include "json-schema-to-grammar.h"
#include "llama.h"

namespace parrot {

Llm::Llm(const std::string& path, int context, int threads) {
    if (!std::getenv("PARROT_VERBOSE")) llama_log_set([](ggml_log_level, const char*, void*) {}, nullptr);
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    // The default load mode maps the file: the weights stay in the page cache, not the app's heap.
    model_ = llama_model_load_from_file(path.c_str(), mp);
    if (!model_) throw std::runtime_error("cannot load the model " + path);
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = static_cast<uint32_t>(context);
    cp.n_batch = 512;
    int n = threads > 0 ? threads : std::max(2, static_cast<int>(std::thread::hardware_concurrency()) - 2);
    cp.n_threads = n;
    cp.n_threads_batch = n;
    ctx_ = llama_init_from_model(model_, cp);
    if (!ctx_) {
        llama_model_free(model_);
        throw std::runtime_error("cannot create a context for the model");
    }
    name_ = std::filesystem::path(path).filename().string();
}

Llm::~Llm() {
    if (ctx_) llama_free(ctx_);
    if (model_) llama_model_free(model_);
}

std::string Llm::complete(const std::string& system, const std::string& user, const std::string& json_schema, int max_tokens) {
    const llama_vocab* vocab = llama_model_get_vocab(model_);

    // The model's own chat template (ChatML for Qwen).
    llama_chat_message msgs[] = {{"system", system.c_str()}, {"user", user.c_str()}};
    const char* tmpl = llama_model_chat_template(model_, nullptr);
    std::vector<char> buf(system.size() + user.size() + 1024);
    int len = llama_chat_apply_template(tmpl, msgs, 2, true, buf.data(), static_cast<int32_t>(buf.size()));
    if (len > static_cast<int>(buf.size())) {
        buf.resize(len + 1);
        len = llama_chat_apply_template(tmpl, msgs, 2, true, buf.data(), static_cast<int32_t>(buf.size()));
    }
    if (len < 0) throw std::runtime_error("the model has no usable chat template");
    std::string prompt(buf.data(), len);

    std::vector<llama_token> tokens(prompt.size() + 16);
    int n = llama_tokenize(vocab, prompt.c_str(), static_cast<int32_t>(prompt.size()), tokens.data(), static_cast<int32_t>(tokens.size()), true, true);
    if (n < 0) throw std::runtime_error("tokenizing the prompt failed");
    tokens.resize(n);
    if (n + max_tokens > static_cast<int>(llama_n_ctx(ctx_))) {
        throw std::runtime_error("the document is too long for the model's context");
    }

    // Greedy (deterministic, like temperature 0), constrained by the schema's grammar. Checking the
    // grammar against the whole vocabulary (150k tokens) at every step is slow; instead the greedy
    // token is checked alone, and the vocabulary filtered only when the grammar rejects it.
    std::unique_ptr<llama_sampler, void (*)(llama_sampler*)> owned(nullptr, llama_sampler_free);
    if (!json_schema.empty()) {
        std::string g = json_schema_to_grammar(common_json::parse(json_schema));
        owned.reset(llama_sampler_init_grammar(vocab, g.c_str(), "root"));
        if (!owned) throw std::runtime_error("the JSON schema did not make a grammar");
    }
    llama_sampler* grammar = owned.get();
    const int n_vocab = llama_vocab_n_tokens(vocab);
    std::vector<llama_token_data> cand(n_vocab);
    auto next_token = [&]() -> llama_token {
        const float* logits = llama_get_logits_ith(ctx_, -1);
        llama_token best = 0;
        for (llama_token v = 1; v < n_vocab; ++v)
            if (logits[v] > logits[best]) best = v;
        if (!grammar) return best;
        llama_token_data one = {best, logits[best], 0.f};
        llama_token_data_array single = {&one, 1, -1, false};
        llama_sampler_apply(grammar, &single);
        if (single.data[0].logit != -INFINITY) return best;
        for (llama_token v = 0; v < n_vocab; ++v) cand[v] = {v, logits[v], 0.f};
        llama_token_data_array all = {cand.data(), cand.size(), -1, false};
        llama_sampler_apply(grammar, &all);
        best = -1;
        for (size_t i = 0; i < all.size; ++i)
            if (all.data[i].logit != -INFINITY && (best < 0 || all.data[i].logit > cand[best].logit)) best = all.data[i].id;
        if (best < 0) throw std::runtime_error("the grammar allows no token");
        return best;
    };

    // Reuse what the memory already holds of this prompt; keep at least the last token to decode,
    // for its logits.
    size_t common = 0;
    while (common < cached_.size() && common < tokens.size() && cached_[common] == tokens[common]) ++common;
    if (common == tokens.size()) --common;
    llama_memory_t mem = llama_get_memory(ctx_);
    if (!llama_memory_seq_rm(mem, 0, static_cast<llama_pos>(common), -1)) {
        llama_memory_clear(mem, true);
        common = 0;
    }
    cached_.assign(tokens.begin(), tokens.begin() + static_cast<long>(common));
    std::string out;
    // The rest of the prompt in batches, then one token at a time.
    for (int i = static_cast<int>(common); i < n; i += 512) {
        int k = std::min(512, n - i);
        if (llama_decode(ctx_, llama_batch_get_one(tokens.data() + i, k)) != 0) {
            cached_.clear();
            llama_memory_clear(mem, true);
            throw std::runtime_error("decoding the prompt failed");
        }
        cached_.insert(cached_.end(), tokens.begin() + i, tokens.begin() + i + k);
    }
    for (int i = 0; i < max_tokens; ++i) {
        llama_token t = next_token();
        if (grammar) llama_sampler_accept(grammar, t);
        if (llama_vocab_is_eog(vocab, t)) break;
        char piece[256];
        int m = llama_token_to_piece(vocab, t, piece, sizeof piece, 0, true);
        if (m > 0) out.append(piece, m);
        if (llama_decode(ctx_, llama_batch_get_one(&t, 1)) != 0) break;
        cached_.push_back(t);
    }
    return out;
}

}  // namespace parrot
