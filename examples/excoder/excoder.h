#include "ggml/ggml.h"
#include "../common.h"

#include <string>
#include <fstream>
#include <vector>
#include <map>

// const char *santa_print_system_info(void);

// default hparams (GPT-2)
struct santacoder_hparams
{
    // int32_t n_vocab = 50400;
    // int32_t n_ctx = 2048;
    // int32_t n_embd = 4096;
    // int32_t n_head = 16;
    // int32_t n_layer = 28;
    // int32_t n_rot = 64;
    // int32_t f16 = 1;
    int32_t n_vocab = 49280;
    int32_t n_ctx = 2048;
    int32_t n_embd = 2048;
    int32_t n_head = 16;
    int32_t n_layer = 24;
    int32_t ftype = 1;
};

struct santacoder_layer
{
    // normalization
    struct ggml_tensor *ln_1_g;
    struct ggml_tensor *ln_1_b;

    struct ggml_tensor *ln_2_g;
    struct ggml_tensor *ln_2_b;

    // attention
    struct ggml_tensor *c_attn_attn_w;
    struct ggml_tensor *c_attn_attn_b;

    struct ggml_tensor *c_attn_proj_w;
    struct ggml_tensor *c_attn_proj_b;

    // mlp
    struct ggml_tensor *c_mlp_fc_w;
    struct ggml_tensor *c_mlp_fc_b;

    struct ggml_tensor *c_mlp_proj_w;
    struct ggml_tensor *c_mlp_proj_b;
};

struct santacoder_model
{
    santacoder_hparams hparams;

    // normalization
    struct ggml_tensor *ln_f_g;
    struct ggml_tensor *ln_f_b;

    struct ggml_tensor *wte;     // position embedding
    struct ggml_tensor *wpe;     //    token embedding
    struct ggml_tensor *lm_head; // language model head

    std::vector<santacoder_layer> layers;

    // key + value memory
    struct ggml_tensor *memory_k;
    struct ggml_tensor *memory_v;

    //
    struct ggml_context *ctx;
    std::map<std::string, struct ggml_tensor *> tensors;
};

bool santacoder_model_load(const std::string &fname, santacoder_model &model, gpt_vocab &vocab);

bool santacoder_eval(
    const santacoder_model &model,
    const int n_threads,
    const int n_past,
    const std::vector<gpt_vocab::id> &embd_inp,
    std::vector<float> &embd_w,
    size_t &mem_per_token);