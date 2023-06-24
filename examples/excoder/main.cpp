#include "ggml/ggml.h"

#include "common.h"
#include "excoder.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <vector>
#include <iostream>

#ifdef _MSC_VER

// https://stackoverflow.com/questions/74166084/what-is-the-windows-equivalent-of-posix-unistd-h-macro-stdin-fileno
// If you are using the posix emulated APIs on Windows, such as _read and _write, you can safely pass 0, 1, and 2
// as stdin, stdout, and stderr file ids. Define your own macos as needed.

/* Standard file descriptors.  */
#define STDIN_FILENO 0  /* Standard input.  */
#define STDOUT_FILENO 1 /* Standard output.  */
#define STDERR_FILENO 2 /* Standard error output.  */

#include <io.h>

#else
#include <unistd.h>
#endif

int main(int argc, char **argv)
{
    ggml_time_init();

    const int64_t t_main_start_us = ggml_time_us();

    gpt_params params;
    params.model = "models/bigcode/gpt_bigcode-santacoder-ggml.bin";

    if (gpt_params_parse(argc, argv, params) == false)
    {
        return 1;
    }

    if (params.seed < 0)
    {
        params.seed = time(NULL);
    }

    printf("%s: seed = %d\n", __func__, params.seed);

    std::mt19937 rng(params.seed);
    if (params.prompt.empty())
    {
        if (!isatty(STDIN_FILENO))
        {
            std::string line;
            while (std::getline(std::cin, line))
            {
                params.prompt = params.prompt + "\n" + line;
            }
        }
        else
        {
            params.prompt = gpt_random_prompt(rng);
        }
    }

    int64_t t_load_us = 0;

    gpt_vocab vocab;
    santacoder_model model;

    // load the model
    {
        const int64_t t_start_us = ggml_time_us();

        if (!santacoder_model_load(params.model, model, vocab))
        {
            fprintf(stderr, "%s: failed to load model from '%s'\n", __func__, params.model.c_str());
            return 1;
        }

        t_load_us = ggml_time_us() - t_start_us;
    }

    int n_past = 0;

    int64_t t_sample_us = 0;
    int64_t t_predict_us = 0;

    std::vector<float> logits;

    // tokenize the prompt
    std::vector<gpt_vocab::id> embd_inp = ::gpt_tokenize(vocab, params.prompt);

    params.n_predict = std::min(params.n_predict, model.hparams.n_ctx - (int)embd_inp.size());

    printf("%s: prompt: '%s'\n", __func__, params.prompt.c_str());
    printf("%s: number of tokens in prompt = %zu, first 8 tokens: ", __func__, embd_inp.size());
    for (int i = 0; i < std::min(8, (int)embd_inp.size()); i++)
    {
        printf("%d ", embd_inp[i]);
    }
    printf("\n\n");

    // submit the input prompt token-by-token
    // this reduces the memory usage during inference, at the cost of a bit of speed at the beginning
    std::vector<gpt_vocab::id> embd;

    // determine the required inference memory per token:
    size_t mem_per_token = 0;
    santacoder_eval(model, params.n_threads, 0, {0, 1, 2, 3}, logits, mem_per_token);

    for (int i = embd.size(); i < embd_inp.size() + params.n_predict; i++)
    {
        // predict
        if (embd.size() > 0)
        {
            const int64_t t_start_us = ggml_time_us();

            if (!santacoder_eval(model, params.n_threads, n_past, embd, logits, mem_per_token))
            {
                printf("Failed to predict\n");
                return 1;
            }

            t_predict_us += ggml_time_us() - t_start_us;
        }

        n_past += embd.size();
        embd.clear();

        if (i >= embd_inp.size())
        {
            // sample next token
            const int top_k = params.top_k;
            const float top_p = params.top_p;
            const float temp = params.temp;

            const int n_vocab = model.hparams.n_vocab;

            gpt_vocab::id id = 0;

            {
                const int64_t t_start_sample_us = ggml_time_us();

                id = gpt_sample_top_k_top_p(vocab, logits.data() + (logits.size() - n_vocab), top_k, top_p, temp, rng);

                t_sample_us += ggml_time_us() - t_start_sample_us;
            }

            // add it to the context
            embd.push_back(id);
        }
        else
        {
            // if here, it means we are still processing the input prompt
            for (int k = i; k < embd_inp.size(); k++)
            {
                embd.push_back(embd_inp[k]);
                if (embd.size() >= params.n_batch)
                {
                    break;
                }
            }
            i += embd.size() - 1;
        }

        // display text
        for (auto id : embd)
        {
            printf("%s", vocab.id_to_token[id].c_str());
        }
        fflush(stdout);

        // check if model is santacoder
        if (model.hparams.n_layer <= 30 && embd.back() == 49152)
        {
            break;
        }
        // check if model is santacoder
        else if (embd.back() == 0)
        { // TODO: this is only for santacoder
            break;
        }
    }

    // report timing
    {
        const int64_t t_main_end_us = ggml_time_us();

        printf("\n\n");
        printf("%s: mem per token = %8zu bytes\n", __func__, mem_per_token);
        printf("%s:     load time = %8.2f ms\n", __func__, t_load_us / 1000.0f);
        printf("%s:   sample time = %8.2f ms\n", __func__, t_sample_us / 1000.0f);
        printf("%s:  predict time = %8.2f ms / %.2f ms per token\n", __func__, t_predict_us / 1000.0f, t_predict_us / 1000.0f / n_past);
        printf("%s:    total time = %8.2f ms\n", __func__, (t_main_end_us - t_main_start_us) / 1000.0f);
    }

    ggml_free(model.ctx);

    return 0;
}
