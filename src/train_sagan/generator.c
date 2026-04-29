#include "generator.h"

#include "linalg.h"

#include <math.h>
#include <stdlib.h>

static TensorTrain* one_hot_classes(int batch, int class_dim, const int* ids) {
    TensorTrain* t = tt_create(batch, 1, 1, class_dim);
    for (int i = 0; i < batch; i++) {
        int cid = ids[i];
        if (cid < 0) cid = 0;
        if (cid >= class_dim) cid = class_dim - 1;
        t->data[i * class_dim + cid] = 1.0f;
    }
    return t;
}

Generator* generator_create(int z_dim, int class_dim, int hidden, int out_dim) {
    Generator* g = (Generator*)malloc(sizeof(Generator));
    g->z_dim = z_dim;
    g->class_dim = class_dim;
    g->hidden = hidden;
    g->out_dim = out_dim;

    g->z_proj = linear_create(z_dim, hidden);
    g->class_embed = linear_create(class_dim, hidden);
    g->fuse = linear_create(hidden, hidden);
    g->attn = attention_create(hidden, hidden / 2);
    g->to_img = linear_create(hidden, out_dim);

    g->z_cache = NULL;
    g->class_cache = NULL;
    g->hidden_cache = NULL;
    g->out_cache = NULL;
    return g;
}

void generator_free(Generator* g) {
    if (!g) return;
    linear_free(g->z_proj);
    linear_free(g->class_embed);
    linear_free(g->fuse);
    attention_free(g->attn);
    linear_free(g->to_img);

    if (g->class_cache) tt_free(g->class_cache);
    free(g);
}

TensorTrain* generator_forward(Generator* g, TensorTrain* z, const int* class_ids) {
    // 1) Condition path: class ids -> one-hot -> embedding projection.
    if (g->class_cache) tt_free(g->class_cache);
    g->class_cache = one_hot_classes(z->n, g->class_dim, class_ids);

    TensorTrain* z_h = linear_forward(g->z_proj, z);
    TensorTrain* c_h = linear_forward(g->class_embed, g->class_cache);

    // 2) Feature fusion acts like class-conditional modulation.
    TensorTrain* fused = tt_create(z->n, 1, 1, g->hidden);
    for (int i = 0; i < z->n * g->hidden; i++) fused->data[i] = z_h->data[i] + c_h->data[i];
    relu_forward(fused->data, z->n * g->hidden);

    TensorTrain* h = linear_forward(g->fuse, fused);
    relu_forward(h->data, z->n * g->hidden);
    tt_free(fused);

    // 3) Self-attention allows each sample in batch to exchange global information.
    // In a full SAGAN image model, attention is per spatial token. Here it is conceptually equivalent.
    TensorTrain* h_attn = attention_forward(g->attn, h);
    relu_forward(h_attn->data, z->n * g->hidden);

    // 4) Project to flattened image vector (e.g. 3*128*128 values).
    TensorTrain* out = linear_forward(g->to_img, h_attn);
    for (int i = 0; i < z->n * g->out_dim; i++) {
        // Output domain is [-1, 1] just like image GANs using tanh.
        float x = out->data[i];
        out->data[i] = (2.0f / (1.0f + expf(-2.0f * x))) - 1.0f;
    }

    g->z_cache = z;
    g->hidden_cache = h_attn;
    g->out_cache = out;
    return out;
}

TensorTrain* generator_backward(Generator* g, TensorTrain* dimg) {
    int total = dimg->n * dimg->w;
    TensorTrain* dtanh = tt_create(dimg->n, 1, 1, dimg->w);
    for (int i = 0; i < total; i++) {
        float y = g->out_cache->data[i];
        dtanh->data[i] = dimg->data[i] * (1.0f - y * y);
    }

    TensorTrain* dh_attn = linear_backward(g->to_img, dtanh);
    tt_free(dtanh);

    TensorTrain* dh = attention_backward(g->attn, dh_attn);
    tt_free(dh_attn);

    TensorTrain* dfuse = linear_backward(g->fuse, dh);
    tt_free(dh);

    // Split gradient to z and class branch through additive fusion.
    TensorTrain* dz_h = tt_create(dfuse->n, 1, 1, dfuse->w);
    TensorTrain* dc_h = tt_create(dfuse->n, 1, 1, dfuse->w);
    for (int i = 0; i < dfuse->n * dfuse->w; i++) {
        dz_h->data[i] = dfuse->data[i];
        dc_h->data[i] = dfuse->data[i];
    }
    tt_free(dfuse);

    TensorTrain* dz = linear_backward(g->z_proj, dz_h);
    TensorTrain* dclass = linear_backward(g->class_embed, dc_h);
    tt_free(dz_h);
    tt_free(dc_h);
    tt_free(dclass);
    return dz;
}

void generator_step(Generator* g, float lr, int step) {
    linear_step(g->z_proj, lr, step);
    linear_step(g->class_embed, lr, step);
    linear_step(g->fuse, lr, step);
    attention_step(g->attn, lr, step);
    linear_step(g->to_img, lr, step);
}
