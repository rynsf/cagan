#include "generator2d.h"

#include "linalg.h"

#include <math.h>
#include <stdlib.h>

static TensorTrain* one_hot(int n, int class_dim, const int* ids) {
    TensorTrain* t = tt_create(n, 1, 1, class_dim);
    for (int i = 0; i < n; i++) {
        int c = ids[i];
        if (c < 0) c = 0;
        if (c >= class_dim) c = class_dim - 1;
        t->data[i * class_dim + c] = 1.0f;
    }
    return t;
}

Generator2D* generator2d_create(int z_dim, int class_dim, int base_c, int base_hw) {
    Generator2D* g = (Generator2D*)malloc(sizeof(Generator2D));
    g->z_dim = z_dim;
    g->class_dim = class_dim;
    g->base_c = base_c;
    g->base_hw = base_hw;
    int flat = base_c * base_hw * base_hw;
    g->z_to_feat = linear_create(z_dim, flat);
    g->cls_to_feat = linear_create(class_dim, flat);
    g->conv1 = conv2d_create(base_c, base_c, 3, 1);
    g->attn = attention2d_create(base_c, base_c / 2);
    g->to_rgb = conv2d_create(base_c, 3, 3, 1);
    g->feat_cache = g->up_cache = g->out_cache = g->class_oh_cache = NULL;
    return g;
}

void generator2d_free(Generator2D* g) {
    if (!g) return;
    linear_free(g->z_to_feat);
    linear_free(g->cls_to_feat);
    conv2d_free(g->conv1);
    attention2d_free(g->attn);
    conv2d_free(g->to_rgb);
    if (g->class_oh_cache) tt_free(g->class_oh_cache);
    free(g);
}

TensorTrain* generator2d_forward(Generator2D* g, TensorTrain* z, const int* class_ids) {
    if (g->class_oh_cache) tt_free(g->class_oh_cache);
    g->class_oh_cache = one_hot(z->n, g->class_dim, class_ids);

    TensorTrain* zf = linear_forward(g->z_to_feat, z);
    TensorTrain* cf = linear_forward(g->cls_to_feat, g->class_oh_cache);

    if (g->feat_cache) tt_free(g->feat_cache);
    g->feat_cache = tt_create(z->n, g->base_c, g->base_hw, g->base_hw);
    for (int i = 0; i < tt_numel(g->feat_cache); i++) g->feat_cache->data[i] = zf->data[i] + cf->data[i];
    relu_forward(g->feat_cache->data, tt_numel(g->feat_cache));

    TensorTrain* up = upsample2x_forward(g->feat_cache);
    if (g->up_cache) tt_free(g->up_cache);
    g->up_cache = up;

    TensorTrain* h = conv2d_forward(g->conv1, up);
    relu_forward(h->data, tt_numel(h));

    TensorTrain* h_attn = attention2d_forward(g->attn, h);
    TensorTrain* rgb = conv2d_forward(g->to_rgb, h_attn);

    int total = tt_numel(rgb);
    for (int i = 0; i < total; i++) rgb->data[i] = tanhf(rgb->data[i]);
    g->out_cache = rgb;
    return rgb;
}

TensorTrain* generator2d_backward(Generator2D* g, TensorTrain* dimg) {
    TensorTrain* dtanh = tt_create(dimg->n, dimg->c, dimg->h, dimg->w);
    int total = tt_numel(dimg);
    for (int i = 0; i < total; i++) {
        float y = g->out_cache->data[i];
        dtanh->data[i] = dimg->data[i] * (1.0f - y * y);
    }

    TensorTrain* dh_attn = conv2d_backward(g->to_rgb, dtanh);
    tt_free(dtanh);
    TensorTrain* dh = attention2d_backward(g->attn, dh_attn);
    tt_free(dh_attn);
    TensorTrain* dup = conv2d_backward(g->conv1, dh);
    tt_free(dh);

    TensorTrain* dfeat = upsample2x_backward(dup);
    tt_free(dup);

    TensorTrain* dflat = tt_create(dfeat->n, 1, 1, dfeat->c * dfeat->h * dfeat->w);
    for (int i = 0; i < tt_numel(dflat); i++) dflat->data[i] = dfeat->data[i];
    tt_free(dfeat);

    TensorTrain* dz = linear_backward(g->z_to_feat, dflat);
    TensorTrain* dcls = linear_backward(g->cls_to_feat, dflat);
    tt_free(dflat);
    tt_free(dcls);
    return dz;
}

void generator2d_step(Generator2D* g, float lr, int t) {
    linear_step(g->z_to_feat, lr, t);
    linear_step(g->cls_to_feat, lr, t);
    conv2d_step(g->conv1, lr, t);
    attention2d_step(g->attn, lr, t);
    conv2d_step(g->to_rgb, lr, t);
}
