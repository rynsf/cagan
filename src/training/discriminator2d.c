#include "discriminator2d.h"

#include "linalg.h"

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

Discriminator2D* discriminator2d_create(int class_dim, int in_hw, int hidden_c) {
    Discriminator2D* d = (Discriminator2D*)malloc(sizeof(Discriminator2D));
    d->class_dim = class_dim;
    d->in_hw = in_hw;
    d->hidden_c = hidden_c;
    d->from_rgb = conv2d_create(3, hidden_c, 3, 1);
    d->attn = attention2d_create(hidden_c, hidden_c / 2);
    d->conv2 = conv2d_create(hidden_c, hidden_c, 3, 1);
    d->class_proj = linear_create(class_dim, hidden_c);
    d->head = linear_create(hidden_c, 1);
    d->pooled_cache = NULL;
    d->class_oh_cache = NULL;
    return d;
}

void discriminator2d_free(Discriminator2D* d) {
    if (!d) return;
    conv2d_free(d->from_rgb);
    attention2d_free(d->attn);
    conv2d_free(d->conv2);
    linear_free(d->class_proj);
    linear_free(d->head);
    if (d->pooled_cache) tt_free(d->pooled_cache);
    if (d->class_oh_cache) tt_free(d->class_oh_cache);
    free(d);
}

TensorTrain* discriminator2d_forward(Discriminator2D* d, TensorTrain* img, const int* class_ids) {
    if (d->class_oh_cache) tt_free(d->class_oh_cache);
    d->class_oh_cache = one_hot(img->n, d->class_dim, class_ids);

    TensorTrain* h = conv2d_forward(d->from_rgb, img);
    leaky_relu_forward(h->data, tt_numel(h), 0.2f);
    TensorTrain* ha = attention2d_forward(d->attn, h);
    TensorTrain* h2 = conv2d_forward(d->conv2, ha);
    leaky_relu_forward(h2->data, tt_numel(h2), 0.2f);

    if (d->pooled_cache) tt_free(d->pooled_cache);
    d->pooled_cache = tt_create(img->n, 1, 1, d->hidden_c);
    int hw = h2->h * h2->w;
    for (int n = 0; n < h2->n; n++) {
        for (int c = 0; c < h2->c; c++) {
            float s = 0.0f;
            for (int i = 0; i < hw; i++) s += h2->data[((n * h2->c + c) * hw) + i];
            d->pooled_cache->data[n * h2->c + c] = s / (float)hw;
        }
    }

    TensorTrain* cp = linear_forward(d->class_proj, d->class_oh_cache);
    for (int i = 0; i < tt_numel(d->pooled_cache); i++) d->pooled_cache->data[i] += cp->data[i];
    return linear_forward(d->head, d->pooled_cache);
}

TensorTrain* discriminator2d_backward(Discriminator2D* d, TensorTrain* dlogits) {
    TensorTrain* dpool = linear_backward(d->head, dlogits);
    TensorTrain* dcp = linear_backward(d->class_proj, dpool);
    tt_free(dcp);

    int n = d->pooled_cache->n;
    int c = d->hidden_c;
    int h = d->conv2->y_cache->h;
    int w = d->conv2->y_cache->w;
    int hw = h * w;

    TensorTrain* dh2 = tt_create(n, c, h, w);
    for (int b = 0; b < n; b++) {
        for (int ch = 0; ch < c; ch++) {
            float g = dpool->data[b * c + ch] / (float)hw;
            for (int i = 0; i < hw; i++) dh2->data[((b * c + ch) * hw) + i] = g;
        }
    }
    tt_free(dpool);

    TensorTrain* dha = conv2d_backward(d->conv2, dh2);
    tt_free(dh2);
    TensorTrain* dh = attention2d_backward(d->attn, dha);
    tt_free(dha);
    TensorTrain* dimg = conv2d_backward(d->from_rgb, dh);
    tt_free(dh);
    return dimg;
}

void discriminator2d_step(Discriminator2D* d, float lr, int t) {
    conv2d_step(d->from_rgb, lr, t);
    attention2d_step(d->attn, lr, t);
    conv2d_step(d->conv2, lr, t);
    linear_step(d->class_proj, lr, t);
    linear_step(d->head, lr, t);
}
