#include "layers.h"

#include "linalg.h"

#include <math.h>
#include <stdlib.h>

LinearLayer* linear_create(int in_features, int out_features) {
    LinearLayer* l = (LinearLayer*)malloc(sizeof(LinearLayer));
    l->x_cache = NULL;
    l->y = NULL;

    l->w = tt_create(1, out_features, 1, in_features);
    l->b = tt_create(1, 1, 1, out_features);
    tt_fill_randn(l->w, 0.02f);
    tt_zero(l->b);

    l->m_w = tt_create(1, out_features, 1, in_features);
    l->v_w = tt_create(1, out_features, 1, in_features);
    l->m_b = tt_create(1, 1, 1, out_features);
    l->v_b = tt_create(1, 1, 1, out_features);
    return l;
}

void linear_free(LinearLayer* layer) {
    if (!layer) return;
    if (layer->y) tt_free(layer->y);
    tt_free(layer->w); tt_free(layer->b);
    tt_free(layer->m_w); tt_free(layer->v_w);
    tt_free(layer->m_b); tt_free(layer->v_b);
    free(layer);
}

TensorTrain* linear_forward(LinearLayer* layer, TensorTrain* x) {
    layer->x_cache = x;
    if (layer->y) tt_free(layer->y);
    layer->y = tt_create(x->n, 1, 1, layer->w->c);

    matmul_forward(x->data, layer->w->data, layer->y->data, x->n, x->w, layer->w->c);
    add_bias_forward(layer->y->data, layer->b->data, x->n, layer->w->c);
    return layer->y;
}

TensorTrain* linear_backward(LinearLayer* layer, TensorTrain* dy) {
    TensorTrain* dx = tt_create(layer->x_cache->n, 1, 1, layer->x_cache->w);
    matmul_backward(layer->x_cache->data, layer->w->data, dy->data,
                    dx->data, layer->w->grad,
                    layer->x_cache->n, layer->x_cache->w, layer->w->c);
    add_bias_backward(dy->data, layer->b->grad, layer->x_cache->n, layer->w->c);
    return dx;
}

void linear_step(LinearLayer* layer, float lr, int t) {
    int w_size = tt_numel(layer->w);
    int b_size = tt_numel(layer->b);
    adam_update(layer->w->data, layer->w->grad, layer->m_w->data, layer->v_w->data,
                w_size, lr, 0.0f, 0.9f, 1e-8f, t);
    adam_update(layer->b->data, layer->b->grad, layer->m_b->data, layer->v_b->data,
                b_size, lr, 0.0f, 0.9f, 1e-8f, t);
}

SelfAttention* attention_create(int dim, int attn_dim) {
    SelfAttention* a = (SelfAttention*)malloc(sizeof(SelfAttention));
    a->w_q = tt_create(1, dim, 1, attn_dim);
    a->w_k = tt_create(1, dim, 1, attn_dim);
    a->w_v = tt_create(1, dim, 1, attn_dim);
    a->w_o = tt_create(1, attn_dim, 1, dim);
    tt_fill_randn(a->w_q, 0.02f);
    tt_fill_randn(a->w_k, 0.02f);
    tt_fill_randn(a->w_v, 0.02f);
    tt_fill_randn(a->w_o, 0.02f);

    a->m_w_q = tt_create(1, dim, 1, attn_dim);
    a->v_w_q = tt_create(1, dim, 1, attn_dim);
    a->m_w_k = tt_create(1, dim, 1, attn_dim);
    a->v_w_k = tt_create(1, dim, 1, attn_dim);
    a->m_w_v = tt_create(1, dim, 1, attn_dim);
    a->v_w_v = tt_create(1, dim, 1, attn_dim);
    a->m_w_o = tt_create(1, attn_dim, 1, dim);
    a->v_w_o = tt_create(1, attn_dim, 1, dim);

    a->x_cache = NULL;
    a->q_cache = NULL;
    a->k_cache = NULL;
    a->v_cache = NULL;
    a->a_cache = NULL;
    a->y_cache = NULL;
    return a;
}

void attention_free(SelfAttention* a) {
    if (!a) return;
    tt_free(a->w_q); tt_free(a->w_k); tt_free(a->w_v); tt_free(a->w_o);
    tt_free(a->m_w_q); tt_free(a->v_w_q);
    tt_free(a->m_w_k); tt_free(a->v_w_k);
    tt_free(a->m_w_v); tt_free(a->v_w_v);
    tt_free(a->m_w_o); tt_free(a->v_w_o);
    if (a->q_cache) tt_free(a->q_cache);
    if (a->k_cache) tt_free(a->k_cache);
    if (a->v_cache) tt_free(a->v_cache);
    if (a->a_cache) tt_free(a->a_cache);
    if (a->y_cache) tt_free(a->y_cache);
    free(a);
}

TensorTrain* attention_forward(SelfAttention* a, TensorTrain* x) {
    a->x_cache = x;
    int b = x->n;
    int dim = x->w;
    int ad = a->w_q->w;

    if (a->q_cache) tt_free(a->q_cache);
    if (a->k_cache) tt_free(a->k_cache);
    if (a->v_cache) tt_free(a->v_cache);
    if (a->a_cache) tt_free(a->a_cache);
    if (a->y_cache) tt_free(a->y_cache);

    a->q_cache = tt_create(b, 1, 1, ad);
    a->k_cache = tt_create(b, 1, 1, ad);
    a->v_cache = tt_create(b, 1, 1, ad);

    matmul_forward(x->data, a->w_q->data, a->q_cache->data, b, dim, ad);
    matmul_forward(x->data, a->w_k->data, a->k_cache->data, b, dim, ad);
    matmul_forward(x->data, a->w_v->data, a->v_cache->data, b, dim, ad);

    a->a_cache = tt_create(b, 1, 1, b);
    for (int i = 0; i < b; i++) {
        for (int j = 0; j < b; j++) {
            float s = 0.0f;
            for (int k = 0; k < ad; k++) s += a->q_cache->data[i * ad + k] * a->k_cache->data[j * ad + k];
            a->a_cache->data[i * b + j] = s / sqrtf((float)ad);
        }
    }
    softmax_forward_rows(a->a_cache->data, b, b);

    TensorTrain* context = tt_create(b, 1, 1, ad);
    matmul_forward(a->a_cache->data, a->v_cache->data, context->data, b, b, ad);

    a->y_cache = tt_create(b, 1, 1, dim);
    matmul_forward(context->data, a->w_o->data, a->y_cache->data, b, ad, dim);
    for (int i = 0; i < b * dim; i++) a->y_cache->data[i] += x->data[i];
    tt_free(context);
    return a->y_cache;
}

TensorTrain* attention_backward(SelfAttention* a, TensorTrain* dy) {
    // Full gradient for attention is long; here we implement key pieces so concepts are explicit.
    int b = a->x_cache->n;
    int dim = a->x_cache->w;
    TensorTrain* dx = tt_create(b, 1, 1, dim);
    for (int i = 0; i < b * dim; i++) dx->data[i] = dy->data[i]; // residual path

    // A complete backward would continue through W_o, V, softmax, Q, K and all matmuls.
    // This scaffold keeps control points and tensors ready for full derivation extensions.
    return dx;
}

void attention_step(SelfAttention* a, float lr, int t) {
    int sz_q = tt_numel(a->w_q);
    int sz_k = tt_numel(a->w_k);
    int sz_v = tt_numel(a->w_v);
    int sz_o = tt_numel(a->w_o);
    adam_update(a->w_q->data, a->w_q->grad, a->m_w_q->data, a->v_w_q->data, sz_q, lr, 0.0f, 0.9f, 1e-8f, t);
    adam_update(a->w_k->data, a->w_k->grad, a->m_w_k->data, a->v_w_k->data, sz_k, lr, 0.0f, 0.9f, 1e-8f, t);
    adam_update(a->w_v->data, a->w_v->grad, a->m_w_v->data, a->v_w_v->data, sz_v, lr, 0.0f, 0.9f, 1e-8f, t);
    adam_update(a->w_o->data, a->w_o->grad, a->m_w_o->data, a->v_w_o->data, sz_o, lr, 0.0f, 0.9f, 1e-8f, t);
}
