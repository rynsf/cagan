#include "attention2d.h"

#include <math.h>
#include <stdlib.h>

static void softmax_rows(float* a, int rows, int cols) {
    for (int r = 0; r < rows; r++) {
        float* row = &a[r * cols];
        float mx = row[0];
        for (int i = 1; i < cols; i++) if (row[i] > mx) mx = row[i];
        float s = 0.0f;
        for (int i = 0; i < cols; i++) { row[i] = expf(row[i] - mx); s += row[i]; }
        float inv = 1.0f / (s + 1e-12f);
        for (int i = 0; i < cols; i++) row[i] *= inv;
    }
}

Attention2D* attention2d_create(int c, int c_attn) {
    Attention2D* a = (Attention2D*)malloc(sizeof(Attention2D));
    a->c = c;
    a->c_attn = c_attn;
    a->theta = conv2d_create(c, c_attn, 1, 0);
    a->phi = conv2d_create(c, c_attn, 1, 0);
    a->g = conv2d_create(c, c_attn, 1, 0);
    a->o = conv2d_create(c_attn, c, 1, 0);
    a->gamma = 0.0f;
    a->gamma_grad = 0.0f;
    a->gamma_m = 0.0f;
    a->gamma_v = 0.0f;
    a->x_cache = a->theta_cache = a->phi_cache = a->g_cache = NULL;
    a->attn_cache = a->out_g_cache = a->y_cache = NULL;
    return a;
}

void attention2d_free(Attention2D* a) {
    if (!a) return;
    conv2d_free(a->theta); conv2d_free(a->phi); conv2d_free(a->g); conv2d_free(a->o);
    if (a->attn_cache) tt_free(a->attn_cache);
    if (a->out_g_cache) tt_free(a->out_g_cache);
    if (a->y_cache) tt_free(a->y_cache);
    free(a);
}

TensorTrain* attention2d_forward(Attention2D* a, TensorTrain* x) {
    a->x_cache = x;
    a->theta_cache = conv2d_forward(a->theta, x);
    a->phi_cache = conv2d_forward(a->phi, x);
    a->g_cache = conv2d_forward(a->g, x);

    int n = x->n;
    int hw = x->h * x->w;
    int ca = a->c_attn;

    if (a->attn_cache) tt_free(a->attn_cache);
    if (a->out_g_cache) tt_free(a->out_g_cache);
    if (a->y_cache) tt_free(a->y_cache);

    a->attn_cache = tt_create(n, 1, hw, hw);
    for (int b = 0; b < n; b++) {
        for (int i = 0; i < hw; i++) {
            for (int j = 0; j < hw; j++) {
                float s = 0.0f;
                for (int c = 0; c < ca; c++) {
                    int iy = i / x->w, ix = i % x->w;
                    int jy = j / x->w, jx = j % x->w;
                    s += TT_AT(a->theta_cache, b, c, iy, ix) * TT_AT(a->phi_cache, b, c, jy, jx);
                }
                TT_AT(a->attn_cache, b, 0, i, j) = s / sqrtf((float)ca);
            }
        }
        softmax_rows(&a->attn_cache->data[b * hw * hw], hw, hw);
    }

    a->out_g_cache = tt_create(n, ca, x->h, x->w);
    for (int b = 0; b < n; b++) {
        for (int c = 0; c < ca; c++) {
            for (int i = 0; i < hw; i++) {
                float sum = 0.0f;
                for (int j = 0; j < hw; j++) {
                    int jy = j / x->w, jx = j % x->w;
                    sum += TT_AT(a->attn_cache, b, 0, i, j) * TT_AT(a->g_cache, b, c, jy, jx);
                }
                TT_AT(a->out_g_cache, b, c, i / x->w, i % x->w) = sum;
            }
        }
    }

    TensorTrain* o = conv2d_forward(a->o, a->out_g_cache);
    a->y_cache = tt_create(n, x->c, x->h, x->w);
    int total = tt_numel(a->y_cache);
    for (int i = 0; i < total; i++) a->y_cache->data[i] = x->data[i] + a->gamma * o->data[i];
    return a->y_cache;
}

TensorTrain* attention2d_backward(Attention2D* a, TensorTrain* dy) {
    TensorTrain* dx = tt_create(dy->n, dy->c, dy->h, dy->w);
    int total = tt_numel(dy);
    for (int i = 0; i < total; i++) dx->data[i] = dy->data[i];

    TensorTrain* do_in = tt_create(dy->n, dy->c, dy->h, dy->w);
    for (int i = 0; i < total; i++) do_in->data[i] = a->gamma * dy->data[i];

    TensorTrain* doutg = conv2d_backward(a->o, do_in);
    tt_free(do_in);

    int n = dy->n;
    int hw = dy->h * dy->w;
    int ca = a->c_attn;

    TensorTrain* dg = tt_create(n, ca, dy->h, dy->w);
    TensorTrain* dattn = tt_create(n, 1, hw, hw);

    for (int b = 0; b < n; b++) {
        for (int i = 0; i < hw; i++) {
            for (int j = 0; j < hw; j++) {
                float d_aij = 0.0f;
                int jy = j / dy->w, jx = j % dy->w;
                for (int c = 0; c < ca; c++) {
                    d_aij += TT_AT(doutg, b, c, i / dy->w, i % dy->w) * TT_AT(a->g_cache, b, c, jy, jx);
                    TT_AT(dg, b, c, jy, jx) += TT_AT(a->attn_cache, b, 0, i, j) * TT_AT(doutg, b, c, i / dy->w, i % dy->w);
                }
                TT_AT(dattn, b, 0, i, j) = d_aij;
            }
        }
    }

    TensorTrain* ds = tt_create(n, 1, hw, hw);
    for (int b = 0; b < n; b++) {
        for (int i = 0; i < hw; i++) {
            float dot = 0.0f;
            for (int j = 0; j < hw; j++) dot += TT_AT(dattn, b, 0, i, j) * TT_AT(a->attn_cache, b, 0, i, j);
            for (int j = 0; j < hw; j++) {
                float p = TT_AT(a->attn_cache, b, 0, i, j);
                TT_AT(ds, b, 0, i, j) = p * (TT_AT(dattn, b, 0, i, j) - dot);
            }
        }
    }

    TensorTrain* dtheta = tt_create(n, ca, dy->h, dy->w);
    TensorTrain* dphi = tt_create(n, ca, dy->h, dy->w);
    float inv_s = 1.0f / sqrtf((float)ca);
    for (int b = 0; b < n; b++) {
        for (int i = 0; i < hw; i++) {
            int iy = i / dy->w, ix = i % dy->w;
            for (int j = 0; j < hw; j++) {
                int jy = j / dy->w, jx = j % dy->w;
                float g = TT_AT(ds, b, 0, i, j) * inv_s;
                for (int c = 0; c < ca; c++) {
                    TT_AT(dtheta, b, c, iy, ix) += g * TT_AT(a->phi_cache, b, c, jy, jx);
                    TT_AT(dphi, b, c, jy, jx) += g * TT_AT(a->theta_cache, b, c, iy, ix);
                }
            }
        }
    }

    TensorTrain* dx_t = conv2d_backward(a->theta, dtheta);
    TensorTrain* dx_p = conv2d_backward(a->phi, dphi);
    TensorTrain* dx_g = conv2d_backward(a->g, dg);

    for (int i = 0; i < total; i++) {
        dx->data[i] += dx_t->data[i] + dx_p->data[i] + dx_g->data[i];
        a->gamma_grad += dy->data[i] * a->o->y_cache->data[i];
    }

    tt_free(doutg); tt_free(dg); tt_free(dattn); tt_free(ds); tt_free(dtheta); tt_free(dphi);
    tt_free(dx_t); tt_free(dx_p); tt_free(dx_g);
    return dx;
}

void attention2d_step(Attention2D* a, float lr, int t) {
    conv2d_step(a->theta, lr, t);
    conv2d_step(a->phi, lr, t);
    conv2d_step(a->g, lr, t);
    conv2d_step(a->o, lr, t);

    float b1 = 0.0f, b2 = 0.9f;
    a->gamma_m = b1 * a->gamma_m + (1.0f - b1) * a->gamma_grad;
    a->gamma_v = b2 * a->gamma_v + (1.0f - b2) * a->gamma_grad * a->gamma_grad;
    float b2p = powf(b2, (float)t);
    float lr_t = lr * sqrtf(1.0f - b2p);
    a->gamma -= lr_t * a->gamma_m / (sqrtf(a->gamma_v) + 1e-8f);
    a->gamma_grad = 0.0f;
}
