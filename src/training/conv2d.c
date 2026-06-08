#include "conv2d.h"

#include "linalg.h"

#include <stdlib.h>

Conv2D* conv2d_create(int in_c, int out_c, int k, int padding) {
    Conv2D* cv = (Conv2D*)malloc(sizeof(Conv2D));
    cv->in_c = in_c;
    cv->out_c = out_c;
    cv->k = k;
    cv->padding = padding;
    cv->w = tt_create(out_c, in_c, k, k);
    cv->b = tt_create(1, out_c, 1, 1);
    tt_fill_randn(cv->w, 0.02f);
    tt_zero(cv->b);
    cv->m_w = tt_create(out_c, in_c, k, k);
    cv->v_w = tt_create(out_c, in_c, k, k);
    cv->m_b = tt_create(1, out_c, 1, 1);
    cv->v_b = tt_create(1, out_c, 1, 1);
    cv->x_cache = NULL;
    cv->y_cache = NULL;
    return cv;
}

void conv2d_free(Conv2D* cv) {
    if (!cv) return;
    tt_free(cv->w); tt_free(cv->b);
    tt_free(cv->m_w); tt_free(cv->v_w);
    tt_free(cv->m_b); tt_free(cv->v_b);
    if (cv->y_cache) tt_free(cv->y_cache);
    free(cv);
}

TensorTrain* conv2d_forward(Conv2D* cv, TensorTrain* x) {
    cv->x_cache = x;
    if (cv->y_cache) tt_free(cv->y_cache);
    cv->y_cache = tt_create(x->n, cv->out_c, x->h, x->w);

    for (int n = 0; n < x->n; n++) {
        for (int oc = 0; oc < cv->out_c; oc++) {
            for (int y = 0; y < x->h; y++) {
                for (int xx = 0; xx < x->w; xx++) {
                    float sum = cv->b->data[oc];
                    for (int ic = 0; ic < cv->in_c; ic++) {
                        for (int ky = 0; ky < cv->k; ky++) {
                            for (int kx = 0; kx < cv->k; kx++) {
                                int iy = y + ky - cv->padding;
                                int ix = xx + kx - cv->padding;
                                if (iy < 0 || ix < 0 || iy >= x->h || ix >= x->w) continue;
                                float xv = TT_AT(x, n, ic, iy, ix);
                                float wv = TT_AT(cv->w, oc, ic, ky, kx);
                                sum += xv * wv;
                            }
                        }
                    }
                    TT_AT(cv->y_cache, n, oc, y, xx) = sum;
                }
            }
        }
    }
    return cv->y_cache;
}

TensorTrain* conv2d_backward(Conv2D* cv, TensorTrain* dy) {
    TensorTrain* x = cv->x_cache;
    TensorTrain* dx = tt_create(x->n, x->c, x->h, x->w);
    tt_zero_grad(cv->w);
    tt_zero_grad(cv->b);

    for (int n = 0; n < x->n; n++) {
        for (int oc = 0; oc < cv->out_c; oc++) {
            for (int y = 0; y < x->h; y++) {
                for (int xx = 0; xx < x->w; xx++) {
                    float g = TT_AT(dy, n, oc, y, xx);
                    cv->b->grad[oc] += g;
                    for (int ic = 0; ic < cv->in_c; ic++) {
                        for (int ky = 0; ky < cv->k; ky++) {
                            for (int kx = 0; kx < cv->k; kx++) {
                                int iy = y + ky - cv->padding;
                                int ix = xx + kx - cv->padding;
                                if (iy < 0 || ix < 0 || iy >= x->h || ix >= x->w) continue;
                                TT_GRAD_AT(cv->w, oc, ic, ky, kx) += TT_AT(x, n, ic, iy, ix) * g;
                                TT_AT(dx, n, ic, iy, ix) += TT_AT(cv->w, oc, ic, ky, kx) * g;
                            }
                        }
                    }
                }
            }
        }
    }
    return dx;
}

void conv2d_step(Conv2D* cv, float lr, int t) {
    adam_update(cv->w->data, cv->w->grad, cv->m_w->data, cv->v_w->data,
                tt_numel(cv->w), lr, 0.0f, 0.9f, 1e-8f, t);
    adam_update(cv->b->data, cv->b->grad, cv->m_b->data, cv->v_b->data,
                tt_numel(cv->b), lr, 0.0f, 0.9f, 1e-8f, t);
}

TensorTrain* upsample2x_forward(TensorTrain* x) {
    TensorTrain* y = tt_create(x->n, x->c, x->h * 2, x->w * 2);
    for (int n = 0; n < x->n; n++) {
        for (int c = 0; c < x->c; c++) {
            for (int oy = 0; oy < y->h; oy++) {
                for (int ox = 0; ox < y->w; ox++) {
                    TT_AT(y, n, c, oy, ox) = TT_AT(x, n, c, oy / 2, ox / 2);
                }
            }
        }
    }
    return y;
}

TensorTrain* upsample2x_backward(TensorTrain* dy) {
    TensorTrain* dx = tt_create(dy->n, dy->c, dy->h / 2, dy->w / 2);
    for (int n = 0; n < dy->n; n++) {
        for (int c = 0; c < dy->c; c++) {
            for (int y = 0; y < dx->h; y++) {
                for (int x = 0; x < dx->w; x++) {
                    float sum = 0.0f;
                    sum += TT_AT(dy, n, c, y * 2 + 0, x * 2 + 0);
                    sum += TT_AT(dy, n, c, y * 2 + 0, x * 2 + 1);
                    sum += TT_AT(dy, n, c, y * 2 + 1, x * 2 + 0);
                    sum += TT_AT(dy, n, c, y * 2 + 1, x * 2 + 1);
                    TT_AT(dx, n, c, y, x) = sum;
                }
            }
        }
    }
    return dx;
}
