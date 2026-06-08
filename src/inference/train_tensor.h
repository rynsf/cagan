#ifndef TRAIN_TENSOR_H
#define TRAIN_TENSOR_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

typedef struct {
    int n;
    int c;
    int h;
    int w;
    float* data;
    float* grad;
} TrainTensor;

#define TT_AT(t, n_idx, c_idx, y_idx, x_idx) \
    ((t)->data[(((n_idx) * (t)->c + (c_idx)) * (t)->h + (y_idx)) * (t)->w + (x_idx)])

#define TT_GRAD_AT(t, n_idx, c_idx, y_idx, x_idx) \
    ((t)->grad[(((n_idx) * (t)->c + (c_idx)) * (t)->h + (y_idx)) * (t)->w + (x_idx)])

static inline int tt_numel(TrainTensor* t) {
    return t->n * t->c * t->h * t->w;
}

static inline TrainTensor* tt_create(int n, int c, int h, int w) {
    TrainTensor* t = (TrainTensor*)malloc(sizeof(TrainTensor));
    t->n = n;
    t->c = c;
    t->h = h;
    t->w = w;
    int total = tt_numel(t);
    t->data = (float*)calloc((size_t)total, sizeof(float));
    t->grad = (float*)calloc((size_t)total, sizeof(float));
    if (!t->data || !t->grad) {
        printf("FATAL: tensor allocation failed.\n");
        exit(1);
    }
    return t;
}

static inline void tt_free(TrainTensor* t) {
    if (!t) return;
    if (t->data) free(t->data);
    if (t->grad) free(t->grad);
    free(t);
}

static inline void tt_zero(TrainTensor* t) {
    memset(t->data, 0, (size_t)tt_numel(t) * sizeof(float));
}

static inline void tt_zero_grad(TrainTensor* t) {
    memset(t->grad, 0, (size_t)tt_numel(t) * sizeof(float));
}

static inline float tt_randn(void) {
    float u1 = (float)rand() / (float)RAND_MAX;
    float u2 = (float)rand() / (float)RAND_MAX;
    if (u1 <= 1e-7f) u1 = 1e-7f;
    return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * 3.14159265358979323846f * u2);
}

static inline void tt_randn_fill(TrainTensor* t, float std) {
    int total = tt_numel(t);
    for (int i = 0; i < total; i++) t->data[i] = tt_randn() * std;
}

static inline void tt_matmul(const float* a, const float* b, float* c, int m, int k, int n) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int p = 0; p < k; p++) sum += a[i * k + p] * b[p * n + j];
            c[i * n + j] = sum;
        }
    }
}

typedef struct {
    TrainTensor** params;
    int count;
    float lr;
    float beta1;
    float beta2;
    float eps;
    float wd;
    int step;
    float* m;
    float* v;
} TrainAdam;

static inline void adam_init(TrainAdam* opt, TrainTensor** params, int count,
                             float lr, float b1, float b2, float eps, float wd) {
    opt->params = params;
    opt->count = count;
    opt->lr = lr;
    opt->beta1 = b1;
    opt->beta2 = b2;
    opt->eps = eps;
    opt->wd = wd;
    opt->step = 0;
    opt->m = (float*)calloc((size_t)count, sizeof(float));
    opt->v = (float*)calloc((size_t)count, sizeof(float));
}

static inline void adam_step(TrainAdam* opt) {
    opt->step += 1;
    for (int p = 0; p < opt->count; p++) {
        TrainTensor* t = opt->params[p];
        int total = tt_numel(t);
        float gmean = 0.0f;
        for (int i = 0; i < total; i++) gmean += fabsf(t->grad[i]);
        gmean /= (float)(total + 1);

        opt->m[p] = opt->beta1 * opt->m[p] + (1.0f - opt->beta1) * gmean;
        opt->v[p] = opt->beta2 * opt->v[p] + (1.0f - opt->beta2) * gmean * gmean;
        float mhat = opt->m[p] / (1.0f - powf(opt->beta1, (float)opt->step));
        float vhat = opt->v[p] / (1.0f - powf(opt->beta2, (float)opt->step));
        float scale = opt->lr * mhat / (sqrtf(vhat) + opt->eps);

        for (int i = 0; i < total; i++) {
            float reg = opt->wd * t->data[i];
            t->data[i] -= scale * (t->grad[i] + reg);
            t->grad[i] = 0.0f;
        }
    }
}

static inline void adam_free(TrainAdam* opt) {
    if (opt->m) free(opt->m);
    if (opt->v) free(opt->v);
}

#endif
