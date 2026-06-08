#include "tensor_train.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static float rand_uniform() {
    return (float)rand() / (float)RAND_MAX;
}

static float rand_normal() {
    float u1 = rand_uniform();
    float u2 = rand_uniform();
    if (u1 < 1e-7f) u1 = 1e-7f;
    return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * 3.14159265359f * u2);
}

int tt_numel(const TensorTrain* t) {
    return t->n * t->c * t->h * t->w;
}

TensorTrain* tt_create(int n, int c, int h, int w) {
    TensorTrain* t = (TensorTrain*)malloc(sizeof(TensorTrain));
    if (!t) {
        printf("FATAL: Failed to allocate TensorTrain struct.\n");
        exit(1);
    }
    t->n = n;
    t->c = c;
    t->h = h;
    t->w = w;

    int total = n * c * h * w;
    t->data = (float*)calloc((size_t)total, sizeof(float));
    t->grad = (float*)calloc((size_t)total, sizeof(float));
    if (!t->data || !t->grad) {
        printf("FATAL: Failed to allocate tensor buffers (%d elements).\n", total);
        exit(1);
    }
    return t;
}

void tt_free(TensorTrain* t) {
    if (!t) return;
    if (t->data) free(t->data);
    if (t->grad) free(t->grad);
    free(t);
}

void tt_zero(TensorTrain* t) {
    int total = tt_numel(t);
    memset(t->data, 0, (size_t)total * sizeof(float));
}

void tt_zero_grad(TensorTrain* t) {
    int total = tt_numel(t);
    memset(t->grad, 0, (size_t)total * sizeof(float));
}

void tt_fill_randn(TensorTrain* t, float stddev) {
    int total = tt_numel(t);
    for (int i = 0; i < total; i++) {
        t->data[i] = rand_normal() * stddev;
    }
}

void tt_copy(TensorTrain* dst, const TensorTrain* src) {
    int dst_total = tt_numel(dst);
    int src_total = tt_numel(src);
    if (dst_total != src_total) {
        printf("FATAL: tt_copy mismatch (%d != %d).\n", dst_total, src_total);
        exit(1);
    }
    memcpy(dst->data, src->data, (size_t)dst_total * sizeof(float));
}
