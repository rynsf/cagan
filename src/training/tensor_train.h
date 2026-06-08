#ifndef TENSOR_TRAIN_H
#define TENSOR_TRAIN_H

#include <stddef.h>

typedef struct {
    int n;
    int c;
    int h;
    int w;
    float* data;
    float* grad;
} TensorTrain;

#define TT_AT(t, n_idx, c_idx, y_idx, x_idx) \
    ((t)->data[(((n_idx) * (t)->c + (c_idx)) * (t)->h + (y_idx)) * (t)->w + (x_idx)])

#define TT_GRAD_AT(t, n_idx, c_idx, y_idx, x_idx) \
    ((t)->grad[(((n_idx) * (t)->c + (c_idx)) * (t)->h + (y_idx)) * (t)->w + (x_idx)])

TensorTrain* tt_create(int n, int c, int h, int w);
void tt_free(TensorTrain* t);
void tt_zero(TensorTrain* t);
void tt_zero_grad(TensorTrain* t);
int tt_numel(const TensorTrain* t);
void tt_fill_randn(TensorTrain* t, float stddev);
void tt_copy(TensorTrain* dst, const TensorTrain* src);

#endif
