#ifndef CONV2D_H
#define CONV2D_H

#include "tensor_train.h"

typedef struct {
    int in_c;
    int out_c;
    int k;
    int padding;
    TensorTrain* w; // [out_c, in_c, k, k]
    TensorTrain* b; // [1, out_c, 1, 1]
    TensorTrain* m_w;
    TensorTrain* v_w;
    TensorTrain* m_b;
    TensorTrain* v_b;
    TensorTrain* x_cache;
    TensorTrain* y_cache;
} Conv2D;

Conv2D* conv2d_create(int in_c, int out_c, int k, int padding);
void conv2d_free(Conv2D* cv);
TensorTrain* conv2d_forward(Conv2D* cv, TensorTrain* x);
TensorTrain* conv2d_backward(Conv2D* cv, TensorTrain* dy);
void conv2d_step(Conv2D* cv, float lr, int t);

TensorTrain* upsample2x_forward(TensorTrain* x);
TensorTrain* upsample2x_backward(TensorTrain* dy);

#endif
