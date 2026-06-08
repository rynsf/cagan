#ifndef ATTENTION2D_H
#define ATTENTION2D_H

#include "conv2d.h"

typedef struct {
    int c;
    int c_attn;
    Conv2D* theta;
    Conv2D* phi;
    Conv2D* g;
    Conv2D* o;
    float gamma;
    float gamma_grad;
    float gamma_m;
    float gamma_v;

    TensorTrain* x_cache;
    TensorTrain* theta_cache;
    TensorTrain* phi_cache;
    TensorTrain* g_cache;
    TensorTrain* attn_cache; // [n, hw, hw]
    TensorTrain* out_g_cache;
    TensorTrain* y_cache;
} Attention2D;

Attention2D* attention2d_create(int c, int c_attn);
void attention2d_free(Attention2D* a);
TensorTrain* attention2d_forward(Attention2D* a, TensorTrain* x);
TensorTrain* attention2d_backward(Attention2D* a, TensorTrain* dy);
void attention2d_step(Attention2D* a, float lr, int t);

#endif
