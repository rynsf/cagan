#ifndef LAYERS_H
#define LAYERS_H

#include "tensor_train.h"

typedef struct {
    TensorTrain* x_cache;
    TensorTrain* w;
    TensorTrain* b;
    TensorTrain* y;
    TensorTrain* m_w;
    TensorTrain* v_w;
    TensorTrain* m_b;
    TensorTrain* v_b;
} LinearLayer;

LinearLayer* linear_create(int in_features, int out_features);
void linear_free(LinearLayer* layer);
TensorTrain* linear_forward(LinearLayer* layer, TensorTrain* x);
TensorTrain* linear_backward(LinearLayer* layer, TensorTrain* dy);
void linear_step(LinearLayer* layer, float lr, int t);

typedef struct {
    TensorTrain* w_q;
    TensorTrain* w_k;
    TensorTrain* w_v;
    TensorTrain* w_o;
    TensorTrain* m_w_q;
    TensorTrain* v_w_q;
    TensorTrain* m_w_k;
    TensorTrain* v_w_k;
    TensorTrain* m_w_v;
    TensorTrain* v_w_v;
    TensorTrain* m_w_o;
    TensorTrain* v_w_o;

    TensorTrain* x_cache;
    TensorTrain* q_cache;
    TensorTrain* k_cache;
    TensorTrain* v_cache;
    TensorTrain* a_cache;
    TensorTrain* y_cache;
} SelfAttention;

SelfAttention* attention_create(int dim, int attn_dim);
void attention_free(SelfAttention* attn);
TensorTrain* attention_forward(SelfAttention* attn, TensorTrain* x);
TensorTrain* attention_backward(SelfAttention* attn, TensorTrain* dy);
void attention_step(SelfAttention* attn, float lr, int t);

#endif
