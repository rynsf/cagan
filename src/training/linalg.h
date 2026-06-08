#ifndef LINALG_H
#define LINALG_H

#include "tensor_train.h"

// Matrix multiply:
// C[M, N] = A[M, K] * B[K, N]
void matmul_forward(const float* a, const float* b, float* c, int m, int k, int n);

// Gradient of matmul:
// dA = dC * B^T, dB = A^T * dC
void matmul_backward(const float* a, const float* b, const float* dc,
                    float* da, float* db, int m, int k, int n);

void add_bias_forward(float* y, const float* b, int rows, int cols);
void add_bias_backward(const float* dy, float* db, int rows, int cols);

void relu_forward(float* x, int size);
void relu_backward(const float* x, const float* dy, float* dx, int size);
void leaky_relu_forward(float* x, int size, float alpha);
void leaky_relu_backward(const float* x, const float* dy, float* dx, int size, float alpha);

void softmax_forward_rows(float* x, int rows, int cols);
void softmax_cross_entropy_backward(const float* probs, const int* labels, float* dlogits, int rows, int cols);

void adam_update(float* p, float* g, float* m, float* v, int size,
                 float lr, float beta1, float beta2, float eps, int t);

#endif
