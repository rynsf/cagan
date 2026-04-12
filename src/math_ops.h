#ifndef MATH_OPS_H
#define MATH_OPS_H

#include "tensor.h"
#include <math.h>

// 1. ReLU (Rectified Linear Unit)
// Replaces all negative values with 0.
void relu(Tensor* t) {
    int total_elements = t->n * t->c * t->h * t->w;
    for (int i = 0; i < total_elements; i++) {
        if (t->data[i] < 0.0f) {
            t->data[i] = 0.0f;
        }
    }
}

// 2. Tensor Addition (out = a + b)
// Used at the end of the ResNet block: h + s
void tensor_add(Tensor* a, Tensor* b, Tensor* out) {
    int total_elements = a->n * a->c * a->h * a->w;
    for (int i = 0; i < total_elements; i++) {
        out->data[i] = a->data[i] + b->data[i];
    }
}

// 3. Nearest Neighbor Upsample (Scale Factor 2.0)
// Takes a tensor (e.g., 4x4) and scales it to (8x8) by duplicating pixels.
void nearest_neighbor_upsample(Tensor* in, Tensor* out) {
    for (int n = 0; n < in->n; n++) {
        for (int c = 0; c < in->c; c++) {
            for (int out_y = 0; out_y < out->h; out_y++) {
                for (int out_x = 0; out_x < out->w; out_x++) {
                    // Integer division inherently floors the value, 
                    // mapping out_x=0,1 to in_x=0; out_x=2,3 to in_x=1, etc.
                    int in_y = out_y / 2;
                    int in_x = out_x / 2;
                    
                    float val = T_AT(in, n, c, in_y, in_x);
                    T_AT(out, n, c, out_y, out_x) = val;
                }
            }
        }
    }
}

#endif
