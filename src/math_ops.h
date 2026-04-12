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

// Fetches the 1D vector for a specific class from a 2D embedding table.
void embedding_lookup(int class_id, Tensor* weight_table, Tensor* output) {
    // weight_table shape: [1000 classes, channels, 1, 1]
    // output shape: [1, channels, 1, 1]
    int channels = weight_table->c;
    
    // Calculate the starting memory index for the requested class
    int start_idx = class_id * channels;
    
    for (int i = 0; i < channels; i++) {
        output->data[i] = weight_table->data[start_idx + i];
    }
}

// Computes out = (input * weight^T) + bias
void linear_layer(Tensor* input, Tensor* weight, Tensor* bias, Tensor* output) {
    // input: [1, in_features]
    // weight: [out_features, in_features]
    // bias: [out_features]
    // output: [1, out_features]
    
    int in_features = input->c; // e.g., 128
    int out_features = weight->n; // e.g., 16384
    
    for (int o = 0; o < out_features; o++) {
        float sum = 0.0f;
        for (int i = 0; i < in_features; i++) {
            // weight indexing: row = o, col = i
            sum += input->data[i] * weight->data[o * in_features + i];
        }
        output->data[o] = sum + bias->data[o];
    }
}

// Normalizes the spatial dimensions and applies class-specific scaling
void conditional_batch_norm_2d(Tensor* x, Tensor* rm, Tensor* rv, Tensor* gamma_emb, Tensor* beta_emb, Tensor* out) {
    float eps = 1e-5f;
    int channels = x->c;
    int spatial_size = x->h * x->w;

    for (int c = 0; c < channels; c++) {
        float mean = rm->data[c];
        float var = rv->data[c];
        float gamma = gamma_emb->data[c];
        float beta = beta_emb->data[c];
        
        // Pre-calculate the combined scale and shift for this channel
        // Formula: (x - mean) / sqrt(var + eps) * (1 + gamma) + beta
        // This condenses down to: x * scale + shift
        float scale = (1.0f + gamma) / sqrtf(var + eps);
        float shift = beta - (mean * scale);
        
        // Apply to all pixels in this channel
        int channel_offset = c * spatial_size;
        for (int i = 0; i < spatial_size; i++) {
            out->data[channel_offset + i] = x->data[channel_offset + i] * scale + shift;
        }
    }
}

#endif
