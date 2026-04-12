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

// Direct Convolution: out = conv2d(in, weight) + bias
void conv2d(Tensor* input, Tensor* weight, Tensor* bias, Tensor* output, int padding) {
    int in_c = input->c;
    int h = input->h;
    int w = input->w;
    
    int out_c = weight->n; // Number of output filters
    int kh = weight->h;    // Kernel height
    int kw = weight->w;    // Kernel width
    
    // Iterate over every output channel
    for (int oc = 0; oc < out_c; oc++) {
        // Bias is optional in some architectures. Handle NULL safely.
        float b_val = (bias != NULL) ? bias->data[oc] : 0.0f;
        
        // Iterate over every pixel in the output spatial grid
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                float sum = b_val;
                
                // Dot product of the kernel and the input region
                for (int ic = 0; ic < in_c; ic++) {
                    for (int ky = 0; ky < kh; ky++) {
                        for (int kx = 0; kx < kw; kx++) {
                            // Map the kernel position to the input image, accounting for padding
                            int in_y = y + ky - padding;
                            int in_x = x + kx - padding;
                            
                            // If the coordinate is within the image (not in the padded zone)
                            if (in_y >= 0 && in_y < h && in_x >= 0 && in_x < w) {
                                float p_val = T_AT(input, 0, ic, in_y, in_x);
                                float w_val = T_AT(weight, oc, ic, ky, kx);
                                sum += p_val * w_val;
                            }
                        }
                    }
                }
                // Write to the output tensor
                T_AT(output, 0, oc, y, x) = sum;
            }
        }
    }
}

// 2x2 Max Pooling with Stride 2
void max_pool_2x2(Tensor* input, Tensor* output) {
    for (int c = 0; c < input->c; c++) {
        for (int y = 0; y < output->h; y++) {
            for (int x = 0; x < output->w; x++) {
                // Map output coordinate to the top-left of the 2x2 input block
                int in_y = y * 2;
                int in_x = x * 2;
                
                float max_val = T_AT(input, 0, c, in_y, in_x);
                
                // Compare with the other 3 pixels in the 2x2 block
                float p1 = T_AT(input, 0, c, in_y, in_x + 1);
                float p2 = T_AT(input, 0, c, in_y + 1, in_x);
                float p3 = T_AT(input, 0, c, in_y + 1, in_x + 1);
                
                if (p1 > max_val) max_val = p1;
                if (p2 > max_val) max_val = p2;
                if (p3 > max_val) max_val = p3;
                
                T_AT(output, 0, c, y, x) = max_val;
            }
        }
    }
}

// Output squeeze
void tanh_activation(Tensor* t) {
    int total = t->n * t->c * t->h * t->w;
    for (int i = 0; i < total; i++) {
        t->data[i] = tanhf(t->data[i]);
    }
}

// Standard Batch Normalization (x - mean) / sqrt(var + eps) * weight + bias
void batch_norm_2d(Tensor* x, Tensor* rm, Tensor* rv, Tensor* weight, Tensor* bias, Tensor* out) {
    float eps = 1e-5f;
    int channels = x->c;
    int spatial_size = x->h * x->w;

    for (int c = 0; c < channels; c++) {
        float mean = rm->data[c];
        float var = rv->data[c];
        float w = weight->data[c];
        float b = bias->data[c];

        // Pre-calculate scale and shift for the entire channel
        float scale = w / sqrtf(var + eps);
        float shift = b - (mean * scale);

        int offset = c * spatial_size;
        for (int i = 0; i < spatial_size; i++) {
            out->data[offset + i] = x->data[offset + i] * scale + shift;
        }
    }
}

// BMM 1: Calculates the raw Attention Map
void attention_bmm1(Tensor* theta, Tensor* phi, Tensor* attn) {
    int spatial_t = theta->h * theta->w; // e.g., 64x64 = 4096
    int spatial_p = phi->h * phi->w;     // e.g., 32x32 = 1024
    int channels = theta->c;

    // Output is stored in 'attn' which we will allocate as [1, 1, spatial_t, spatial_p]
    for (int i = 0; i < spatial_t; i++) {
        for (int j = 0; j < spatial_p; j++) {
            float sum = 0.0f;
            for (int c = 0; c < channels; c++) {
                // Dot product across the channel dimension
                sum += theta->data[c * spatial_t + i] * phi->data[c * spatial_p + j];
            }
            attn->data[i * spatial_p + j] = sum;
        }
    }
}

// Softmax applied to the last dimension of the Attention Map
void softmax_last_dim(Tensor* attn) {
    int rows = attn->h; // 4096
    int cols = attn->w; // 1024

    for (int i = 0; i < rows; i++) {
        int row_offset = i * cols;
        
        // 1. Find max for numerical stability
        float max_val = attn->data[row_offset];
        for (int j = 1; j < cols; j++) {
            if (attn->data[row_offset + j] > max_val) {
                max_val = attn->data[row_offset + j];
            }
        }

        // 2. Calculate Exponentials and Sum
        float sum = 0.0f;
        for (int j = 0; j < cols; j++) {
            attn->data[row_offset + j] = expf(attn->data[row_offset + j] - max_val);
            sum += attn->data[row_offset + j];
        }

        // 3. Normalize to probabilities
        for (int j = 0; j < cols; j++) {
            attn->data[row_offset + j] /= sum;
        }
    }
}

// BMM 2: Applies the Attention Map to the 'G' tensor
void attention_bmm2(Tensor* attn, Tensor* g, Tensor* out) {
    int spatial_t = attn->h; // 4096
    int spatial_p = attn->w; // 1024
    int channels = g->c;     // 64

    for (int c = 0; c < channels; c++) {
        for (int i = 0; i < spatial_t; i++) {
            float sum = 0.0f;
            for (int j = 0; j < spatial_p; j++) {
                // attn[i, j] * g[c, j]
                sum += attn->data[i * spatial_p + j] * g->data[c * spatial_p + j];
            }
            // Writes directly into standard NCHW format!
            out->data[c * spatial_t + i] = sum;
        }
    }
}

#endif
