#ifndef MODEL_H
#define MODEL_H

#include "tensor.h"
#include <string.h>

// --- Weight Structures ---
typedef struct {
    Tensor *c1_w, *c1_b, *c2_w, *c2_b, *sc_w, *sc_b;
    Tensor *rm1, *rv1, *w_emb1, *b_emb1;
    Tensor *rm2, *rv2, *w_emb2, *b_emb2;
} ResBlockWeights;

typedef struct {
    Tensor *theta_w, *phi_w, *g_w, *o_w, *gamma;
} AttentionWeights;

typedef struct {
    Tensor *n2f_w, *n2f_b; // Dense layer
    ResBlockWeights blocks[6]; // Indices 0, 1, 2, 3, and 5 (4 is attention)
    AttentionWeights attn; // Block 4
    Tensor *rgb_conv_w, *rgb_conv_b;
    Tensor *rgb_rm, *rgb_rv, *rgb_w, *rgb_b;
} SAGAN_Weights;

// --- Smart Loader Helper ---
// Helper macro to allocate and load in one line
#define ALLOC_AND_LOAD(t_ptr, n, c, h, w, fmt, ...) do { \
    char filepath[256]; \
    snprintf(filepath, sizeof(filepath), fmt, __VA_ARGS__); \
    t_ptr = create_tensor(n, c, h, w); \
    if (!load_weights(t_ptr, filepath)) { \
        printf("FATAL: Failed to load %s\n", filepath); \
        exit(1); \
    } \
} while(0)

// --- Load a single ResNet Block ---
void load_resblock(ResBlockWeights* b, const char* dir, int idx, int in_c, int out_c) {
    // Conv 1 (Upsamples, so weights are [out_c, in_c, 3, 3])
    ALLOC_AND_LOAD(b->c1_w, out_c, in_c, 3, 3, "%s/generator_ema_conv_blocks_%d_conv_1_conv_w.bin", dir, idx);
    ALLOC_AND_LOAD(b->c1_b, 1, out_c, 1, 1, "%s/generator_ema_conv_blocks_%d_conv_1_conv_b.bin", dir, idx);
    
    // Conv 2 (Maintains channels: [out_c, out_c, 3, 3])
    ALLOC_AND_LOAD(b->c2_w, out_c, out_c, 3, 3, "%s/generator_ema_conv_blocks_%d_conv_2_conv_w.bin", dir, idx);
    ALLOC_AND_LOAD(b->c2_b, 1, out_c, 1, 1, "%s/generator_ema_conv_blocks_%d_conv_2_conv_b.bin", dir, idx);
    
    // Shortcut (1x1 conv)
    ALLOC_AND_LOAD(b->sc_w, out_c, in_c, 1, 1, "%s/generator_ema_conv_blocks_%d_shortcut_conv_w.bin", dir, idx);
    ALLOC_AND_LOAD(b->sc_b, 1, out_c, 1, 1, "%s/generator_ema_conv_blocks_%d_shortcut_conv_b.bin", dir, idx);

    // Conditional BatchNorm 1 (operates on in_c)
    ALLOC_AND_LOAD(b->rm1, 1, in_c, 1, 1, "%s/generator_ema_conv_blocks_%d_norm_1_norm_rm.bin", dir, idx);
    ALLOC_AND_LOAD(b->rv1, 1, in_c, 1, 1, "%s/generator_ema_conv_blocks_%d_norm_1_norm_rv.bin", dir, idx);
    ALLOC_AND_LOAD(b->w_emb1, 1000, in_c, 1, 1, "%s/generator_ema_conv_blocks_%d_norm_1_weight_embedding_w.bin", dir, idx);
    ALLOC_AND_LOAD(b->b_emb1, 1000, in_c, 1, 1, "%s/generator_ema_conv_blocks_%d_norm_1_bias_embedding_w.bin", dir, idx); // Note: named _w in Python export

    // Conditional BatchNorm 2 (operates on out_c)
    ALLOC_AND_LOAD(b->rm2, 1, out_c, 1, 1, "%s/generator_ema_conv_blocks_%d_norm_2_norm_rm.bin", dir, idx);
    ALLOC_AND_LOAD(b->rv2, 1, out_c, 1, 1, "%s/generator_ema_conv_blocks_%d_norm_2_norm_rv.bin", dir, idx);
    ALLOC_AND_LOAD(b->w_emb2, 1000, out_c, 1, 1, "%s/generator_ema_conv_blocks_%d_norm_2_weight_embedding_w.bin", dir, idx);
    ALLOC_AND_LOAD(b->b_emb2, 1000, out_c, 1, 1, "%s/generator_ema_conv_blocks_%d_norm_2_bias_embedding_w.bin", dir, idx);
}

// --- Main Loader ---
SAGAN_Weights* load_all_weights(const char* dir) {
    SAGAN_Weights* model = (SAGAN_Weights*)malloc(sizeof(SAGAN_Weights));
    printf("Loading Dense Layer...\n");
    ALLOC_AND_LOAD(model->n2f_w, 16384, 128, 1, 1, "%s/generator_ema_noise2feat_w.bin", dir);
    ALLOC_AND_LOAD(model->n2f_b, 16384, 1, 1, 1, "%s/generator_ema_noise2feat_b.bin", dir);

    printf("Loading ResNet Blocks...\n");
    // The channel progression from our Python extraction
    int in_channels[]  = {1024, 1024, 512, 256, 128, 128};
    int out_channels[] = {1024, 512,  256, 128, 128, 64}; 

    for (int i = 0; i < 6; i++) {
        if (i == 4) continue; // Skip 4, it's the Attention block
        load_resblock(&model->blocks[i], dir, i, in_channels[i], out_channels[i]);
    }

    printf("Loading Attention Block (Block 4)...\n");
    ALLOC_AND_LOAD(model->attn.theta_w, 16, 128, 1, 1, "%s/generator_ema_conv_blocks_4_theta_conv_w.bin", dir);
    ALLOC_AND_LOAD(model->attn.phi_w, 16, 128, 1, 1, "%s/generator_ema_conv_blocks_4_phi_conv_w.bin", dir);
    ALLOC_AND_LOAD(model->attn.g_w, 64, 128, 1, 1, "%s/generator_ema_conv_blocks_4_g_conv_w.bin", dir);
    ALLOC_AND_LOAD(model->attn.o_w, 128, 64, 1, 1, "%s/generator_ema_conv_blocks_4_o_conv_w.bin", dir);
    ALLOC_AND_LOAD(model->attn.gamma, 1, 1, 1, 1, "%s/generator_ema_conv_blocks_4_gamma_w.bin", dir);

    printf("Loading RGB Output Layer...\n");
    ALLOC_AND_LOAD(model->rgb_conv_w, 3, 64, 3, 3, "%s/generator_ema_to_rgb_conv_w.bin", dir);
    ALLOC_AND_LOAD(model->rgb_conv_b, 3, 1, 1, 1, "%s/generator_ema_to_rgb_conv_b.bin", dir);
    ALLOC_AND_LOAD(model->rgb_rm, 64, 1, 1, 1, "%s/generator_ema_to_rgb_bn_rm.bin", dir);
    ALLOC_AND_LOAD(model->rgb_rv, 64, 1, 1, 1, "%s/generator_ema_to_rgb_bn_rv.bin", dir);
    ALLOC_AND_LOAD(model->rgb_w, 64, 1, 1, 1, "%s/generator_ema_to_rgb_bn_w.bin", dir);
    ALLOC_AND_LOAD(model->rgb_b, 64, 1, 1, 1, "%s/generator_ema_to_rgb_bn_b.bin", dir);

    return model;
}

#endif
