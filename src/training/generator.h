#ifndef GENERATOR_H
#define GENERATOR_H

#include "layers.h"

typedef struct {
    int z_dim;
    int class_dim;
    int hidden;
    int out_dim;

    LinearLayer* z_proj;
    LinearLayer* class_embed;
    LinearLayer* fuse;
    SelfAttention* attn;
    LinearLayer* to_img;

    TensorTrain* z_cache;
    TensorTrain* class_cache;
    TensorTrain* hidden_cache;
    TensorTrain* out_cache;
} Generator;

Generator* generator_create(int z_dim, int class_dim, int hidden, int out_dim);
void generator_free(Generator* g);

TensorTrain* generator_forward(Generator* g, TensorTrain* z, const int* class_ids);
TensorTrain* generator_backward(Generator* g, TensorTrain* dimg);
void generator_step(Generator* g, float lr, int step);

#endif
