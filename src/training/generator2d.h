#ifndef GENERATOR2D_H
#define GENERATOR2D_H

#include "attention2d.h"
#include "layers.h"

typedef struct {
    int z_dim;
    int class_dim;
    int base_c;
    int base_hw;

    LinearLayer* z_to_feat;
    LinearLayer* cls_to_feat;
    Conv2D* conv1;
    Attention2D* attn;
    Conv2D* to_rgb;

    TensorTrain* feat_cache;
    TensorTrain* up_cache;
    TensorTrain* out_cache;
    TensorTrain* class_oh_cache;
} Generator2D;

Generator2D* generator2d_create(int z_dim, int class_dim, int base_c, int base_hw);
void generator2d_free(Generator2D* g);
TensorTrain* generator2d_forward(Generator2D* g, TensorTrain* z, const int* class_ids);
TensorTrain* generator2d_backward(Generator2D* g, TensorTrain* dimg);
void generator2d_step(Generator2D* g, float lr, int t);

#endif
