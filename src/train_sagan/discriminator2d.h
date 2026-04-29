#ifndef DISCRIMINATOR2D_H
#define DISCRIMINATOR2D_H

#include "attention2d.h"
#include "layers.h"

typedef struct {
    int class_dim;
    int in_hw;
    int hidden_c;

    Conv2D* from_rgb;
    Attention2D* attn;
    Conv2D* conv2;
    LinearLayer* class_proj;
    LinearLayer* head;

    TensorTrain* pooled_cache;
    TensorTrain* class_oh_cache;
} Discriminator2D;

Discriminator2D* discriminator2d_create(int class_dim, int in_hw, int hidden_c);
void discriminator2d_free(Discriminator2D* d);
TensorTrain* discriminator2d_forward(Discriminator2D* d, TensorTrain* img, const int* class_ids);
TensorTrain* discriminator2d_backward(Discriminator2D* d, TensorTrain* dlogits);
void discriminator2d_step(Discriminator2D* d, float lr, int t);

#endif
