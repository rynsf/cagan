#ifndef DISCRIMINATOR_H
#define DISCRIMINATOR_H

#include "layers.h"

typedef struct {
    int in_dim;
    int class_dim;
    int hidden;

    LinearLayer* from_img;
    SelfAttention* attn;
    LinearLayer* head_real_fake;
    LinearLayer* class_proj;

    TensorTrain* class_cache;
    TensorTrain* feat_cache;
    TensorTrain* logits_cache;
} Discriminator;

Discriminator* discriminator_create(int in_dim, int class_dim, int hidden);
void discriminator_free(Discriminator* d);

TensorTrain* discriminator_forward(Discriminator* d, TensorTrain* img, const int* class_ids);
TensorTrain* discriminator_backward(Discriminator* d, TensorTrain* dlogits);
void discriminator_step(Discriminator* d, float lr, int step);

#endif
