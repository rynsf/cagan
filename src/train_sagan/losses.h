#ifndef LOSSES_H
#define LOSSES_H

#include "tensor_train.h"

float bce_with_logits_real(const TensorTrain* logits, TensorTrain* dlogits);
float bce_with_logits_fake(const TensorTrain* logits, TensorTrain* dlogits);
float hinge_d_real(const TensorTrain* logits, TensorTrain* dlogits);
float hinge_d_fake(const TensorTrain* logits, TensorTrain* dlogits);
float hinge_g(const TensorTrain* logits, TensorTrain* dlogits);

#endif
