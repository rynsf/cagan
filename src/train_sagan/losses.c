#include "losses.h"

#include <math.h>

float bce_with_logits_real(const TensorTrain* logits, TensorTrain* dlogits) {
    int n = logits->n;
    float loss = 0.0f;
    for (int i = 0; i < n; i++) {
        float x = logits->data[i];
        float p = 1.0f / (1.0f + expf(-x));
        loss += -logf(p + 1e-12f);
        dlogits->data[i] = (p - 1.0f) / (float)n;
    }
    return loss / (float)n;
}

float bce_with_logits_fake(const TensorTrain* logits, TensorTrain* dlogits) {
    int n = logits->n;
    float loss = 0.0f;
    for (int i = 0; i < n; i++) {
        float x = logits->data[i];
        float p = 1.0f / (1.0f + expf(-x));
        loss += -logf(1.0f - p + 1e-12f);
        dlogits->data[i] = p / (float)n;
    }
    return loss / (float)n;
}

float hinge_d_real(const TensorTrain* logits, TensorTrain* dlogits) {
    int n = logits->n;
    float loss = 0.0f;
    for (int i = 0; i < n; i++) {
        float x = logits->data[i];
        float m = 1.0f - x;
        if (m > 0.0f) {
            loss += m;
            dlogits->data[i] = -1.0f / (float)n;
        } else {
            dlogits->data[i] = 0.0f;
        }
    }
    return loss / (float)n;
}

float hinge_d_fake(const TensorTrain* logits, TensorTrain* dlogits) {
    int n = logits->n;
    float loss = 0.0f;
    for (int i = 0; i < n; i++) {
        float x = logits->data[i];
        float m = 1.0f + x;
        if (m > 0.0f) {
            loss += m;
            dlogits->data[i] = 1.0f / (float)n;
        } else {
            dlogits->data[i] = 0.0f;
        }
    }
    return loss / (float)n;
}

float hinge_g(const TensorTrain* logits, TensorTrain* dlogits) {
    int n = logits->n;
    float loss = 0.0f;
    for (int i = 0; i < n; i++) {
        loss += -logits->data[i];
        dlogits->data[i] = -1.0f / (float)n;
    }
    return loss / (float)n;
}
