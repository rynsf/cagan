#include "discriminator.h"

#include "linalg.h"

#include <stdlib.h>

static TensorTrain* one_hot_classes(int batch, int class_dim, const int* ids) {
    TensorTrain* t = tt_create(batch, 1, 1, class_dim);
    for (int i = 0; i < batch; i++) {
        int cid = ids[i];
        if (cid < 0) cid = 0;
        if (cid >= class_dim) cid = class_dim - 1;
        t->data[i * class_dim + cid] = 1.0f;
    }
    return t;
}

Discriminator* discriminator_create(int in_dim, int class_dim, int hidden) {
    Discriminator* d = (Discriminator*)malloc(sizeof(Discriminator));
    d->in_dim = in_dim;
    d->class_dim = class_dim;
    d->hidden = hidden;

    d->from_img = linear_create(in_dim, hidden);
    d->attn = attention_create(hidden, hidden / 2);
    d->head_real_fake = linear_create(hidden, 1);
    d->class_proj = linear_create(class_dim, hidden);

    d->class_cache = NULL;
    d->feat_cache = NULL;
    d->logits_cache = NULL;
    return d;
}

void discriminator_free(Discriminator* d) {
    if (!d) return;
    linear_free(d->from_img);
    attention_free(d->attn);
    linear_free(d->head_real_fake);
    linear_free(d->class_proj);
    if (d->class_cache) tt_free(d->class_cache);
    free(d);
}

TensorTrain* discriminator_forward(Discriminator* d, TensorTrain* img, const int* class_ids) {
    if (d->class_cache) tt_free(d->class_cache);
    d->class_cache = one_hot_classes(img->n, d->class_dim, class_ids);

    TensorTrain* feat = linear_forward(d->from_img, img);
    leaky_relu_forward(feat->data, img->n * d->hidden, 0.2f);

    TensorTrain* proj = linear_forward(d->class_proj, d->class_cache);
    for (int i = 0; i < img->n * d->hidden; i++) feat->data[i] += proj->data[i];

    TensorTrain* feat_attn = attention_forward(d->attn, feat);
    leaky_relu_forward(feat_attn->data, img->n * d->hidden, 0.2f);

    TensorTrain* logits = linear_forward(d->head_real_fake, feat_attn);

    d->feat_cache = feat_attn;
    d->logits_cache = logits;
    return logits;
}

TensorTrain* discriminator_backward(Discriminator* d, TensorTrain* dlogits) {
    TensorTrain* dfeat = linear_backward(d->head_real_fake, dlogits);
    TensorTrain* datt = attention_backward(d->attn, dfeat);
    tt_free(dfeat);

    TensorTrain* dimg = linear_backward(d->from_img, datt);

    // Class projection participates in conditioning and gets gradients too.
    TensorTrain* dclass = linear_backward(d->class_proj, datt);
    tt_free(dclass);
    tt_free(datt);
    return dimg;
}

void discriminator_step(Discriminator* d, float lr, int step) {
    linear_step(d->from_img, lr, step);
    attention_step(d->attn, lr, step);
    linear_step(d->head_real_fake, lr, step);
    linear_step(d->class_proj, lr, step);
}
