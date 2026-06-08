#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "discriminator.h"
#include "generator.h"
#include "losses.h"
#include "tensor_train.h"

typedef struct {
    int batch;
    int z_dim;
    int class_dim;
    int image_dim;
    int hidden;
    int steps;
    float g_lr;
    float d_lr;
    int seed;
} TrainConfig;

static void config_defaults(TrainConfig* cfg) {
    cfg->batch = 8;
    cfg->z_dim = 128;
    cfg->class_dim = 1000;
    cfg->image_dim = 3 * 128 * 128;
    cfg->hidden = 1024;
    cfg->steps = 200;
    cfg->g_lr = 1e-4f;
    cfg->d_lr = 4e-4f;
    cfg->seed = 69;
}

static void usage(const char* prog) {
    printf("Usage: %s [options]\n", prog);
    printf("  -b <int>  batch size (default 8)\n");
    printf("  -t <int>  train steps (default 200)\n");
    printf("  -z <int>  latent dimension (default 128)\n");
    printf("  -H <int>  hidden width (default 1024)\n");
    printf("  -I <int>  flattened image dim (default 49152)\n");
    printf("  -s <int>  seed (default 69)\n");
}

static int random_class(int class_dim) {
    return rand() % class_dim;
}

static void synthesize_real_batch(TensorTrain* real_img, int* class_ids, int class_dim) {
    // This codebase focuses on GAN mechanics from scratch.
    // Real data pipeline is replaced by synthetic placeholders so training logic is transparent.
    tt_fill_randn(real_img, 0.5f);
    for (int i = 0; i < real_img->n; i++) class_ids[i] = random_class(class_dim);
}

int main(int argc, char** argv) {
    TrainConfig cfg;
    config_defaults(&cfg);

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-b") && i + 1 < argc) cfg.batch = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-t") && i + 1 < argc) cfg.steps = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-z") && i + 1 < argc) cfg.z_dim = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-H") && i + 1 < argc) cfg.hidden = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-I") && i + 1 < argc) cfg.image_dim = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-s") && i + 1 < argc) cfg.seed = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-h")) { usage(argv[0]); return 0; }
    }

    srand((unsigned int)cfg.seed);
    printf("--- SAGAN C Training Scaffold ---\n");
    printf("batch=%d steps=%d z_dim=%d hidden=%d image_dim=%d\n",
           cfg.batch, cfg.steps, cfg.z_dim, cfg.hidden, cfg.image_dim);

    Generator* G = generator_create(cfg.z_dim, cfg.class_dim, cfg.hidden, cfg.image_dim);
    Discriminator* D = discriminator_create(cfg.image_dim, cfg.class_dim, cfg.hidden);

    TensorTrain* real_img = tt_create(cfg.batch, 1, 1, cfg.image_dim);
    TensorTrain* z = tt_create(cfg.batch, 1, 1, cfg.z_dim);
    TensorTrain* dlogits = tt_create(cfg.batch, 1, 1, 1);

    int* class_ids = (int*)malloc((size_t)cfg.batch * sizeof(int));
    int* fake_class_ids = (int*)malloc((size_t)cfg.batch * sizeof(int));

    for (int step = 1; step <= cfg.steps; step++) {
        // ---------------------------
        // 1) Train Discriminator
        // ---------------------------
        synthesize_real_batch(real_img, class_ids, cfg.class_dim);

        TensorTrain* real_logits = discriminator_forward(D, real_img, class_ids);
        float d_real = hinge_d_real(real_logits, dlogits);
        TensorTrain* dreal_img = discriminator_backward(D, dlogits);
        tt_free(dreal_img);

        tt_fill_randn(z, 1.0f);
        for (int i = 0; i < cfg.batch; i++) fake_class_ids[i] = random_class(cfg.class_dim);
        TensorTrain* fake_img = generator_forward(G, z, fake_class_ids);

        TensorTrain* fake_logits = discriminator_forward(D, fake_img, fake_class_ids);
        float d_fake = hinge_d_fake(fake_logits, dlogits);
        TensorTrain* dfake_img = discriminator_backward(D, dlogits);
        tt_free(dfake_img);

        discriminator_step(D, cfg.d_lr, step);

        // ---------------------------
        // 2) Train Generator
        // ---------------------------
        TensorTrain* fake_logits_for_g = discriminator_forward(D, fake_img, fake_class_ids);
        float g_loss = hinge_g(fake_logits_for_g, dlogits);
        TensorTrain* dimg = discriminator_backward(D, dlogits);
        TensorTrain* dz = generator_backward(G, dimg);
        tt_free(dimg);
        tt_free(dz);

        generator_step(G, cfg.g_lr, step);

        if (step % 10 == 0 || step == 1) {
            printf("step %04d | d_real=%.5f d_fake=%.5f g=%.5f\n", step, d_real, d_fake, g_loss);
        }
    }

    free(class_ids);
    free(fake_class_ids);
    tt_free(real_img);
    tt_free(z);
    tt_free(dlogits);
    generator_free(G);
    discriminator_free(D);

    printf("Training run complete.\n");
    return 0;
}
