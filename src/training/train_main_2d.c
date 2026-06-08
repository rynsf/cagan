#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "debug_vis.h"
#include "discriminator2d.h"
#include "generator2d.h"
#include "losses.h"

typedef struct {
    int batch;
    int steps;
    int seed;
    int z_dim;
    int class_dim;
    int base_c;
    int base_hw;
    float g_lr;
    float d_lr;
    int debug_every;
    char debug_dir[256];
} CFG;

static void defaults(CFG* c) {
    c->batch = 2;
    c->steps = 5;
    c->seed = 69;
    c->z_dim = 64;
    c->class_dim = 1000;
    c->base_c = 16;
    c->base_hw = 8;
    c->g_lr = 1e-4f;
    c->d_lr = 4e-4f;
    c->debug_every = 0;
    strcpy(c->debug_dir, "debug_out");
}

static int rclass(int class_dim) { return rand() % class_dim; }

static void fake_real_data(TensorTrain* t, int* ids, int class_dim) {
    tt_fill_randn(t, 0.7f);
    for (int i = 0; i < t->n; i++) ids[i] = rclass(class_dim);
}

int main(int argc, char** argv) {
    CFG c;
    defaults(&c);
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-b") && i + 1 < argc) c.batch = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-t") && i + 1 < argc) c.steps = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-z") && i + 1 < argc) c.z_dim = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-C") && i + 1 < argc) c.base_c = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-S") && i + 1 < argc) c.base_hw = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-s") && i + 1 < argc) c.seed = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-V") && i + 1 < argc) c.debug_every = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-D") && i + 1 < argc) {
            strncpy(c.debug_dir, argv[++i], sizeof(c.debug_dir) - 1);
            c.debug_dir[sizeof(c.debug_dir) - 1] = '\0';
        }
    }
    srand((unsigned int)c.seed);

    int out_hw = c.base_hw * 2;
    printf("2D SAGAN train: batch=%d steps=%d base=%dx%dx%d out=%dx%d\n",
           c.batch, c.steps, c.base_c, c.base_hw, c.base_hw, out_hw, out_hw);
    if (c.debug_every > 0) {
        debug_vis_prepare_dir(c.debug_dir);
        printf("debug dumps enabled: every=%d dir=%s\n", c.debug_every, c.debug_dir);
    }

    Generator2D* G = generator2d_create(c.z_dim, c.class_dim, c.base_c, c.base_hw);
    Discriminator2D* D = discriminator2d_create(c.class_dim, out_hw, c.base_c);

    TensorTrain* z = tt_create(c.batch, 1, 1, c.z_dim);
    TensorTrain* real = tt_create(c.batch, 3, out_hw, out_hw);
    TensorTrain* dlogits = tt_create(c.batch, 1, 1, 1);

    int* y_real = (int*)malloc((size_t)c.batch * sizeof(int));
    int* y_fake = (int*)malloc((size_t)c.batch * sizeof(int));

    for (int step = 1; step <= c.steps; step++) {
        clock_t t0 = clock();
        printf("\n[step %03d] phase=prepare\n", step);
        fake_real_data(real, y_real, c.class_dim);
        tt_fill_randn(z, 1.0f);
        for (int i = 0; i < c.batch; i++) y_fake[i] = rclass(c.class_dim);

        printf("[step %03d] phase=generator_forward\n", step);
        TensorTrain* fake = generator2d_forward(G, z, y_fake);
        debug_log_tensor_stats("g.out(fake_img)", fake);

        printf("[step %03d] phase=discriminator_real\n", step);
        TensorTrain* lr = discriminator2d_forward(D, real, y_real);
        float d_real = hinge_d_real(lr, dlogits);
        TensorTrain* dr = discriminator2d_backward(D, dlogits);
        tt_free(dr);

        printf("[step %03d] phase=discriminator_fake\n", step);
        TensorTrain* lf = discriminator2d_forward(D, fake, y_fake);
        float d_fake = hinge_d_fake(lf, dlogits);
        TensorTrain* df = discriminator2d_backward(D, dlogits);
        tt_free(df);
        discriminator2d_step(D, c.d_lr, step);

        printf("[step %03d] phase=generator_update\n", step);
        TensorTrain* lfg = discriminator2d_forward(D, fake, y_fake);
        float g = hinge_g(lfg, dlogits);
        TensorTrain* dimg = discriminator2d_backward(D, dlogits);
        TensorTrain* dz = generator2d_backward(G, dimg);
        debug_log_tensor_stats("dL/d(fake_img)", dimg);
        tt_free(dz);
        generator2d_step(G, c.g_lr, step);

        if (c.debug_every > 0 && (step % c.debug_every == 0)) {
            char p[512];
            snprintf(p, sizeof(p), "%s/step_%04d_fake.ppm", c.debug_dir, step);
            debug_dump_rgb_ppm(fake, 0, p);

            snprintf(p, sizeof(p), "%s/step_%04d_g_feat_ch0.pgm", c.debug_dir, step);
            debug_dump_channel_pgm(G->feat_cache, 0, 0, p, 0);
            snprintf(p, sizeof(p), "%s/step_%04d_g_up_ch0.pgm", c.debug_dir, step);
            debug_dump_channel_pgm(G->up_cache, 0, 0, p, 0);
            snprintf(p, sizeof(p), "%s/step_%04d_g_q_ch0.pgm", c.debug_dir, step);
            debug_dump_channel_pgm(G->attn->theta_cache, 0, 0, p, 0);
            snprintf(p, sizeof(p), "%s/step_%04d_g_k_ch0.pgm", c.debug_dir, step);
            debug_dump_channel_pgm(G->attn->phi_cache, 0, 0, p, 0);
            snprintf(p, sizeof(p), "%s/step_%04d_g_v_ch0.pgm", c.debug_dir, step);
            debug_dump_channel_pgm(G->attn->g_cache, 0, 0, p, 0);
            snprintf(p, sizeof(p), "%s/step_%04d_g_attn_row0.pgm", c.debug_dir, step);
            debug_dump_channel_pgm(G->attn->attn_cache, 0, 0, p, 0);
            snprintf(p, sizeof(p), "%s/step_%04d_g_self_attn_ch0.pgm", c.debug_dir, step);
            debug_dump_channel_pgm(G->attn->y_cache, 0, 0, p, 0);

            snprintf(p, sizeof(p), "%s/step_%04d_d_in_ch0.pgm", c.debug_dir, step);
            debug_dump_channel_pgm(real, 0, 0, p, 0);
            snprintf(p, sizeof(p), "%s/step_%04d_d_feat_ch0.pgm", c.debug_dir, step);
            debug_dump_channel_pgm(D->from_rgb->y_cache, 0, 0, p, 0);
            snprintf(p, sizeof(p), "%s/step_%04d_d_q_ch0.pgm", c.debug_dir, step);
            debug_dump_channel_pgm(D->attn->theta_cache, 0, 0, p, 0);
            snprintf(p, sizeof(p), "%s/step_%04d_d_k_ch0.pgm", c.debug_dir, step);
            debug_dump_channel_pgm(D->attn->phi_cache, 0, 0, p, 0);
            snprintf(p, sizeof(p), "%s/step_%04d_d_v_ch0.pgm", c.debug_dir, step);
            debug_dump_channel_pgm(D->attn->g_cache, 0, 0, p, 0);
            snprintf(p, sizeof(p), "%s/step_%04d_d_attn_row0.pgm", c.debug_dir, step);
            debug_dump_channel_pgm(D->attn->attn_cache, 0, 0, p, 0);
            snprintf(p, sizeof(p), "%s/step_%04d_d_self_attn_ch0.pgm", c.debug_dir, step);
            debug_dump_channel_pgm(D->attn->y_cache, 0, 0, p, 0);
            snprintf(p, sizeof(p), "%s/step_%04d_grad_dimg_ch0_abs.pgm", c.debug_dir, step);
            debug_dump_channel_pgm(dimg, 0, 0, p, 1);
        }

        tt_free(dimg);

        double ms = 1000.0 * (double)(clock() - t0) / (double)CLOCKS_PER_SEC;
        printf("[step %03d] done | d_real=%.5f d_fake=%.5f g=%.5f | time=%.2fms\n", step, d_real, d_fake, g, ms);
    }

    free(y_real); free(y_fake);
    tt_free(z); tt_free(real); tt_free(dlogits);
    generator2d_free(G); discriminator2d_free(D);
    return 0;
}
