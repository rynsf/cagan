#include "debug_vis.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>

void debug_vis_prepare_dir(const char* dir) {
    mkdir(dir, 0755);
}

static void compute_min_max(const float* a, int size, int use_abs, float* out_min, float* out_max) {
    float mn = use_abs ? fabsf(a[0]) : a[0];
    float mx = mn;
    for (int i = 1; i < size; i++) {
        float v = use_abs ? fabsf(a[i]) : a[i];
        if (v < mn) mn = v;
        if (v > mx) mx = v;
    }
    *out_min = mn;
    *out_max = mx;
}

void debug_dump_channel_pgm(const TensorTrain* t, int n_idx, int c_idx, const char* filepath, int use_abs) {
    FILE* f = fopen(filepath, "wb");
    if (!f) return;

    int hw = t->h * t->w;
    const float* base = &t->data[((n_idx * t->c + c_idx) * hw)];
    float mn, mx;
    compute_min_max(base, hw, use_abs, &mn, &mx);
    float scale = 255.0f / ((mx - mn) + 1e-12f);

    fprintf(f, "P5\n%d %d\n255\n", t->w, t->h);
    for (int i = 0; i < hw; i++) {
        float v = use_abs ? fabsf(base[i]) : base[i];
        int px = (int)((v - mn) * scale);
        if (px < 0) px = 0;
        if (px > 255) px = 255;
        unsigned char b = (unsigned char)px;
        fwrite(&b, 1, 1, f);
    }
    fclose(f);
}

void debug_dump_rgb_ppm(const TensorTrain* t, int n_idx, const char* filepath) {
    if (t->c < 3) return;
    FILE* f = fopen(filepath, "wb");
    if (!f) return;

    int hw = t->h * t->w;
    const float* r = &t->data[((n_idx * t->c + 0) * hw)];
    const float* g = &t->data[((n_idx * t->c + 1) * hw)];
    const float* b = &t->data[((n_idx * t->c + 2) * hw)];

    fprintf(f, "P6\n%d %d\n255\n", t->w, t->h);
    for (int i = 0; i < hw; i++) {
        int rv = (int)((r[i] + 1.0f) * 127.5f);
        int gv = (int)((g[i] + 1.0f) * 127.5f);
        int bv = (int)((b[i] + 1.0f) * 127.5f);
        if (rv < 0) rv = 0; if (rv > 255) rv = 255;
        if (gv < 0) gv = 0; if (gv > 255) gv = 255;
        if (bv < 0) bv = 0; if (bv > 255) bv = 255;
        unsigned char px[3];
        px[0] = (unsigned char)rv;
        px[1] = (unsigned char)gv;
        px[2] = (unsigned char)bv;
        fwrite(px, 1, 3, f);
    }
    fclose(f);
}

void debug_log_tensor_stats(const char* tag, const TensorTrain* t) {
    int total = tt_numel((TensorTrain*)t);
    float mn = t->data[0], mx = t->data[0], mean = 0.0f;
    for (int i = 0; i < total; i++) {
        float v = t->data[i];
        if (v < mn) mn = v;
        if (v > mx) mx = v;
        mean += v;
    }
    mean /= (float)total;
    printf("[stats] %-24s shape=[%d,%d,%d,%d] min=% .5f max=% .5f mean=% .5f\n",
           tag, t->n, t->c, t->h, t->w, mn, mx, mean);
}
