#ifndef DEBUG_VIS_H
#define DEBUG_VIS_H

#include "tensor_train.h"

typedef struct {
    int enabled;
    int every;
    char out_dir[256];
} DebugVisConfig;

void debug_vis_prepare_dir(const char* dir);
void debug_dump_channel_pgm(const TensorTrain* t, int n_idx, int c_idx, const char* filepath, int use_abs);
void debug_dump_rgb_ppm(const TensorTrain* t, int n_idx, const char* filepath);
void debug_log_tensor_stats(const char* tag, const TensorTrain* t);

#endif
