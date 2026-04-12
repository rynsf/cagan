#include <stdio.h>
#include "tensor.h"

int main() {
    printf("Initializing SAGAN C-Engine Verification...\n\n");

    // 1. Target the first Dense layer weight file we extracted
    // Make sure this path points to where your .bin files are saved
    const char* target_file = "../bin/generator_ema_noise2feat_w.bin";

    // 2. Allocate memory based on the Python extraction shape [16384, 128]
    // In our NCHW struct, we treat this as [1, 1, 16384, 128] for 2D matrices
    Tensor* dense_weight = create_tensor(1, 1, 16384, 128);

    // 3. Load the binary file
    if (load_weights(dense_weight, target_file)) {
        printf("SUCCESS: Loaded weights into memory.\n");
        // 4. Print verification stats
        print_tensor_info(dense_weight, "noise2feat_w");
    }

    // 5. Cleanup
    free_tensor(dense_weight);

    return 0;
}
