#include <stdio.h>
#include "tensor.h"
#include "model.h"

int main() {
    printf("Initializing SAGAN C-Engine...\n");

    // The directory where you exported your .bin files
    const char* weight_dir = "../bin";

    // 1. Load the entire model
    SAGAN_Weights* model = load_all_weights(weight_dir);
    printf("\n[SUCCESS] Entire 150MB model successfully loaded into RAM.\n");

    // 2. Quick validation of a deep layer (e.g., the Attention Gamma scalar)
    print_tensor_info(model->attn.gamma, "Attention Gamma");

    // 3. Quick validation of the final RGB Bias
    print_tensor_info(model->rgb_conv_b, "To RGB Conv Bias");

    printf("\nMemory is primed. Ready for inference math.\n");

    // We will add the free_model() function later, the OS reclaims RAM on exit for now.
    free_model(model);
    return 0;
}
