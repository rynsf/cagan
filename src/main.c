#include <stdio.h>
#include "tensor.h"
#include "model.h"
#include "math_ops.h" // Include your new math ops

int main() {
    printf("Initializing SAGAN C-Engine...\n");
    const char* weight_dir = "../bin";
    SAGAN_Weights* model = load_all_weights(weight_dir);
    printf("[SUCCESS] Entire 150MB model successfully loaded into RAM.\n\n");

    // --- SANITY CHECK 1: Linear Layer ---
    printf("Executing Sanity Check: Linear Layer...\n");
    
    // Create an input latent vector [1, 128]
    Tensor* latent_z = create_tensor(1, 128, 1, 1);
    for (int i = 0; i < 128; i++) {
        latent_z->data[i] = 0.5f; // Fill with a static test value
    }

    // Allocate the output tensor [1, 16384]
    Tensor* dense_out = create_tensor(1, 16384, 1, 1);

    // Run the math!
    linear_layer(latent_z, model->n2f_w, model->n2f_b, dense_out);

    // Print the results to verify it didn't crash or produce garbage
    print_tensor_info(dense_out, "Linear Layer Output");

    // Cleanup
    free_tensor(latent_z);
    free_tensor(dense_out);
    free_model(model);

    printf("\nSanity check complete. Exiting clean.\n");
    return 0;
}
