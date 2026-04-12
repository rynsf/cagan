#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "tensor.h"
#include "model.h"
#include "math_ops.h"
#include "forward.h"

#define M_PI 3.14159265358979323846

// Generates normal distribution noise matching torch.randn().clamp(-1.0, 1.0)
float randn_clamped() {
    float u1 = (float)rand() / RAND_MAX;
    float u2 = (float)rand() / RAND_MAX;
    if (u1 == 0.0f) u1 = 1e-7f; // Prevent log(0)
    
    // Box-Muller transform
    float z = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * (float)M_PI * u2);
    
    // Truncation trick
    if (z > 1.0f) z = 1.0f;
    if (z < -1.0f) z = -1.0f;
    return z;
}

// Exports the C-Tensor as an image file
void save_ppm(Tensor* img, const char* filename) {
    FILE* f = fopen(filename, "wb");
    fprintf(f, "P6\n%d %d\n255\n", img->w, img->h);
    int spatial = img->h * img->w;
    
    for (int i = 0; i < spatial; i++) {
        // MMagic outputs BGR. So Channel 0 is Blue, 1 is Green, 2 is Red.
        // We read them backwards to write standard RGB to the file.
        float b_val = img->data[0 * spatial + i];
        float g_val = img->data[1 * spatial + i];
        float r_val = img->data[2 * spatial + i];

        // Map [-1, 1] back to [0, 255]
        int r = (int)(((r_val + 1.0f) / 2.0f) * 255.0f);
        int g = (int)(((g_val + 1.0f) / 2.0f) * 255.0f);
        int b = (int)(((b_val + 1.0f) / 2.0f) * 255.0f);

        // Safety clamp
        r = r > 255 ? 255 : (r < 0 ? 0 : r);
        g = g > 255 ? 255 : (g < 0 ? 0 : g);
        b = b > 255 ? 255 : (b < 0 ? 0 : b);

        fputc(r, f); fputc(g, f); fputc(b, f);
    }
    fclose(f);
}

int main() {
    printf("Initializing SAGAN Pure C Inference Engine...\n");
    srand(42); // Set seed for reproducibility

    SAGAN_Weights* model = load_all_weights("../bin");
    printf("[SUCCESS] Model Loaded. Commencing Generation...\n\n");

    int target_class = 985; 
    printf("Target Class: %d\n", target_class);

    // 1. Create Latent Space (Noise Vector)
    Tensor* z = create_tensor(1, 128, 1, 1);
    for (int i = 0; i < 128; i++) z->data[i] = randn_clamped();

    // 2. Dense Layer mapping -> 4x4
    printf("1/7: Dense Layer...\n");
    Tensor* dense_out = create_tensor(1, 16384, 1, 1);
    linear_layer(z, model->n2f_w, model->n2f_b, dense_out);
    free_tensor(z);
    
    // Reshape [1, 16384, 1, 1] into [1, 1024, 4, 4]
    Tensor* x4 = create_tensor(1, 1024, 4, 4);
    for (int i=0; i<16384; i++) x4->data[i] = dense_out->data[i];
    free_tensor(dense_out);

    // 3. Upsampling Stages
    printf("2/7: ResNet Block 0 (4x4 -> 8x8)...\n");
    Tensor* x8 = forward_resblock(x4, target_class, &model->blocks[0]);
    free_tensor(x4);

    printf("3/7: ResNet Block 1 (8x8 -> 16x16)...\n");
    Tensor* x16 = forward_resblock(x8, target_class, &model->blocks[1]);
    free_tensor(x8);

    printf("4/7: ResNet Block 2 (16x16 -> 32x32)...\n");
    Tensor* x32 = forward_resblock(x16, target_class, &model->blocks[2]);
    free_tensor(x16);

    printf("5/7: ResNet Block 3 (32x32 -> 64x64)...\n");
    Tensor* x64 = forward_resblock(x32, target_class, &model->blocks[3]);
    free_tensor(x32);

    // 4. Self-Attention (Operates at 64x64)
    printf("6/7: Self-Attention Block...\n");
    Tensor* x64_attn = forward_attention(x64, &model->attn);
    free_tensor(x64);

    // 5. Final Upsample
    printf("7/7: ResNet Block 5 (64x64 -> 128x128)...\n");
    Tensor* x128 = forward_resblock(x64_attn, target_class, &model->blocks[5]);
    free_tensor(x64_attn);

    // 6. RGB Output Stage
    printf("Formatting Output RGB Image...\n");
    Tensor* rgb_bn = create_tensor(1, x128->c, x128->h, x128->w);
    batch_norm_2d(x128, model->rgb_rm, model->rgb_rv, model->rgb_w, model->rgb_b, rgb_bn);
    free_tensor(x128);
    relu(rgb_bn);

    Tensor* final_img = create_tensor(1, 3, 128, 128);
    conv2d(rgb_bn, model->rgb_conv_w, model->rgb_conv_b, final_img, 1);
    free_tensor(rgb_bn);
    tanh_activation(final_img);

    // 7. Save and Cleanup
    const char* out_file = "pure_c_dog.ppm";
    save_ppm(final_img, out_file);
    free_tensor(final_img);
    free_model(model);

    printf("\n[VICTORY] Image successfully saved to: %s\n", out_file);
    return 0;
}
