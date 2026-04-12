#ifndef FORWARD_H
#define FORWARD_H

#include "tensor.h"
#include "model.h"
#include "math_ops.h"

// --- ResNet Forward Pass ---
Tensor* forward_resblock(Tensor* x, int class_id, ResBlockWeights* ws) {
    // 1. Path A: Conditional BN -> ReLU -> Upsample -> Conv
    Tensor* gamma1 = create_tensor(1, x->c, 1, 1);
    Tensor* beta1 = create_tensor(1, x->c, 1, 1);
    embedding_lookup(class_id, ws->w_emb1, gamma1);
    embedding_lookup(class_id, ws->b_emb1, beta1);

    Tensor* h1 = create_tensor(1, x->c, x->h, x->w);
    conditional_batch_norm_2d(x, ws->rm1, ws->rv1, gamma1, beta1, h1);
    relu(h1);

    Tensor* h1_up = create_tensor(1, x->c, x->h * 2, x->w * 2);
    nearest_neighbor_upsample(h1, h1_up);

    Tensor* h2 = create_tensor(1, ws->c1_w->n, h1_up->h, h1_up->w);
    conv2d(h1_up, ws->c1_w, ws->c1_b, h2, 1); // padding = 1

    // 2. Path A (Continued): Conditional BN -> ReLU -> Conv
    Tensor* gamma2 = create_tensor(1, h2->c, 1, 1);
    Tensor* beta2 = create_tensor(1, h2->c, 1, 1);
    embedding_lookup(class_id, ws->w_emb2, gamma2);
    embedding_lookup(class_id, ws->b_emb2, beta2);

    Tensor* h3 = create_tensor(1, h2->c, h2->h, h2->w);
    conditional_batch_norm_2d(h2, ws->rm2, ws->rv2, gamma2, beta2, h3);
    relu(h3);

    Tensor* h4 = create_tensor(1, ws->c2_w->n, h3->h, h3->w);
    conv2d(h3, ws->c2_w, ws->c2_b, h4, 1); // padding = 1

    // 3. Path B: Shortcut
    Tensor* s_up = create_tensor(1, x->c, x->h * 2, x->w * 2);
    nearest_neighbor_upsample(x, s_up);

    Tensor* s_conv = create_tensor(1, ws->sc_w->n, s_up->h, s_up->w);
    conv2d(s_up, ws->sc_w, ws->sc_b, s_conv, 0); // padding = 0 (1x1 conv)

    // 4. Output Addition
    Tensor* out = create_tensor(1, h4->c, h4->h, h4->w);
    tensor_add(h4, s_conv, out);

    // 5. Aggressive Memory Cleanup
    free_tensor(gamma1); free_tensor(beta1); free_tensor(h1); free_tensor(h1_up);
    free_tensor(h2); free_tensor(gamma2); free_tensor(beta2); free_tensor(h3);
    free_tensor(h4); free_tensor(s_up); free_tensor(s_conv);

    return out; // Return the final block tensor
}

// --- Self-Attention Forward Pass ---
Tensor* forward_attention(Tensor* x, AttentionWeights* aw) {
    // 1. Theta Path
    Tensor* theta = create_tensor(1, aw->theta_w->n, x->h, x->w);
    conv2d(x, aw->theta_w, NULL, theta, 0);

    // 2. Phi Path (Conv -> MaxPool)
    Tensor* phi_tmp = create_tensor(1, aw->phi_w->n, x->h, x->w);
    conv2d(x, aw->phi_w, NULL, phi_tmp, 0);
    Tensor* phi = create_tensor(1, aw->phi_w->n, x->h / 2, x->w / 2);
    max_pool_2x2(phi_tmp, phi);

    // 3. G Path (Conv -> MaxPool)
    Tensor* g_tmp = create_tensor(1, aw->g_w->n, x->h, x->w);
    conv2d(x, aw->g_w, NULL, g_tmp, 0);
    Tensor* g = create_tensor(1, aw->g_w->n, x->h / 2, x->w / 2);
    max_pool_2x2(g_tmp, g);

    // 4. Multiply and Softmax
    Tensor* attn = create_tensor(1, 1, theta->h * theta->w, phi->h * phi->w);
    attention_bmm1(theta, phi, attn);
    softmax_last_dim(attn);

    // 5. Apply Attention to G
    Tensor* out_g = create_tensor(1, g->c, x->h, x->w);
    attention_bmm2(attn, g, out_g);

    // 6. Final Conv and Gamma scale
    Tensor* out_conv = create_tensor(1, aw->o_w->n, x->h, x->w);
    conv2d(out_g, aw->o_w, NULL, out_conv, 0);

    Tensor* final_out = create_tensor(1, x->c, x->h, x->w);
    float gamma_val = aw->gamma->data[0];
    int total = final_out->n * final_out->c * final_out->h * final_out->w;
    
    // out = x + gamma * out_conv
    for(int i = 0; i < total; i++) {
        final_out->data[i] = x->data[i] + (gamma_val * out_conv->data[i]);
    }

    // 7. Cleanup
    free_tensor(theta); free_tensor(phi_tmp); free_tensor(phi);
    free_tensor(g_tmp); free_tensor(g); free_tensor(attn);
    free_tensor(out_g); free_tensor(out_conv);

    return final_out;
}

#endif
