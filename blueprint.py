import torch
import torch.nn.functional as F
import numpy as np

def run_blueprint():
    # 1. LOAD WEIGHTS
    ckpt_path = "sagan_128_woReLUinplace_noaug_bigGAN_imagenet1k_b32x8_Glr1e-4_Dlr-4e-4_ndisc1_20210818_210232-3f5686af.pth"
    print(f"Loading weights from {ckpt_path}...")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get('state_dict', ckpt)
    PREFIX = "generator_ema"

    # Helper: Simulates the C-side "Load Baked Weight" function
    def get_w(name):
        if f"{name}.weight_orig" in state_dict:
            w_orig = state_dict[f"{name}.weight_orig"]
            u = state_dict[f"{name}.weight_u"]
            v = state_dict[f"{name}.weight_v"]
            w_mat = w_orig.view(w_orig.size(0), -1)
            sigma = torch.dot(u, torch.mv(w_mat, v))
            return w_orig / sigma  # Baked Spectral Norm
        elif f"{name}.weight" in state_dict:
            return state_dict[f"{name}.weight"]
        return state_dict[name]

    def get_b(name):
        return state_dict.get(f"{name}.bias", None)

    # 2. DEFINE THE RESNET BLOCK (Your primary C function)
    def resblock(x, y, prefix):
        # Path A: Main Convolutional Branch
        rm1 = state_dict[f"{prefix}.norm_1.norm.running_mean"]
        rv1 = state_dict[f"{prefix}.norm_1.norm.running_var"]
        gamma1 = F.embedding(y, get_w(f"{prefix}.norm_1.weight_embedding"))
        beta1 = F.embedding(y, get_w(f"{prefix}.norm_1.bias_embedding"))

        h = F.batch_norm(x, rm1, rv1, eps=1e-5)
        h = h * (1.0 + gamma1.view(-1, x.shape[1], 1, 1)) + beta1.view(-1, x.shape[1], 1, 1)
        h = F.relu(h)
        h = F.interpolate(h, scale_factor=2.0, mode='nearest')
        h = F.conv2d(h, get_w(f"{prefix}.conv_1.conv"), get_b(f"{prefix}.conv_1.conv"), padding=1)

        rm2 = state_dict[f"{prefix}.norm_2.norm.running_mean"]
        rv2 = state_dict[f"{prefix}.norm_2.norm.running_var"]
        gamma2 = F.embedding(y, get_w(f"{prefix}.norm_2.weight_embedding"))
        beta2 = F.embedding(y, get_w(f"{prefix}.norm_2.bias_embedding"))

        h = F.batch_norm(h, rm2, rv2, eps=1e-5)
        h = h * (1.0 + gamma2.view(-1, h.shape[1], 1, 1)) + beta2.view(-1, h.shape[1], 1, 1)
        h = F.relu(h)
        h = F.conv2d(h, get_w(f"{prefix}.conv_2.conv"), get_b(f"{prefix}.conv_2.conv"), padding=1)

        # Path B: Shortcut Branch
        s = F.interpolate(x, scale_factor=2.0, mode='nearest')
        s = F.conv2d(s, get_w(f"{prefix}.shortcut.conv"), get_b(f"{prefix}.shortcut.conv"))
        
        # Addition
        return h + s

    # 3. INITIALIZE INPUTS
    target_class = 985  # Golden Retriever
    print(f"Executing Forward Pass for Class {target_class}...")
    
    torch.manual_seed(1234)
    z = torch.randn(1, 128).clamp(-1.0, 1.0) 
    y = torch.tensor([target_class])

    # 4. FORWARD PASS SEQUENCE
    # Dense layer mapping
    x = F.linear(z, get_w(f"{PREFIX}.noise2feat"), get_b(f"{PREFIX}.noise2feat"))
    x = x.view(1, 1024, 4, 4)

    # Upsampling
    x = resblock(x, y, f"{PREFIX}.conv_blocks.0") # -> 8x8
    x = resblock(x, y, f"{PREFIX}.conv_blocks.1") # -> 16x16
    x = resblock(x, y, f"{PREFIX}.conv_blocks.2") # -> 32x32
    x = resblock(x, y, f"{PREFIX}.conv_blocks.3") # -> 64x64

    # Self-Attention (Operates at 64x64)
    attn_prefix = f"{PREFIX}.conv_blocks.4"
    theta = F.conv2d(x, get_w(f"{attn_prefix}.theta.conv"))
    phi = F.max_pool2d(F.conv2d(x, get_w(f"{attn_prefix}.phi.conv")), 2)
    g = F.max_pool2d(F.conv2d(x, get_w(f"{attn_prefix}.g.conv")), 2)
    
    B, C_t, H, W = theta.size()
    C_g = g.size(1) 
    
    theta = theta.view(B, C_t, -1).permute(0, 2, 1)
    phi = phi.view(B, C_t, -1)
    attn = F.softmax(torch.bmm(theta, phi), dim=-1)
    
    g = g.view(B, C_g, -1).permute(0, 2, 1)
    out = torch.bmm(attn, g).permute(0, 2, 1).view(B, C_g, H, W)
    out = F.conv2d(out, get_w(f"{attn_prefix}.o.conv"))
    x = x + get_w(f"{attn_prefix}.gamma") * out

    # Final Upsampling
    x = resblock(x, y, f"{PREFIX}.conv_blocks.5") # -> 128x128

    # 5. RGB OUTPUT FORMATTING
    rm_f = state_dict[f"{PREFIX}.to_rgb.bn.running_mean"]
    rv_f = state_dict[f"{PREFIX}.to_rgb.bn.running_var"]
    x = F.batch_norm(x, rm_f, rv_f, get_w(f"{PREFIX}.to_rgb.bn"), get_b(f"{PREFIX}.to_rgb.bn"), eps=1e-5)
    x = F.relu(x)
    x = F.conv2d(x, get_w(f"{PREFIX}.to_rgb.conv"), get_b(f"{PREFIX}.to_rgb.conv"), padding=1)
    x = torch.tanh(x)

    # 6. IMAGE EXPORT
    # Map from [-1, 1] to [0, 255]
    img = ((x[0].detach().numpy().transpose(1, 2, 0) + 1.0) / 2.0 * 255.0).clip(0, 255).astype(np.uint8)
    
    # Fix MMagic's BGR training data format to standard RGB
    img = img[:, :, ::-1]

    filename = "blueprint_output.ppm"
    with open(filename, "wb") as f:
        f.write(f"P6\n128 128\n255\n".encode())
        f.write(img.tobytes())
        
    print(f"Done. Verify {filename}.")

if __name__ == "__main__":
    run_blueprint()
