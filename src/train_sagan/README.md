# SAGAN Training in Pure C (From Scratch)

This module adds a full **training-side implementation scaffold** for Self-Attention GAN style learning in C, matching the spirit of your existing `src` inference code: direct control, explicit loops, and no deep learning framework.

It intentionally keeps every major concept visible:
- explicit tensors with `.data` and `.grad`
- matrix multiplications implemented from scratch
- discriminator and generator as separate modules
- self-attention with explicit Q/K/V projections and softmax
- adversarial losses and optimizer updates in C

This code is designed for understanding and extension first. It is complete conceptually, and runnable as a scaffold, but not yet a production-grade ImageNet trainer.

## Folder Layout

- `tensor_train.h/.c`: tensor object, allocation, gradients, random initialization
- `linalg.h/.c`: matmul forward/backward, activation math, softmax, Adam updates
- `layers.h/.c`: reusable trainable layers (`LinearLayer`, `SelfAttention`)
- `generator.h/.c`: conditional generator path and backprop wiring
- `discriminator.h/.c`: conditional discriminator path and backprop wiring
- `losses.h/.c`: BCE and hinge GAN losses with explicit gradient outputs
- `train_main.c`: training loop orchestration (D step + G step)
- `build_train.sh`: one-command compile script

## Build

From `src/train_sagan`:

```bash
sh build_train.sh
```

This creates:
- `sagan_train.out` (flattened educational trainer)
- `sagan_train_2d.out` (convolutional + 2D self-attention trainer)

## `sagan_train_2d.out` Flags (Full Reference)

- `-b <int>`: batch size (samples per step)
- `-t <int>`: number of training steps
- `-z <int>`: latent noise dimension
- `-C <int>`: base channel width (`base_c`) for conv feature maps
- `-S <int>`: base spatial size (`base_hw`) for generator seed map; output becomes `2S x 2S`
- `-s <int>`: random seed
- `-V <int>`: debug dump interval (`0` disables, `1` every step, `5` every 5 steps)
- `-D <path>`: debug dump output directory

Default values:
- `-b 2`
- `-t 5`
- `-z 64`
- `-C 16`
- `-S 8`
- `-s 69`
- `-V 0`
- `-D debug_out`

Run:

```bash
./sagan_train.out -b 8 -t 200 -z 128 -H 1024 -I 49152 -s 69

# 2D feature-map + self-attention trainer
./sagan_train_2d.out -b 2 -t 5 -z 64 -C 16 -S 8 -s 69

# with visual debug dumps every step
./sagan_train_2d.out -b 1 -t 3 -z 16 -C 4 -S 4 -s 1 -V 1 -D debug_out
```

## What Is Implemented

There are now two training tracks:
- `sagan_train.out`: flattened-feature educational scaffold
- `sagan_train_2d.out`: convolutional feature-map SAGAN path with explicit Q/K/V attention map math and backward pass

## 1) Tensor and gradient memory

`TensorTrain` stores shape and two arrays:
- `data`: forward values
- `grad`: backward gradients

This gives direct ownership over memory and gradient flow without hidden graph engines.

## 2) Matrix math from first principles

`matmul_forward` and `matmul_backward` are handwritten loop nests:
- forward: `C = A * B`
- backward: `dA = dC * B^T`, `dB = A^T * dC`

No BLAS, no external kernels.

## 3) Generator and Discriminator separation

Both networks are standalone structs with explicit trainable components and optimizer state.

Generator (`generator.c`):
- latent projection
- class-conditioning projection
- feature fusion
- self-attention block
- image projection + tanh output

Discriminator (`discriminator.c`):
- image feature projection
- class-conditioning projection
- self-attention block
- real/fake scalar head

## 4) Self-attention concept path

Attention module contains:
- `W_q`, `W_k`, `W_v`, `W_o`
- attention score matrix (`QK^T / sqrt(d)`)
- row-wise softmax
- context = `A * V`
- residual output

The code includes full forward path and scaffolded backward structure with explicit comments where full derivative expansion can be extended.

For the 2D trainer (`train_main_2d.c`), attention is implemented over spatial feature maps:
- conv-generated `theta` (query), `phi` (key), `g` (value)
- attention logits per location pair
- row-wise softmax to form attention map
- attended value aggregation to self-attention feature maps
- projection and residual output
- backward pass through projection, value path, softmax Jacobian, and query/key gradients

## Debugging Forward/Backward Visually

`sagan_train_2d.out` includes debug visualization flags:
- `-V <int>`: dump every N steps (0 disables)
- `-D <path>`: output directory for dumps

Important behavior:
- the debug directory is **not automatically cleared** between runs
- files with the same step name are overwritten
- extra old files can remain when run length/config changes

For clean runs:

```bash
rm -rf debug_out && mkdir -p debug_out
./sagan_train_2d.out -b 1 -t 3 -z 16 -C 4 -S 4 -s 1 -V 1 -D debug_out
```

Or use timestamped output dirs:

```bash
./sagan_train_2d.out -b 1 -t 3 -z 16 -C 4 -S 4 -s 1 -V 1 -D "debug_out_$(date +%Y%m%d_%H%M%S)"
```

When enabled, each dump step writes:
- generated RGB image (`*.ppm`)
- generator feature maps (`feat`, `upsample`, `q`, `k`, `v`, `attn`, self-attn output)
- discriminator feature maps (`input`, early feat, `q`, `k`, `v`, `attn`, self-attn output)
- gradient map (`dL/d(fake_img)` absolute heatmap)

Training also prints detailed phase logs and tensor stats each step.

## Training Logs Explained

Startup logs:
- `2D SAGAN train: batch=... steps=... base=CxSxS out=HxW`
  - `H=W=2S` based on `-S`
- `debug dumps enabled: every=N dir=...`
  - printed only when `-V > 0`

Per-step phase logs:
- `[step NNN] phase=prepare`
  - synthesize real batch and sample latent/class conditions
- `[step NNN] phase=generator_forward`
  - generator creates fake image batch
- `[step NNN] phase=discriminator_real`
  - discriminator real pass + backward contribution
- `[step NNN] phase=discriminator_fake`
  - discriminator fake pass + backward contribution + discriminator optimizer step
- `[step NNN] phase=generator_update`
  - generator loss pass through discriminator signal + generator optimizer step

Tensor stats logs:
- `[stats] g.out(fake_img) ...`
  - reports generated image tensor shape/min/max/mean
- `[stats] dL/d(fake_img) ...`
  - reports gradient signal entering generator output

Step summary:
- `[step NNN] done | d_real=... d_fake=... g=... | time=...ms`
- `d_real`: hinge real term `max(0, 1 - D(real))`
- `d_fake`: hinge fake term `max(0, 1 + D(fake))`
- `g`: generator hinge objective `-mean(D(fake))`
- `time`: step runtime in milliseconds

Debug output filenames:
- Generator: `step_xxxx_fake.ppm`, `step_xxxx_g_feat_ch0.pgm`, `step_xxxx_g_up_ch0.pgm`, `step_xxxx_g_q_ch0.pgm`, `step_xxxx_g_k_ch0.pgm`, `step_xxxx_g_v_ch0.pgm`, `step_xxxx_g_attn_row0.pgm`, `step_xxxx_g_self_attn_ch0.pgm`
- Discriminator: `step_xxxx_d_in_ch0.pgm`, `step_xxxx_d_feat_ch0.pgm`, `step_xxxx_d_q_ch0.pgm`, `step_xxxx_d_k_ch0.pgm`, `step_xxxx_d_v_ch0.pgm`, `step_xxxx_d_attn_row0.pgm`, `step_xxxx_d_self_attn_ch0.pgm`
- Backward: `step_xxxx_grad_dimg_ch0_abs.pgm`

## 5) Losses and training steps

`losses.c` provides:
- BCE with logits variants
- hinge D-real, D-fake, and G loss

`train_main.c` executes:
1. discriminator update on real batch
2. discriminator update on fake batch
3. generator update through discriminator signal

## Why It Is A Scaffold (Current Limits)

- Real image loader is intentionally synthetic (`synthesize_real_batch`) so the training mechanics are clear.
- Attention backward is partially scaffolded (residual path implemented, full Q/K/V Jacobians marked for extension).
- Architecture is compact and vectorized as flattened features rather than full 2D conv-resblocks from the inference graph.

That means this is ideal for understanding and extending, but not yet equivalent to large-scale SAGAN ImageNet training.


## Optional Tools To Install

You can keep this dependency-free, but these tools help development:
- `clang` or `gcc` (compiler)
- `gdb` (debugger)
- `valgrind` (memory diagnostics)
- `python3` (optional scripts for dataset conversion/checkpoint inspection)

Ubuntu example:

```bash
sudo apt update
sudo apt install -y build-essential gdb valgrind python3
```
