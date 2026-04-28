# SAGAN-C: Dependency-Free ImageNet Generation

This repository contains a pure, dependency-free C implementation of a Self-Attention Generative Adversarial Network (SAGAN). It is capable of generating 128x128 images across 1000 ImageNet classes from mathematical first principles.

There is no Python, no PyTorch, no BLAS/LAPACK, and no external matrix libraries utilized. Every convolution, batch normalization, and attention matrix multiplication is implemented from scratch using standard C arrays and pointers.

---

## 🚀 Usage

### Compilation
Because this program relies heavily on nested `for` loops for convolutions and matrix multiplications, compiler optimization (`-O3`) is strictly required to enable auto-vectorization (AVX/SSE).

```bash
gcc src/main.c -o sagan_c -O3 -lm
```

### Command Line Interface
The executable operates as a standalone CLI tool for image generation.

```bash
./sagan_c -w <weight_dir> -c <class_id> -s <seed> -o <output_file>
```

**Options:**
* `-w` : Directory containing the `.bin` baked weight files (Default: `c_ready_128_bin`).
* `-c` : ImageNet Class ID from 0 to 999. (Default: 207, Golden Retriever).
* `-s` : Random seed for the latent space initialization (Default: 42).
* `-o` : Output filename for the generated image (Default: `output.ppm`).

**Example:** Generating a Goldfish (Class 1) with seed 99:
```bash
./sagan_c -c 1 -s 99 -o goldfish.ppm
```

---

## 🧠 Architecture Deep Dive: Mathematical Implementation

This implementation maps high-level deep learning abstractions directly to 1D contiguous memory blocks. Below is a detailed breakdown of how the SAGAN network computes an image at the matrix level.

### 1. Tensor Data Structure (NCHW Memory Layout)
Neural networks operate on 4D Tensors: Batch, Channels, Height, and Width. Because C does not natively support dynamic multi-dimensional arrays efficiently, all 4D tensors are flattened into contiguous 1D `float` arrays.

To simulate 4D memory allocation, we use a stride-based indexing macro:
```c
#define T_AT(t, n, c, y, x) \
    ((t)->data[(((n) * (t)->c + (c)) * (t)->h + (y)) * (t)->w + (x)])
```
This guarantees cache-friendly, contiguous memory access during sequential spatial operations.

### 2. The Latent Space & Truncation
Generation begins with a latent noise vector <img src="https://latex.codecogs.com/svg.latex?\color{white}z%20\in%20\mathbb{R}^{128}" align="absmiddle" />. We generate this using a Box-Muller transform to simulate a normal distribution. 

To ensure anatomical coherence, we apply the Truncation Trick. The noise values are bounded using a clamp function:

<p align="center">
  <img src="https://latex.codecogs.com/svg.latex?\color{white}z_{i}%20=%20\max(-1.0,%20\min(1.0,%20z_{i}))" />
</p>

This vector is projected via a dense linear layer:

<p align="center">
  <img src="https://latex.codecogs.com/svg.latex?\color{white}y%20=%20z%20W^T%20%2B%20b" />
</p>

Where <img src="https://latex.codecogs.com/svg.latex?\color{white}W%20\in%20\mathbb{R}^{16384%20\times%20128}" align="absmiddle" />. The resulting 16,384-element vector is reshaped into our base 1024x4x4 spatial grid.

### 3. Class-Specific Feature Modulation (Conditional Batch Normalization)
Standard convolutions process spatial textures. Conditional Batch Normalization (CBN) modulates these features based on the target class.

For a given feature map `x`, we first normalize it across the spatial dimensions:

<p align="center">
  <img src="https://latex.codecogs.com/svg.latex?\color{white}\hat{x}%20=%20\frac{x%20-%20\mu}{\sqrt{\sigma^2%20%2B%20\epsilon}}" />
</p>

Next, we perform an embedding lookup based on the user's `class_id`. This retrieves a specific <img src="https://latex.codecogs.com/svg.latex?\color{white}\gamma" align="absmiddle" /> (scale) and <img src="https://latex.codecogs.com/svg.latex?\color{white}\beta" align="absmiddle" /> (shift) vector for that specific ImageNet class. We apply this to the normalized tensor:

<p align="center">
  <img src="https://latex.codecogs.com/svg.latex?\color{white}y%20=%20\gamma_{class}%20\hat{x}%20%2B%20\beta_{class}" />
</p>


**C-Optimization Note:** Computing the <img src="https://latex.codecogs.com/svg.latex?\color{white}\sqrt{x}" align="absmiddle" /> and division for every pixel introduces high latency. In our implementation, the `conditional_batch_norm_2d` function pre-computes a single `scale` and `shift` constant for each channel, reducing the inner spatial loop to a simple <img src="https://latex.codecogs.com/svg.latex?\color{white}y%20=%20x%20\times%20\text{scale}%20%2B%20\text{shift}" align="absmiddle" /> operation.

### 4. 2D Convolutions (Direct Computation)
Instead of utilizing memory-heavy `im2col` matrix unrolling, this engine computes convolutions directly using a nested loop hierarchy. 

For an output tensor `O`, input `I`, and kernel weights `W`, the mathematical operation for a pixel at coordinates `(y, x)` in output channel `o` is:

<p align="center">
  <img src="https://latex.codecogs.com/svg.latex?\color{white}O_{o,%20y,%20x}%20=%20b_o%20%2B%20\sum_{i=0}^{C_{in}-1}%20\sum_{ky=0}^{K_h-1}%20\sum_{kx=0}^{K_w-1}%20I_{i,%20y%2Bky-p,%20x%2Bkx-p}%20\times%20W_{o,%20i,%20ky,%20kx}" />
</p>

*(Where `p` is the padding).*

This approach requires exactly zero temporary memory allocations, keeping the memory footprint strictly bound to the size of the pre-allocated layer tensors.

### 5. The Self-Attention Mechanism
Traditional convolutions have a strictly local receptive field (e.g., 3x3). The Self-Attention block (operating at the 64x64 resolution stage) computes the relationship between every pixel coordinate across the entire spatial grid.

1. **Feature Projections:** The input is convolved into three separate feature spaces using 1x1 convolutions:
   * **Theta (<img src="https://latex.codecogs.com/svg.latex?\color{white}\theta" align="absmiddle" />):** The "Query"
   * **Phi (<img src="https://latex.codecogs.com/svg.latex?\color{white}\phi" align="absmiddle" />):** The "Key"
   * **G:** The "Value"

2. **Downsampling for Efficiency:** At 64x64, there are `N = 4096` spatial locations. An attention map of 4096x4096 is highly memory-intensive. Our implementation applies a 2x2 MaxPool to the Key (<img src="https://latex.codecogs.com/svg.latex?\color{white}\phi" align="absmiddle" />) and Value matrices, reducing their spatial dimensions to 1024. 

3. **The Attention Map (BMM 1):** We flatten the spatial dimensions and compute the dot product between the Query (<img src="https://latex.codecogs.com/svg.latex?\color{white}\theta" align="absmiddle" />) and Key (<img src="https://latex.codecogs.com/svg.latex?\color{white}\phi" align="absmiddle" />) matrices. 

<p align="center">
  <img src="https://latex.codecogs.com/svg.latex?\color{white}\beta_{j,%20i}%20=%20\text{softmax}(\theta(x_i)^T%20\phi(x_j))" />
</p>

   This results in a 4096x1024 matrix representing the attention probability distribution.

4. **Applying Values (BMM 2):** The attention probabilities are multiplied against the Value tensor:

<p align="center">
  <img src="https://latex.codecogs.com/svg.latex?\color{white}o_j%20=%20\sum_{i=1}^{N}%20\beta_{j,%20i}%20g(x_i)" />
</p>

5. **Final Output:** The result is scaled by a learnable scalar variable and added back to the original input as a residual connection:

<p align="center">
  <img src="https://latex.codecogs.com/svg.latex?\color{white}y_i%20=%20\gamma%20o_i%20%2B%20x_i" />
</p>

In the C code, batch matrix multiplication is computed by iterating through the 1D flattened arrays using customized stride offsets, eliminating the requirement to actively transpose memory blocks.

### 6. Output Formatting and BGR Conversion
The final layer utilizes a hyperbolic tangent function:

<p align="center">
  <img src="https://latex.codecogs.com/svg.latex?\color{white}\tanh(x)%20=%20\frac{e^x%20-%20e^{-x}}{e^x%20%2B%20e^{-x}}" />
</p>

This bounds the neural network floating-point values to a strict `[-1.0, 1.0]` range.

To format a valid image file, we linearly scale this range to standard `[0, 255]` 8-bit integers. Because the original OpenMMLab training pipeline utilized OpenCV (which defaults to BGR color space), our engine explicitly reverses the channel reading order before writing the byte data to the `.ppm` (Portable Pixmap) specification.
