# SAGAN-C: Bare-Metal ImageNet Generation

This repository contains a pure, dependency-free C implementation of a Self-Attention Generative Adversarial Network (SAGAN). It is capable of generating $128 \times 128$ images across 1000 ImageNet classes from pure mathematical first principles.

There is no Python, no PyTorch, no BLAS/LAPACK, and no external matrix libraries. Every convolution, batch normalization, and attention matrix multiplication is written from scratch using standard C arrays and pointers.

The weights used in this engine are derived from the OpenMMLab (MMagic) BigGAN-schedule SAGAN implementation, utilizing Exponential Moving Average (EMA) and Spectral Normalization (baked entirely offline).

---

## 🚀 Usage

### Compilation
Because this engine relies heavily on deeply nested `for` loops for convolutions and matrix multiplications, compiler optimization (`-O3`) is strictly required to enable auto-vectorization (AVX/SSE).

```bash
gcc src/main.c -o sagan_c -O3 -lm
```

### Command Line Interface
The executable acts as a standalone CLI tool for image generation.

```bash
./sagan_c -w <weight_dir> -c <class_id> -s <seed> -o <output_file>
```

**Options:**
* `-w` : Directory containing the `.bin` baked weight files (Default: `c_ready_128_bin`).
* `-c` : ImageNet Class ID from 0 to 999. (Default: 207, Golden Retriever).
* `-s` : Random seed for the latent space generation (Default: 42).
* `-o` : Output filename for the generated image (Default: `output.ppm`).

**Example:** Generating a Goldfish (Class 1) with seed 99:
```bash
./sagan_c -c 1 -s 99 -o goldfish.ppm
```

---

## 🧠 Architecture Deep Dive: Math to Memory

This implementation maps high-level deep learning abstractions directly to 1D memory blocks. Below is a detailed breakdown of how the SAGAN network computes an image, down to the matrix level.

### 1. The Tensor Engine (NCHW Memory Layout)
Neural networks operate on 4D Tensors: Batch ($N$), Channels ($C$), Height ($H$), and Width ($W$). Because C does not natively support dynamic multi-dimensional arrays efficiently, all 4D tensors are flattened into contiguous 1D `float` arrays.

To simulate 4D space, we use a stride-based indexing macro:
```c
#define T_AT(t, n, c, y, x) \
    ((t)->data[(((n) * (t)->c + (c)) * (t)->h + (y)) * (t)->w + (x)])
```
This guarantees cache-friendly, contiguous memory access during sequential spatial operations.

### 2. The Latent Space & Truncation
Generation begins with a latent noise vector $z \in \mathbb{R}^{128}$. We generate this using a Box-Muller transform to simulate a normal distribution. 

To ensure high-fidelity (realistic) anatomy, we apply the **Truncation Trick**. The noise values are mathematically clamped:
$z_{i} = \max(-1.0, \min(1.0, z_{i}))$

This vector is projected via a dense linear layer:
$y = z W^T + b$
Where $W \in \mathbb{R}^{16384 \times 128}$. The resulting $16384$-element vector is reshaped into our base $1024 \times 4 \times 4$ spatial grid.

### 3. Conditional Batch Normalization (The "Brain" of the Class)
Standard Convolutions only know how to draw textures. **Conditional Batch Norm (CBN)** is how we tell the network *what* to draw (e.g., a dog vs. a plane).

For a given feature map $x$, we first normalize it across the spatial dimensions:
$$ \hat{x} = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} $$

Next, we perform an embedding lookup based on the user's `class_id`. This retrieves a specific $\gamma$ (scale) and $\beta$ (shift) vector for that specific ImageNet class. We apply this to the normalized tensor:
$$ y = \gamma_{class} \hat{x} + \beta_{class} $$

**C-Optimization Note:** Calculating $\sqrt{x}$ and division for every pixel is incredibly slow. In our implementation, the `conditional_batch_norm_2d` function pre-calculates a single `scale` and `shift` constant for each channel, reducing the inner spatial loop to a simple $y = x \times \text{scale} + \text{shift}$ operation.

### 4. 2D Convolutions (Direct Computation)
Instead of utilizing memory-heavy `im2col` matrix unrolling, this engine computes convolutions directly using a 7-deep nested loop. 

For an output tensor $O$, input $I$, and kernel weights $W$, the mathematical operation for a pixel at $(y, x)$ in output channel $o$ is:
$$ O_{o, y, x} = b_o + \sum_{i=0}^{C_{in}-1} \sum_{ky=0}^{K_h-1} \sum_{kx=0}^{K_w-1} I_{i, y+ky-p, x+kx-p} \times W_{o, i, ky, kx} $$
*(Where $p$ is the padding).*

This approach requires exactly **zero temporary memory allocations**, keeping the memory footprint strictly bound to the size of the layer tensors.

### 5. The Self-Attention Mechanism
Traditional convolutions have a strictly local "receptive field" (usually $3 \times 3$). If the network is drawing a dog, a convolution drawing the back legs cannot "see" what the convolutions drawing the front legs are doing. 

The **Self-Attention** block (operating at the $64 \times 64$ resolution stage) fixes this by computing the relationship between *every* pixel and *every other* pixel.

1. **Feature Projections:** The input $x$ is convolved into three separate spaces using $1 \times 1$ convolutions:
   * **Theta ($\theta$):** The "Query"
   * **Phi ($\phi$):** The "Key"
   * **G ($g$):** The "Value"

2. **Downsampling for Efficiency:** At $64 \times 64$, there are $N = 4096$ spatial locations. An attention map of $4096 \times 4096$ requires enormous memory. Our implementation applies a $2 \times 2$ MaxPool to $\phi$ and $g$, reducing their spatial dimensions to $1024$. 

3. **The Attention Map (BMM 1):** We flatten the spatial dimensions and compute the dot product between $\theta$ and $\phi$. 
   $$ \beta_{j, i} = \text{softmax}(\theta(x_i)^T \phi(x_j)) $$
   This results in a $4096 \times 1024$ matrix representing the "attention" each pixel should pay to every other downsampled region.

4. **Applying Values (BMM 2):** The attention probabilities are multiplied against the value tensor $g$:
   $$ o_j = \sum_{i=1}^{N} \beta_{j, i} g(x_i) $$

5. **Final Output:** The result is scaled by a learnable scalar $\gamma$ and added back to the original input (a residual connection):
   $$ y_i = \gamma o_i + x_i $$

In the C code, `torch.bmm` (Batch Matrix Multiply) is simulated by iterating through our 1D flattened arrays using customized stride offsets, eliminating the need to actively transpose memory blocks.

### 6. Output Formatting and BGR Conversion
The final layer utilizes a hyperbolic tangent function:
$$ \tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}} $$
This safely bounds the unbounded neural network floats to a strict $[-1.0, 1.0]$ range.

To generate a viewable image file, we linearly scale this range to standard $[0, 255]$ 8-bit integers. Because the original OpenMMLab model was trained using OpenCV (which defaults to BGR color space), our C-engine manually reverses the channel reading order (from $0, 1, 2$ to $2, 1, 0$) before writing the bytes to a `.ppm` (Portable Pixmap) file.
