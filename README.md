# SAGAN-C 

This repository contains two pure-C SAGAN codebases:
- inference engine for image generation
- training scaffold for GAN learning simulation

No deep learning frameworks are used; all tensor math and model logic are implemented in C.

## Project Layout

- `src/inference/`: SAGAN inference source (`main.c`, model/tensor/forward headers)
- `src/training/`: SAGAN training scaffold source (1D and 2D trainer variants)
- `scripts/`: canonical build/run scripts
- `docs/`: focused documentation per workflow
- `bin/`: compiled executables (build output)
- `outputs/`: generated images and debug dumps

Compatibility wrappers are preserved:
- `src/build.sh` -> calls `scripts/build_inference.sh`
- `src/run.sh` -> calls `scripts/run_inference_batch.sh`
- `src/training/build_train.sh` -> calls `scripts/build_training.sh`

## Quick Start

Build inference binary:

```bash
sh scripts/build_inference.sh
```

Run batch inference using weights from `./sagan_128_imagenet/`:

```bash
sh scripts/run_inference_batch.sh
```

Build training binaries:

```bash
sh scripts/build_training.sh
```

Compatibility wrapper commands:

```bash
sh src/build.sh
sh src/run.sh
sh src/training/build_train.sh
```

Notes:
- `scripts/run_inference_batch.sh` writes images to `outputs/output*.ppm`.
- `scripts/build_training.sh` builds `bin/sagan_train.out` and `bin/sagan_train_2d.out`.
- There is currently no dedicated `.sh` launcher for single-image inference or for starting the training binaries.

## More Documentation

- Inference details: `docs/inference.md`
- Training details: `docs/training.md`
# cagan
# cagan
