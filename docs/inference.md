# Inference System (Pure C SAGAN)

## Build

```bash
sh scripts/build_inference.sh
```

Produces: `bin/sagan.out`

## CLI

```bash
./bin/sagan.out -w <weight_dir> -c <class_id> -s <seed> -o <output_file>
```

Options:
- `-w`: directory with float32 `.bin` weights (default in code: `sagan_128_imagenet/`)
- `-c`: ImageNet class id `0..999`
- `-s`: RNG seed
- `-o`: output `.ppm` file path

Example:

```bash
./bin/sagan.out -w ./sagan_128_imagenet -c 1 -s 99 -o outputs/goldfish.ppm
```

## Batch Run Helper

```bash
sh scripts/run_inference_batch.sh
```

This runs 100 parallelized generations using all CPU cores and writes `outputs/output*.ppm` using weights from `sagan_128_imagenet/`.
