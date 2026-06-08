# Training Scaffold (Pure C SAGAN)

## Build

```bash
sh scripts/build_training.sh
```

Produces:
- `bin/sagan_train.out`
- `bin/sagan_train_2d.out`

## 1D Educational Trainer

```bash
./bin/sagan_train.out -b 8 -t 200 -z 128 -H 1024 -I 49152 -s 69
```

Flags:
- `-b`: batch size
- `-t`: train steps
- `-z`: latent dimension
- `-H`: hidden width
- `-I`: flattened image dimension
- `-s`: random seed

## 2D Conv + Attention Trainer

```bash
./bin/sagan_train_2d.out -b 2 -t 5 -z 64 -C 16 -S 8 -s 69
```

Additional flags:
- `-C`: base channels
- `-S`: base spatial size (output becomes `2S x 2S`)
- `-V`: debug dump interval (`0` disables)
- `-D`: debug dump output directory

Debug example:

```bash
./bin/sagan_train_2d.out -b 1 -t 3 -z 16 -C 4 -S 4 -s 1 -V 1 -D outputs/debug_out
```

## Source Layout

- `src/training/train_main.c`: 1D scaffold training loop
- `src/training/train_main_2d.c`: 2D conv + self-attention training loop
- `src/training/layers.c`, `src/training/linalg.c`, `src/training/losses.c`: math/layer/loss internals
