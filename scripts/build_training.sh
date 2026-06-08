#!/bin/sh

gcc src/training/train_main.c src/training/tensor_train.c src/training/linalg.c src/training/layers.c src/training/generator.c src/training/discriminator.c src/training/losses.c -o bin/sagan_train.out -O3 -lm
gcc src/training/train_main_2d.c src/training/tensor_train.c src/training/linalg.c src/training/layers.c src/training/conv2d.c src/training/attention2d.c src/training/generator2d.c src/training/discriminator2d.c src/training/losses.c src/training/debug_vis.c -o bin/sagan_train_2d.out -O3 -lm
