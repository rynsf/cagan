#!/bin/sh

gcc train_main.c tensor_train.c linalg.c layers.c generator.c discriminator.c losses.c -o sagan_train.out -O3 -lm
gcc train_main_2d.c tensor_train.c linalg.c layers.c conv2d.c attention2d.c generator2d.c discriminator2d.c losses.c debug_vis.c -o sagan_train_2d.out -O3 -lm
