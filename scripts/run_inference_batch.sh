#!/bin/sh

TOTAL_ITERATIONS=100
CORES=$(nproc)
WEIGHTS_DIR="./sagan_128_imagenet"

echo "Starting $TOTAL_ITERATIONS iterations using $CORES parallel processes..."

seq 1 "$TOTAL_ITERATIONS" | xargs -I {} -P "$CORES" ./bin/sagan.out -w "$WEIGHTS_DIR" -c 207 -s {} -o outputs/output{}.ppm

echo "Execution complete."
