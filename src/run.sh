#!/bin/bash

# Define the total number of iterations
TOTAL_ITERATIONS=100

# Determine the number of available CPU cores for optimal parallelization
# nproc is standard on Linux; for macOS use: $(sysctl -n hw.logicalcpu)
CORES=$(nproc)

echo "Starting $TOTAL_ITERATIONS iterations using $CORES parallel processes..."

# seq generates the range of numbers from 1 to 1000
# xargs -I {} replaces {} with the current number from the sequence
# -P $CORES runs as many processes simultaneously as there are CPU cores
seq 1 $TOTAL_ITERATIONS | xargs -I {} -P $CORES ./sagan.out -w ../bin -c 207 -s {} -o output{}.ppm

echo "Execution complete."
