#ifndef TENSOR_H
#define TENSOR_H

#include <stdio.h>
#include <stdlib.h>

// --- Data Structure ---
typedef struct {
    int n; // Batch
    int c; // Channels
    int h; // Height
    int w; // Width
    float* data; // Flat 1D array holding the NCHW data
} Tensor;

// --- Memory Management ---
Tensor* create_tensor(int n, int c, int h, int w) {
    Tensor* t = (Tensor*)malloc(sizeof(Tensor));
    t->n = n; t->c = c; t->h = h; t->w = w;
    // calloc sets memory to 0, which is safer for debugging
    t->data = (float*)calloc(n * c * h * w, sizeof(float));
    if (!t->data) {
        printf("FATAL: Memory allocation failed for tensor.\n");
        exit(1);
    }
    return t;
}

void free_tensor(Tensor* t) {
    if (t) {
        if (t->data) free(t->data);
        free(t);
    }
}

// --- The Macro for 4D Indexing ---
// Usage: float val = T_AT(t, n, c, y, x);
#define T_AT(t, n_idx, c_idx, y_idx, x_idx) \
    ((t)->data[(((n_idx) * (t)->c + (c_idx)) * (t)->h + (y_idx)) * (t)->w + (x_idx)])

// --- Binary Loader ---
int load_weights(Tensor* t, const char* filepath) {
    FILE* f = fopen(filepath, "rb");
    if (!f) {
        printf("ERROR: Could not open %s\n", filepath);
        return 0; // Fail
    }
    
    size_t elements = (size_t)t->n * t->c * t->h * t->w;
    size_t read_count = fread(t->data, sizeof(float), elements, f);
    fclose(f);
    
    if (read_count != elements) {
        printf("ERROR: Mismatch in %s. Expected %zu floats, got %zu.\n", filepath, elements, read_count);
        return 0;
    }
    return 1; // Success
}

// --- Verification & Printing ---
void print_tensor_info(Tensor* t, const char* name) {
    printf("Tensor [%s]: Shape [%d, %d, %d, %d] | Total Elements: %d\n", 
           name, t->n, t->c, t->h, t->w, t->n * t->c * t->h * t->w);
    
    printf("First 5 values: ");
    int limit = (t->n * t->c * t->h * t->w < 5) ? (t->n * t->c * t->h * t->w) : 5;
    for (int i = 0; i < limit; i++) {
        printf("%f ", t->data[i]);
    }
    printf("\n---------------------------------------------------\n");
}

#endif
