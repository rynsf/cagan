#include "linalg.h"

#include <math.h>
#include <string.h>

void matmul_forward(const float* a, const float* b, float* c, int m, int k, int n) {
    for (int i = 0; i < m * n; i++) c[i] = 0.0f;

    // Cache-friendly loop order: i -> p -> j
    for (int i = 0; i < m; i++) {
        for (int p = 0; p < k; p++) {
            float a_ip = a[i * k + p];
            const float* b_row = &b[p * n];
            float* c_row = &c[i * n];
            for (int j = 0; j < n; j++) {
                c_row[j] += a_ip * b_row[j];
            }
        }
    }
}

void matmul_backward(const float* a, const float* b, const float* dc,
                    float* da, float* db, int m, int k, int n) {
    memset(da, 0, (size_t)(m * k) * sizeof(float));
    memset(db, 0, (size_t)(k * n) * sizeof(float));

    // dA = dC * B^T
    for (int i = 0; i < m; i++) {
        for (int p = 0; p < k; p++) {
            float sum = 0.0f;
            for (int j = 0; j < n; j++) {
                sum += dc[i * n + j] * b[p * n + j];
            }
            da[i * k + p] += sum;
        }
    }

    // dB = A^T * dC
    for (int p = 0; p < k; p++) {
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int i = 0; i < m; i++) {
                sum += a[i * k + p] * dc[i * n + j];
            }
            db[p * n + j] += sum;
        }
    }
}

void add_bias_forward(float* y, const float* b, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            y[i * cols + j] += b[j];
        }
    }
}

void add_bias_backward(const float* dy, float* db, int rows, int cols) {
    for (int j = 0; j < cols; j++) db[j] = 0.0f;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            db[j] += dy[i * cols + j];
        }
    }
}

void relu_forward(float* x, int size) {
    for (int i = 0; i < size; i++) {
        if (x[i] < 0.0f) x[i] = 0.0f;
    }
}

void relu_backward(const float* x, const float* dy, float* dx, int size) {
    for (int i = 0; i < size; i++) {
        dx[i] = (x[i] > 0.0f) ? dy[i] : 0.0f;
    }
}

void leaky_relu_forward(float* x, int size, float alpha) {
    for (int i = 0; i < size; i++) {
        if (x[i] < 0.0f) x[i] *= alpha;
    }
}

void leaky_relu_backward(const float* x, const float* dy, float* dx, int size, float alpha) {
    for (int i = 0; i < size; i++) {
        dx[i] = (x[i] > 0.0f) ? dy[i] : (alpha * dy[i]);
    }
}

void softmax_forward_rows(float* x, int rows, int cols) {
    for (int r = 0; r < rows; r++) {
        float* row = &x[r * cols];
        float maxv = row[0];
        for (int j = 1; j < cols; j++) if (row[j] > maxv) maxv = row[j];

        float sum = 0.0f;
        for (int j = 0; j < cols; j++) {
            row[j] = expf(row[j] - maxv);
            sum += row[j];
        }

        float inv = 1.0f / (sum + 1e-12f);
        for (int j = 0; j < cols; j++) row[j] *= inv;
    }
}

void softmax_cross_entropy_backward(const float* probs, const int* labels, float* dlogits, int rows, int cols) {
    float inv_rows = 1.0f / (float)rows;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            float grad = probs[i * cols + j];
            if (j == labels[i]) grad -= 1.0f;
            dlogits[i * cols + j] = grad * inv_rows;
        }
    }
}

void adam_update(float* p, float* g, float* m, float* v, int size,
                 float lr, float beta1, float beta2, float eps, int t) {
    float b1_pow = powf(beta1, (float)t);
    float b2_pow = powf(beta2, (float)t);
    float lr_t = lr * sqrtf(1.0f - b2_pow) / (1.0f - b1_pow + 1e-12f);

    for (int i = 0; i < size; i++) {
        m[i] = beta1 * m[i] + (1.0f - beta1) * g[i];
        v[i] = beta2 * v[i] + (1.0f - beta2) * g[i] * g[i];
        p[i] -= lr_t * m[i] / (sqrtf(v[i]) + eps);
        g[i] = 0.0f;
    }
}
