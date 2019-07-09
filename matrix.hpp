/***************************************************************************
start of matrix.hpp
***************************************************************************/
#include <cuda_runtime.h>
#include <bitset>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include "half.hpp"
using half_float::half;
typedef bool bit;
using namespace std;

#define CHECK(call)                                                                      \
    {                                                                                    \
        const cudaError_t error = call;                                                  \
        if (error != cudaSuccess) {                                                      \
            fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);                       \
            fprintf(stderr, "code: %d, reason: %s\n", error, cudaGetErrorString(error)); \
            exit(1);                                                                     \
        }                                                                                \
    }

#ifndef __MATRIX__
#define __MATRIX__

enum Layout { NCHW, NHWC };

template <typename T>
class Matrix {
   private:
    T *m_data;
    T *m_d_data;
    Layout m_layout;
    bool m_isTransposed;
    int m_num_batch;
    int m_num_channel;
    int m_num_row;
    int m_num_col;

   public:
    Matrix(Layout layout, int num_batch, int num_channel, int rows, int cols, bool isTransposed = false);

    ~Matrix();

    void randomize(double mean, double scale, int sparsity = 0);

    void randomize(int max_value);

    int index(int batch, int channel, int row, int col);

    T *data();

    T *d_data();

    void set(int batch, int channel, int row, int col, T val);

    T get(int batch, int channel, int row, int col);

    int num_row();

    int num_col();

    int num_batch();

    int num_channel();

    void print(const char *name);

    bool matrix_compare(const char *name, Matrix<T> &ref_matrix, float max_error = 1.e-4);

    void d_cudaMemcpy();
};

#endif

template <typename T>
Matrix<T>::Matrix(Layout layout, int num_batch, int num_channel, int rows, int cols, bool isTransposed)
    : m_layout(layout),
      m_num_batch(num_batch),
      m_num_channel(num_channel),
      m_num_row(rows),
      m_num_col(cols),
      m_isTransposed(isTransposed) {
    m_data = (T *)malloc(num_batch * num_channel * rows * cols * sizeof(T));
    CHECK(cudaMalloc((int **)&m_d_data, num_batch * num_channel * rows * cols * sizeof(T)));
}

template <typename T>
Matrix<T>::~Matrix() {
    CHECK(cudaFree(m_d_data));
    free(m_data);
}

template <typename T>
void Matrix<T>::d_cudaMemcpy() {
    size_t nBytes = num_batch() * num_channel() * num_row() * num_col() * sizeof(T);
    CHECK(cudaMemset(m_d_data, 0, nBytes));
    CHECK(cudaMemcpy(m_data, m_d_data, nBytes, cudaMemcpyHostToDevice));
}

template <typename T>
void Matrix<T>::randomize(double mean, double scale, int sparsity) {
    if (is_same<T, float>::value || is_same<T, double>::value) {
        for (int batch = 0; batch < num_batch(); batch++) {
            for (int channel = 0; channel < num_channel(); channel++) {
                for (int row = 0; row < num_row(); row++) {
                    for (int col = 0; col < num_col(); col++) {
                        T r;
                        if ((rand() % 100) < sparsity) {
                            set(batch, channel, row, col, 0.f);
                        } else {
                            // Generate a random number from 0 to 1.0
                            r = static_cast<T>(rand()) / static_cast<T>(RAND_MAX);

                            // Convert to -.5 to .5
                            r -= 0.5;

                            // Scale and shift
                            r = r * scale + mean;

                            set(batch, channel, row, col, r);
                        }
                    }
                }
            }
        }
    } else {
        printf("Type error: randomize() only supports floating-point types\n");
    }
}

template <>
void Matrix<half>::randomize(double mean, double scale, int sparsity) {
    for (int batch = 0; batch < num_batch(); batch++) {
        for (int channel = 0; channel < num_channel(); channel++) {
            for (int row = 0; row < num_row(); row++) {
                for (int col = 0; col < num_col(); col++) {
                    // Generate a random number from 0 to 1.0
                    if ((rand() % 100) < sparsity) {
                        half a(0.0);
                        set(batch, channel, row, col, a);
                    } else {
                        float f = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
                        f -= 0.5;
                        f = f * scale + mean;
                        set(batch, channel, row, col, half(f));
                    }
                }
            }
        }
    }
}

template <typename T>
void Matrix<T>::randomize(int max_value) {
    printf("Type error: randomize(int max_value) only supports integer type\n");
}

template <>
void Matrix<int>::randomize(int max_value) {
    for (int batch = 0; batch < num_batch(); batch++) {
        for (int channel = 0; channel < num_channel(); channel++) {
            for (int row = 0; row < num_row(); row++) {
                for (int col = 0; col < num_col(); col++) {
                    set(batch, channel, row, col, rand() % (max_value + 1));
                }
            }
        }
    }
}

template <>
void Matrix<bit>::randomize(int max_value) {
    assert(max_value == 0 || max_value == 1);
    for (int batch = 0; batch < num_batch(); batch++) {
        for (int channel = 0; channel < num_channel(); channel++) {
            for (int row = 0; row < num_row(); row++) {
                for (int col = 0; col < num_col(); col++) {
                    set(batch, channel, row, col, rand() % (max_value + 1));
                }
            }
        }
    }
}

template <typename T>
int Matrix<T>::index(int batch, int channel, int row, int col) {
    if (m_layout == NCHW) {
        if (m_isTransposed) {
            return col + row * m_num_col;
        } else {
            return batch * m_num_channel * m_num_row * m_num_col + channel * m_num_row * m_num_col + col * m_num_row +
                   row;
        }
    } else {
        return batch * m_num_channel * m_num_row * m_num_col + col * m_num_row * m_num_channel + row * m_num_channel +
               channel;
    }
}

template <typename T>
void Matrix<T>::set(int batch, int channel, int row, int col, T val) {
    assert(batch < m_num_batch);
    assert(channel < m_num_channel);
    assert(row < m_num_row);
    assert(col < m_num_col);
    m_data[index(batch, channel, row, col)] = val;
}

template <typename T>
T Matrix<T>::get(int batch, int channel, int row, int col) {
    assert(batch < m_num_batch);
    assert(channel < m_num_channel);
    assert(row < m_num_row);
    assert(col < m_num_col);
    return m_data[index(batch, channel, row, col)];
}

template <typename T>
int Matrix<T>::num_row() {
    return m_num_row;
}

template <typename T>
int Matrix<T>::num_col() {
    return m_num_col;
}

template <typename T>
int Matrix<T>::num_batch() {
    return m_num_batch;
}

template <typename T>
int Matrix<T>::num_channel() {
    return m_num_channel;
}

template <typename T>
void Matrix<T>::print(const char *name) {
    for (int batch = 0; batch < num_batch(); batch++) {
        for (int channel = 0; channel < num_channel(); channel++) {
            for (int row = 0; row < num_row(); row++) {
                for (int col = 0; col < num_col(); col++) {
                    cout << name << "[" << batch << "][" << channel << "][" << row << "][" << col
                         << "] = " << get(batch, channel, row, col) << endl;
                }
            }
        }
    }
}

template <typename T>
T *Matrix<T>::data() {
    return m_data;
}

template <typename T>
T *Matrix<T>::d_data() {
    return m_d_data;
}

template <typename T>
bool is_equal(T val1, T val2, float max_error) {
    return val1 == val2;
};

template <>
bool is_equal<float>(float val1, float val2, float max_error) {
    bool is_correct = false;
    if (fabs(val1) < 0.0000001 && fabs(val2) < 0.0000001)
        is_correct = true;
    else {
        if ((val1 == 0.f && val2 != 0.f) || (val1 != 0.f && val2 == 0.f)) {
            is_correct = false;
        } else {
            is_correct = (val1 == 0.f && val2 == 0.f) ||
                         ((fabs(val2 - val1) <= max_error) && (fabs(val2 / val1 - 1) <= max_error));
        }
    }
    if (is_correct == false) {
        printf("                error rate: %.7f\n", fabs(val2 / val1 - 1));
    }
    return is_correct;
};

template <>
bool is_equal<double>(double val1, double val2, float max_error) {
    bool is_correct = false;
    if (fabs(val1) < 0.0000001 && fabs(val2) < 0.0000001)
        is_correct = true;
    else {
        if ((val1 == 0.f && val2 != 0.f) || (val1 != 0.f && val2 == 0.f)) {
            is_correct = false;
        } else {
            is_correct = (val1 == 0.f && val2 == 0.f) ||
                         ((fabs(val2 - val1) <= max_error) && (fabs(val2 / val1 - 1) <= max_error));
        }
    }
    if (is_correct == false) {
        printf("                error rate: %.7f\n", fabs(val2 / val1 - 1));
    }
    return is_correct;
};

template <typename T>
bool Matrix<T>::matrix_compare(const char *name, Matrix<T> &ref_matrix, float max_error) {
    assert(num_batch() == ref_matrix.num_batch());
    assert(num_channel() == ref_matrix.num_channel());
    assert(num_row() == ref_matrix.num_row());
    assert(num_col() == ref_matrix.num_col());

    for (int batch = 0; batch < num_batch(); batch++) {
        for (int channel = 0; channel < num_channel(); channel++) {
            for (int row = 0; row < num_row(); row++) {
                for (int col = 0; col < num_col(); col++) {
                    T my_matrix_data = get(batch, channel, row, col);
                    T ref_matrix_data = ref_matrix.get(batch, channel, row, col);

                    if (is_equal<T>(my_matrix_data, ref_matrix_data, max_error) == false) {
                        cout << name << " mismatch at [" << batch << ", " << channel << ", " << row << ", " << col
                             << "]:" << endl;
                        T var = my_matrix_data;
                        if (var - (int)var == 0)
                            cout << "                my = " << my_matrix_data << " vs ref = " << ref_matrix_data
                                 << endl;
                        else
                            cout << "                my = " << my_matrix_data << " vs ref = " << ref_matrix_data
                                 << endl;

                        return false;
                    }
                }
            }
        }
    }
    return true;
}

template <typename T>
bool matrix_compare(const char *name, Matrix<T> &m1, Matrix<T> &m2, float max_error = 1.e-4) {
    return m1.matrix_compare(name, m2, max_error);
}
/***************************************************************************
end of matrix.cpp
***************************************************************************/
