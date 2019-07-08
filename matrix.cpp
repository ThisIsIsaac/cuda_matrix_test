#include <assert.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "half.hpp"
using half_float::half;
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
    int m_rows;
    int m_cols;

   public:
    Matrix(Layout layout, int num_batch, int num_channel, int rows, int cols, bool isTransposed = false);

    ~Matrix();

    void randomize_float(float mean, float scale, int sparsity = 0);

    void randomize_int(int max_value);

    void randomize_double(double mean, double scale, int sparsity = 0);

    void randomize_half(half mean, half scale, int sparsity = 0);

    int index(int batch, int channel, int row, int col);

    void set(int batch, int channel, int row, int col, T val);

    T get(int batch, int channel, int row, int col);

    int rows();

    int cols();

    int num_batch();

    int num_channel();

    void print(const char *name);

    bool matrix_compare(const char *name, Matrix<T> &ref_matirx, float max_error = 1.e-4);

    void d_cudaMemcpy();

    T *data();
    T *d_data();

    // bool is_same(T val1, T val2, float max_error);
};

#endif

template <typename T>
Matrix<T>::Matrix(Layout layout, int num_batch, int num_channel, int rows, int cols, bool isTransposed)
    : m_layout(layout),
      m_num_batch(num_batch),
      m_num_channel(num_channel),
      m_rows(rows),
      m_cols(cols),
      m_isTransposed(isTransposed) {
    m_data = (T *)malloc(num_batch * num_channel * rows * cols * sizeof(T));
    CHECK(cudaMalloc((int**) &m_d_data, num_batch * num_channel * rows * cols * sizeof(T))));
}

template <typename T>
Matrix<T>::~Matrix() {
    CHECK(cudaFree(m_d_data));
    free(m_data);
}

template <typename T>
void Matrix<T>::d_cudaMemcpy() {
    size_t nBytes = num_batch() * num_channel() * rows() * cols() * sizeof(T);
    CHECK(cudaMemset(m_d_data, 0, nBytes));
    CHECK(cudaMemcpy(m_data, m_d_data, nBytes, cudaMemcpyHostToDevice));
}

template <typename T>
void Matrix<T>::randomize_float(float mean, float scale, int sparsity) {
    for (int batch = 0; batch < num_batch(); batch++) {
        for (int channel = 0; channel < num_channel(); channel++) {
            for (int row = 0; row < rows(); row++) {
                for (int col = 0; col < cols(); col++) {
                    if ((rand() % 100) < sparsity) {
                        // printf("this is %f\n", 0.f);
                        set(batch, channel, row, col, 0.f);
                    } else {
                        // Generate a random number from 0 to 1.0
                        float r = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);

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
}

template <typename T>
void Matrix<T>::randomize_double(double mean, double scale, int sparsity) {
    for (int batch = 0; batch < num_batch(); batch++) {
        for (int channel = 0; channel < num_channel(); channel++) {
            for (int row = 0; row < rows(); row++) {
                for (int col = 0; col < cols(); col++) {
                    if ((rand() % 100) < sparsity) {
                        set(batch, channel, row, col, 0.0);
                    } else {
                        double r = static_cast<double>(rand()) / static_cast<double>(RAND_MAX);

                        r -= 0.5;

                        r = r * scale + mean;

                        set(batch, channel, row, col, r);
                    }
                }
            }
        }
    }
}

template <typename T>
void Matrix<T>::randomize_half(half mean, half scale, int sparsity) {
    for (int batch = 0; batch < num_batch(); batch++) {
        for (int channel = 0; channel < num_channel(); channel++) {
            for (int row = 0; row < rows(); row++) {
                for (int col = 0; col < cols(); col++) {
                    if ((rand() % 100) < sparsity) {
                        half a(0.0);
                        set(batch, channel, row, col, a);
                    } else {
                        float r = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);

                        r -= 0.5;

                        r = r * scale + mean;

                        half val(r);

                        set(batch, channel, row, col, val);
                    }
                }
            }
        }
    }
}

template <typename T>
void Matrix<T>::randomize_int(int max_value) {
    for (int batch = 0; batch < num_batch(); batch++) {
        for (int channel = 0; channel < num_channel(); channel++) {
            for (int row = 0; row < rows(); row++) {
                for (int col = 0; col < cols(); col++) {
                    set(batch, channel, row, col, rand() % max_value);
                }
            }
        }
    }
}

template <typename T>
int Matrix<T>::index(int batch, int channel, int row, int col) {
    if (m_layout == NCHW) {
        if (m_isTransposed) {
            return col + row * m_cols;
        } else {
            return batch * m_num_channel * m_rows * m_cols + channel * m_rows * m_cols + col * m_rows + row;
        }
    } else {
        return batch * m_num_channel * m_rows * m_cols + col * m_rows * m_num_channel + row * m_num_channel + channel;
    }
}

template <typename T>
void Matrix<T>::set(int batch, int channel, int row, int col, T val) {
    assert(batch < m_num_batch);
    assert(channel < m_num_channel);
    assert(row < m_rows);
    assert(col < m_cols);
    // printf("T Val %d\n", val);
    m_data[index(batch, channel, row, col)] = val;
}

template <typename T>
T Matrix<T>::get(int batch, int channel, int row, int col) {
    assert(batch < m_num_batch);
    assert(channel < m_num_channel);
    assert(row < m_rows);
    assert(col < m_cols);
    return m_data[index(batch, channel, row, col)];
}

template <typename T>
int Matrix<T>::rows() {
    return m_rows;
}

template <typename T>
int Matrix<T>::cols() {
    return m_cols;
}

template <typename T>
int Matrix<T>::num_batch() {
    return m_num_batch;
}

template <typename T>
int Matrix<T>::num_channel() {
    return m_num_channel;
}

template <>
void Matrix<float>::print(const char *name) {
    for (int batch = 0; batch < num_batch(); batch++) {
        for (int channel = 0; channel < num_channel(); channel++) {
            for (int row = 0; row < rows(); row++) {
                for (int col = 0; col < cols(); col++) {
                    printf("%s[%d][%d][%d][%d] = %f\n", name, batch, channel, row, col, get(batch, channel, row, col));
                }
            }
        }
    }
}

template <>
void Matrix<int>::print(const char *name) {
    for (int batch = 0; batch < num_batch(); batch++) {
        for (int channel = 0; channel < num_channel(); channel++) {
            for (int row = 0; row < rows(); row++) {
                for (int col = 0; col < cols(); col++) {
                    printf("%s[%d][%d][%d][%d] = %d\n", name, batch, channel, row, col, get(batch, channel, row, col));
                }
            }
        }
    }
}

template <>
void Matrix<double>::print(const char *name) {
    for (int batch = 0; batch < num_batch(); batch++) {
        for (int channel = 0; channel < num_channel(); channel++) {
            for (int row = 0; row < rows(); row++) {
                for (int col = 0; col < cols(); col++) {
                    printf("%s[%d][%d][%d][%d] = %f\n", name, batch, channel, row, col, get(batch, channel, row, col));
                }
            }
        }
    }
}

template <>
void Matrix<half>::print(const char *name) {
    for (int batch = 0; batch < num_batch(); batch++) {
        for (int channel = 0; channel < num_channel(); channel++) {
            for (int row = 0; row < rows(); row++) {
                for (int col = 0; col < cols(); col++) {
                    std::cout << name << "[" << batch << "][" << channel << "][" << row << "][" << col
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
            // is_correct = (val1 <= 0.f && val2 == 0.f) || (fabs(val2 - val1)
            // <= 0.000001f) || (fabs(val2/val1 - 1) <= max_error);
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

template <>
bool is_equal<int>(int val1, int val2, float max_error) {
    bool is_correct = false;

    if (val1 == val2) is_correct = true;

    return is_correct;
}

template <>
bool is_equal<half>(half val1, half val2, float max_error) {
    bool is_correct = false;

    if (half_float::detail::functions::isequal(val1, val2)) is_correct = true;

    return is_correct;
}

template <>
bool Matrix<float>::matrix_compare(const char *name, Matrix<float> &ref_matrix, float max_error) {
    assert(num_batch() == ref_matrix.num_batch());
    assert(num_channel() == ref_matrix.num_channel());
    assert(rows() == ref_matrix.rows());
    assert(cols() == ref_matrix.cols());

    for (int batch = 0; batch < num_batch(); batch++) {
        for (int channel = 0; channel < num_channel(); channel++) {
            for (int row = 0; row < rows(); row++) {
                for (int col = 0; col < cols(); col++) {
                    float my_matrix_data = get(batch, channel, row, col);
                    float ref_matrix_data = ref_matrix.get(batch, channel, row, col);

                    if (is_equal<float>(my_matrix_data, ref_matrix_data, max_error) == false) {
                        printf("            %s mismatch at [%d, %d, %d,%d]:\n", name, batch, channel, row, col);
                        float var = my_matrix_data;
                        if (var - (int)var == 0)
                            printf("                my = %.7f vs ref = %.7f\n", my_matrix_data, ref_matrix_data);
                        else
                            printf("                my = %.7f vs ref = %.7f\n", my_matrix_data, ref_matrix_data);

                        return false;
                    }
                }
            }
        }
    }
    return true;
}

template <>
bool Matrix<double>::matrix_compare(const char *name, Matrix<double> &ref_matrix, float max_error) {
    assert(num_batch() == ref_matrix.num_batch());
    assert(num_channel() == ref_matrix.num_channel());
    assert(rows() == ref_matrix.rows());
    assert(cols() == ref_matrix.cols());

    for (int batch = 0; batch < num_batch(); batch++) {
        for (int channel = 0; channel < num_channel(); channel++) {
            for (int row = 0; row < rows(); row++) {
                for (int col = 0; col < cols(); col++) {
                    double my_matrix_data = get(batch, channel, row, col);
                    double ref_matrix_data = ref_matrix.get(batch, channel, row, col);

                    if (is_equal<double>(my_matrix_data, ref_matrix_data, max_error) == false) {
                        printf("            %s mismatch at [%d, %d, %d,%d]:\n", name, batch, channel, row, col);
                        float var = my_matrix_data;
                        if (var - (int)var == 0)
                            printf("                my = %.7f vs ref = %.7f\n", my_matrix_data, ref_matrix_data);
                        else
                            printf("                my = %.7f vs ref = %.7f\n", my_matrix_data, ref_matrix_data);

                        return false;
                    }
                }
            }
        }
    }
    return true;
}

template <>
bool Matrix<int>::matrix_compare(const char *name, Matrix<int> &ref_matrix, float max_error) {
    assert(num_batch() == ref_matrix.num_batch());
    assert(num_channel() == ref_matrix.num_channel());
    assert(rows() == ref_matrix.rows());
    assert(cols() == ref_matrix.cols());

    for (int batch = 0; batch < num_batch(); batch++) {
        for (int channel = 0; channel < num_channel(); channel++) {
            for (int row = 0; row < rows(); row++) {
                for (int col = 0; col < cols(); col++) {
                    int my_matrix_data = get(batch, channel, row, col);
                    int ref_matrix_data = ref_matrix.get(batch, channel, row, col);

                    if (is_equal<int>(my_matrix_data, ref_matrix_data, max_error) == false) {
                        printf("            %s mismatch at [%d, %d, %d,%d]:\n", name, batch, channel, row, col);
                        printf("                my = %.7d vs ref = %.7d\n", my_matrix_data, ref_matrix_data);

                        return false;
                    }
                }
            }
        }
    }
    return true;
}

template <>
bool Matrix<half>::matrix_compare(const char *name, Matrix<half> &ref_matrix, float max_error) {
    assert(num_batch() == ref_matrix.num_batch());
    assert(num_channel() == ref_matrix.num_channel());
    assert(rows() == ref_matrix.rows());
    assert(cols() == ref_matrix.cols());

    for (int batch = 0; batch < num_batch(); batch++) {
        for (int channel = 0; channel < num_channel(); channel++) {
            for (int row = 0; row < rows(); row++) {
                for (int col = 0; col < cols(); col++) {
                    half my_matrix_data = get(batch, channel, row, col);
                    half ref_matrix_data = ref_matrix.get(batch, channel, row, col);

                    if (is_equal<half>(my_matrix_data, ref_matrix_data, max_error) == false) {
                        std::cout << name << " mismatch at [" << batch << ", " << channel << ", " << row << "," << col
                                  << "]:" << endl;
                        half var = my_matrix_data;
                        if (var - (int)var == 0)
                            std::cout << "                my = " << my_matrix_data << " vs ref = " << ref_matrix_data
                                      << endl;
                        else
                            std::cout << "                my = " << my_matrix_data << " vs ref = " << ref_matrix_data
                                      << endl;

                        return false;
                    }
                }
            }
        }
    }
    return true;
}
