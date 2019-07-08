#include <stdio.h>
#include "matrix.hpp"

template <typename T>
class Test {
   public:
    void test_randomize();
    void test_matrix_compare();
};

template <typename T>
void Test<T>::test_randomize() {
    Matrix<T> m1(NCHW, 10, 11, 12, 13);
    m1.randomize(150, 20, 0);
    for (int batch = 0; batch < m1.num_batch(); batch++) {
        for (int channel = 0; channel < m1.num_channel(); channel++) {
            for (int row = 0; row < m1.num_row(); row++) {
                for (int col = 0; col < m1.num_col(); col++) {
                    T val = m1.get(batch, channel, row, col);
                    assert(130 <= val && val <= 170);
                }
            }
        }
    }
}

template <>
void Test<int>::test_randomize() {
    Matrix<int> m1(NCHW, 10, 11, 12, 13);
    m1.randomize(100);
    for (int batch = 0; batch < m1.num_batch(); batch++) {
        for (int channel = 0; channel < m1.num_channel(); channel++) {
            for (int row = 0; row < m1.num_row(); row++) {
                for (int col = 0; col < m1.num_col(); col++) {
                    int val = m1.get(batch, channel, row, col);
                    assert(0 <= val && val <= 100);
                }
            }
        }
    }
}

template <typename T>
void Test<T>::test_matrix_compare() {
    try {
        Matrix<T> m1(NCHW, 11, 12, 13, 14);
        Matrix<T> m2(NCHW, 11, 12, 13, 14);
        int k = 0;
        for (int batch = 0; batch < m1.num_batch(); batch++) {
            for (int channel = 0; channel < m1.num_channel(); channel++) {
                for (int row = 0; row < m1.num_row(); row++) {
                    for (int col = 0; col < m1.num_col(); col++) {
                        m1.set(batch, channel, row, col, (float)1.2 * (k));
                        m2.set(batch, channel, row, col, (float)1.2 * (k++));
                    }
                }
            }
        }

        if (!m1.matrix_compare("Compare", m2)) throw 1;
        if (!m2.matrix_compare("Compare", m1)) throw 2;

        Matrix<T> m3(NCHW, 11, 12, 13, 14);
        Matrix<T> m4(NCHW, 11, 12, 13, 14);
        k = 0;
        for (int batch = 0; batch < m3.num_batch(); batch++) {
            for (int channel = 0; channel < m3.num_channel(); channel++) {
                for (int row = 0; row < m3.num_row(); row++) {
                    for (int col = 0; col < m3.num_col(); col++) {
                        m3.set(batch, channel, row, col, (float)1.2 * (k));
                        m4.set(batch, channel, row, col, (float)1.2 * (k != 20 ? k++ : ++k));
                    }
                }
            }
        }
        if (m3.matrix_compare("Compare", m4)) throw 3;
        if (m4.matrix_compare("Compare", m3)) throw 4;

        printf("Test Passed!\n");
    } catch (int exp) {
        printf("Test failed: %d\n", exp);
    }
}

template <>
void Test<half>::test_matrix_compare() {
    try {
        Matrix<half> m1(NCHW, 11, 12, 13, 14);
        Matrix<half> m2(NCHW, 11, 12, 13, 14);
        int k = 0;
        for (int batch = 0; batch < m1.num_batch(); batch++) {
            for (int channel = 0; channel < m1.num_channel(); channel++) {
                for (int row = 0; row < m1.num_row(); row++) {
                    for (int col = 0; col < m1.num_col(); col++) {
                        m1.set(batch, channel, row, col, half(1.2 * (k)));
                        m2.set(batch, channel, row, col, half(1.2 * (k++)));
                    }
                }
            }
        }

        if (!m1.matrix_compare("Compare", m2)) throw 1;
        if (!m2.matrix_compare("Compare", m1)) throw 2;

        Matrix<half> m3(NCHW, 11, 12, 13, 14);
        Matrix<half> m4(NCHW, 11, 12, 13, 14);
        k = 0;
        for (int batch = 0; batch < m3.num_batch(); batch++) {
            for (int channel = 0; channel < m3.num_channel(); channel++) {
                for (int row = 0; row < m3.num_row(); row++) {
                    for (int col = 0; col < m3.num_col(); col++) {
                        m3.set(batch, channel, row, col, half(1.2 * (k)));
                        m4.set(batch, channel, row, col, half(1.2 * (k != 20 ? k++ : ++k)));
                    }
                }
            }
        }
        if (m3.matrix_compare("Compare", m4)) throw 3;
        if (m4.matrix_compare("Compare", m3)) throw 4;

        printf("Test Passed!\n");
    } catch (int exp) {
        printf("Test failed: %d\n", exp);
    }
}

int main() {
    printf("Test cases for class Matrix\n");

    Test<int> int_test;
    Test<float> float_test;
    Test<double> double_test;
    Test<half> half_test;

    puts("Randomize test for int");
    int_test.test_randomize();
    puts("Randomize test for float");
    float_test.test_randomize();
    puts("Randomize test for double");
    double_test.test_randomize();
    puts("Randomize test for half");
    half_test.test_randomize();

    puts("Compare test for int");
    int_test.test_matrix_compare();
    puts("Compare test for float");
    float_test.test_matrix_compare();
    puts("Compare test for double");
    double_test.test_matrix_compare();
    puts("Compare test for half");
    half_test.test_matrix_compare();
}