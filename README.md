# cuda_matrix_test

C++ matrix library used test correctness of CUDA kernels.
Supports `int`, `float`, `double`, `half` (fp16), `bit`.

## How to build

`make` or `make test`

## Manual

`randomize(double mean, double scale, int sparsity = 0)`
- [mean - scale, mean + scale] 범위의 값들을 갖는 Matrix를 생성한다. 단 Matrix의 템플릿 타입이 half, float, double 중 하나여야 한다.
- 이때 sparsity (percent) 확률로 0으로 채운다.

`randomize(int max_value)`
- [0, max_value] 범위의 값들을 갖는 Matrix를 생성한다. 단 Matrix의 템플릿 타입이 `int`나 `bit`여야 한다.

`index(int batch, int channel, int row, int col)`
- 주어진 parameter에 해당하는 원소의 인덱스를 구한다. 실제 데이터 배열은 1차원이므로 수 하나로 나타낼 수 있다.

`set(int batch, int channel, int row, int col)`, `get(int batch, int channel, int row, int col)`
- Matrix의 원소의 값을 변경하거나 가져온다.

`num_row(), num_col(), num_batch(), num_channel()`
- 각각 row, col, batch, channel의 수를 반환한다.

`print(const char *name)`
- Matrix를 출력한다.

`data(), d_data()`
- Matrix 데이터가 저장된 host/device pointer를 반환한다.

`is_equal(<type> val1, <type> val2, [optional] float max_error)`
- 두 값이 같은지 확인한다. floating-point type의 경우에는 max_error 값보다 차이가 작으면 같은 것으로 간주한다.

`matrix_compare(const char *name, Matrix<T> &ref_matrix, float max_error)`
- 두 Matrix가 같은지 확인한다.
- m1.matrix_compare(name, m2, max_error); 로 사용할 수도 있고
- matrix_compare(name, m1, m2, max_error); 로 사용할 수도 있다.

## contributions

[matrix.cpp](https://github.com/NVIDIA/nv-wavenet/blob/master/matrix.cpp) was used as the base of this repository.
