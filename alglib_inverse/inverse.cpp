#include <iostream>
#include <chrono>
#include "src/linalg.h"

using namespace alglib;
using namespace std;
using namespace std::chrono;

// 生成随机矩阵
void generate_random_matrix(real_2d_array &matrix, int size) {
    matrix.setlength(size, size);
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            matrix(i, j) = rand() % 100;
        }
    }
}

// 矩阵求逆
void matrix_inverse_test(int size) {
    real_2d_array a, a_inv;
    generate_random_matrix(a, size);

    auto start = high_resolution_clock::now();
    matinvreport rep;
    rmatrixinverse(a, size, rep);
    auto end = high_resolution_clock::now();
    duration<double> elapsed = end - start;

    cout << "Matrix inverse (" << size << "x" << size << ") took " << elapsed.count() << " seconds." << endl;
}

int main() {
    matrix_inverse_test(100);  // 低维度
    matrix_inverse_test(1000); // 高维度

    return 0;
}
