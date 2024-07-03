#include <iostream>
#include <chrono>
#include "dataanalysis.h"
#include "ap.h"
#include "alglibmisc.h"
#include "linalg.h"

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

// 矩阵乘法
void matrix_multiplication_test(int size) {
    real_2d_array a, b, c;
    generate_random_matrix(a, size);
    generate_random_matrix(b, size);
    c.setlength(size, size);

    auto start = high_resolution_clock::now();
    rmatrixgemm(size, size, size, 1.0, a, 0, 0, 0, b, 0, 0, 0, 0.0, c, 0, 0);
    auto end = high_resolution_clock::now();
    duration<double> elapsed = end - start;

    cout << "Matrix multiplication (" << size << "x" << size << ") took " << elapsed.count() << " seconds." << endl;
}

int main() {
    matrix_multiplication_test(100);  // 低维度
    matrix_multiplication_test(1000); // 高维度

    return 0;
}
