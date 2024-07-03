#include <iostream>
#include <chrono>
#include "alglibmisc.h"
#include "linalg.h"

using namespace alglib;
using namespace std;
using namespace chrono;

// Function to generate random matrix of given size
real_2d_array generateRandomMatrix(int rows, int cols) {
    real_2d_array matrix;
    matrix.setlength(rows, cols);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            matrix[i][j] = rand() % 100; // Random value between 0 and 99
        }
    }
    return matrix;
}

// Function to measure execution time of matrix operation
void measureTime(const char* operation, const function<void()> &operationFunc) {
    auto start = high_resolution_clock::now();
    operationFunc();
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(stop - start);
    cout << operation << " Time: " << duration.count() << " milliseconds" << endl;
}

int main() {
    const int dim100 = 100; // Dimension of low-dimensional matrix
    const int dim1000 = 1000; // Dimension of high-dimensional matrix

    // Generate random matrices
    real_2d_array A = generateRandomMatrix(dim100, dim100);
    real_2d_array B = generateRandomMatrix(dim100, dim100);
    real_2d_array C = generateRandomMatrix(dim1000, dim1000);
    real_2d_array D = generateRandomMatrix(dim1000, dim1000);

    // Measure time for matrix addition and subtraction
    measureTime("100x100 Matrix Addition", [&]() {
        real_2d_array result;
        rmatrixadd(A, B, result);
    });

    measureTime("100x100 Matrix Subtraction", [&]() {
        real_2d_array result;
        rmatrixsub(A, B, result);
    });

    measureTime("1000x1000 Matrix Addition", [&]() {
        real_2d_array result;
        rmatrixadd(C, D, result);
    });

    measureTime("1000x1000 Matrix Subtraction", [&]() {
        real_2d_array result;
        rmatrixsub(C, D, result);
    });

    return 0;
}
