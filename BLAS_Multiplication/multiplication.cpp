#include <iostream>
#include <vector>
#include <chrono>
extern "C" {
    #include <cblas.h>
}

void testMatrixMultiplication(int dim) {
    std::vector<double> A(dim * dim);
    std::vector<double> B(dim * dim);
    std::vector<double> C(dim * dim, 0.0);

    // Fill A and B with random values
    for(int i = 0; i < dim * dim; ++i) {
        A[i] = static_cast<double>(rand()) / RAND_MAX;
        B[i] = static_cast<double>(rand()) / RAND_MAX;
    }

    auto start = std::chrono::high_resolution_clock::now();
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, dim, dim, dim, 1.0, A.data(), dim, B.data(), dim, 0.0, C.data(), dim);
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Matrix Multiplication (" << dim << "x" << dim << ") took " << elapsed.count() << " seconds.\n";
}

int main() {
    testMatrixMultiplication(100);  // Test low-dimensional data
    testMatrixMultiplication(1000); // Test high-dimensional data
    return 0;
}

