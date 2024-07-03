#include <iostream>
#include <vector>
#include <chrono>
#include <cblas.h>
#include <lapacke.h>

void testMatrixInverse(int dim) {
    std::vector<double> A(dim * dim);

    // Fill A with random values
    for(int i = 0; i < dim * dim; ++i) {
        A[i] = static_cast<double>(rand()) / RAND_MAX;
    }

    std::vector<int> ipiv(dim);
    int info;

    auto start = std::chrono::high_resolution_clock::now();
    info = LAPACKE_dgetrf(LAPACK_ROW_MAJOR, dim, dim, A.data(), dim, ipiv.data());
    if(info == 0) {
        info = LAPACKE_dgetri(LAPACK_ROW_MAJOR, dim, A.data(), dim, ipiv.data());
    }
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Matrix Inversion (" << dim << "x" << dim << ") took " << elapsed.count() << " seconds.\n";
}

int main() {
    testMatrixInverse(100);  // Test low-dimensional data
    testMatrixInverse(1000); // Test high-dimensional data
    return 0;
}
