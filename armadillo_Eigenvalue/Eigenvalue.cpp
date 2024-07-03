#include <armadillo>
#include <chrono>
#include <iostream>

void testMatrixEigenvalues(int dim) {
    arma::mat A = arma::randu<arma::mat>(dim, dim);
    
    auto start = std::chrono::high_resolution_clock::now();
    arma::vec eigval = arma::eig_sym(A);
    auto end = std::chrono::high_resolution_clock::now();
    
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Matrix Eigenvalues (" << dim << "x" << dim << ") computation took " << elapsed.count() << " seconds.\n";
}

int main() {
    testMatrixEigenvalues(100);
    testMatrixEigenvalues(1000);
    return 0;
}
