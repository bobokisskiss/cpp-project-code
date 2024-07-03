#include <Eigen/Dense>
#include <chrono>
#include <iostream>

void testMatrixEigenvalues(int dim) {
    std::cout << "Testing matrix of dimension: " << dim << "x" << dim << std::endl;
    Eigen::MatrixXd mat = Eigen::MatrixXd::Random(dim, dim);
    
    auto start = std::chrono::high_resolution_clock::now();
    Eigen::EigenSolver<Eigen::MatrixXd> es(mat);
    auto end = std::chrono::high_resolution_clock::now();
    
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Matrix Eigenvalues (" << dim << "x" << dim << ") computation took " << elapsed.count() << " seconds.\n" << std::flush;
}

int main() {
    std::cout << "Starting tests..." << std::endl;
    testMatrixEigenvalues(100);  // Test low-dimensional data
    testMatrixEigenvalues(500); // Test high-dimensional data
    std::cout << "Tests completed." << std::endl;
    return 0;
}
