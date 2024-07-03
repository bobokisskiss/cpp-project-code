#include <Eigen/Dense>
#include <chrono>
#include <iostream>

void testMatrixAdditionAndSubtraction(int dim) {
    Eigen::MatrixXd mat1 = Eigen::MatrixXd::Random(dim, dim);
    Eigen::MatrixXd mat2 = Eigen::MatrixXd::Random(dim, dim);
    
    auto start = std::chrono::high_resolution_clock::now();
    Eigen::MatrixXd result_add = mat1 + mat2;
    Eigen::MatrixXd result_sub = mat1 - mat2;
    auto end = std::chrono::high_resolution_clock::now();
    
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Matrix Addition and Subtraction (" << dim << "x" << dim << ") took " << elapsed.count() << " seconds.\n";
}

int main() {
    testMatrixAdditionAndSubtraction(100);  // Test low-dimensional data
    testMatrixAdditionAndSubtraction(1000); // Test high-dimensional data
    return 0;
}
