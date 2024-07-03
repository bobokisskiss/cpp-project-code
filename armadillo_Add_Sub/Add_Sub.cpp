#include <armadillo>
#include <chrono>
#include <iostream>

void testMatrixAdditionAndSubtraction(int dim) {
    arma::mat A = arma::randu<arma::mat>(dim, dim);
    arma::mat B = arma::randu<arma::mat>(dim, dim);

    auto start = std::chrono::high_resolution_clock::now();
    arma::mat C = A + B;
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsedAdd = end - start;
    std::cout << "Matrix Addition (" << dim << "x" << dim << ") took " << elapsedAdd.count() << " seconds.\n";

    start = std::chrono::high_resolution_clock::now();
    arma::mat D = A - B;
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsedSub = end - start;
    std::cout << "Matrix Subtraction (" << dim << "x" << dim << ") took " << elapsedSub.count() << " seconds.\n";
}

int main() {
    testMatrixAdditionAndSubtraction(100);
    testMatrixAdditionAndSubtraction(1000);
    return 0;
}
