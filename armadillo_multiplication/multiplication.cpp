#include <armadillo>
#include <chrono>
#include <iostream>

void testMatrixMultiplication(int dim) {
    arma::mat A= arma::randu<arma::mat>(dim,dim);
    arma::mat B= arma::randu<arma::mat>(dim,dim);

    auto start = std::chrono::high_resolution_clock::now();
    arma::mat C = A*B;
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Matrix Multiplication (" << dim << "x" << dim << ") took " << elapsed.count() << " seconds.\n";
}

int main(){
    testMatrixMultiplication(100);
    testMatrixMultiplication(1000);
    return 0;
}