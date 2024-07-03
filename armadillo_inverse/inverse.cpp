#include <armadillo>
#include <chrono>
#include <iostream>

void testMatrixInverse(int dim){
    arma::mat A = arma::randu<arma::mat>(dim, dim);
    
    auto start = std::chrono::high_resolution_clock::now();
    arma:: mat A_inv = arma::inv(A);
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Matrix Inversion (" << dim << "x" << dim << ") took " << elapsed.count() << " seconds.\n";
    
}

int main(){
    testMatrixInverse(100);
    testMatrixInverse(1000);
    
    return 0;
}