
#include<Eigen/Dense>
#include<chrono>
#include<iostream>

void testMatrixInverse(int dim){
    Eigen::MatrixXd mat= Eigen::MatrixXd::Random(dim,dim);

    auto start = std::chrono::high_resolution_clock::now();
    Eigen::MatrixXd inverse = mat. inverse();
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Matrix Inverse (" << dim << "x" << dim << ") took " << elapsed.count() << " seconds.\n";

}

int main(){
    testMatrixInverse(100);
    testMatrixInverse(1000);
    
}