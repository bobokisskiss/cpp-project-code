#include <Eigen/Dense>
#include <chrono>
#include <iostream>
#include <vector>

using namespace std;
using namespace Eigen;

void testMatrixMultiplication (int dim){
    Eigen::MatrixXd mat1= Eigen::MatrixXd::Random(dim,dim);
    Eigen::MatrixXd mat2= Eigen::MatrixXd::Random(dim,dim);
    //generate two dim*dim random matrix

    auto start= std::chrono::high_resolution_clock::now();// remember the start of the time
    Eigen::MatrixXd result= mat1*mat2;
    auto end= std::chrono::high_resolution_clock::now();// remember the end of the time

    // calculate the time difference
    std::chrono::duration<double> elapsed = end-start;
    //output the time difference and the result
    std::cout << "the matrix multiplication took " << elapsed.count() << " seconds" << std::endl;
  
}

int main(){
    //text low dimention matrix
    testMatrixMultiplication(100);
    //text high dimention matrix
    testMatrixMultiplication(1000);
    return 0;

}