#include <iostream>
#include <fstream>
#include <vector>
#include <armadillo>

std::vector<int> dimensions = {5, 10, 50, 100, 500, 1000, 5000, 10000};

void generate_matrices() {
    for (int n : dimensions) {
        arma::mat matrix = arma::randu<arma::mat>(n, n);  // 生成n x n的随机矩阵
        std::string filename = "data/arma_matrix_" + std::to_string(n) + ".txt";
        matrix.save(filename, arma::raw_ascii);
        std::cout << "Matrix saved to: " << filename << std::endl;
    }
}

int main() {
    generate_matrices();
    std::cout << "Matrices generated successfully." << std::endl;
    return 0;
}
