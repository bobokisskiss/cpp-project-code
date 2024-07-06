#include <iostream>
#include <fstream>
#include <chrono>
#include <Eigen/Dense>

std::vector<int> dimensions = {5, 10, 50, 100, 500, 1000, 5000};
const int repetitions = 100;

template<typename MatrixType>
MatrixType read_matrix(const std::string& filename, int rows, int cols) {
    std::ifstream file(filename);
    MatrixType matrix(rows, cols);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            file >> matrix(i, j);
        }
    }
    return matrix;
}

void test_eigen() {
    std::cout << "Testing Eigen LU decomposition..." << std::endl;
    for (int n : dimensions) {
        Eigen::MatrixXd eigen_matrix = read_matrix<Eigen::MatrixXd>("data/eigen_matrix_" + std::to_string(n) + ".txt", n, n);
        
        double total_time = 0.0;
        for (int i = 0; i < repetitions; ++i) {
            auto start = std::chrono::high_resolution_clock::now();
            Eigen::FullPivLU<Eigen::MatrixXd> lu(eigen_matrix);
            auto end = std::chrono::high_resolution_clock::now();

            std::chrono::duration<double> elapsed = end - start;
            total_time += elapsed.count();
        }
        
        std::cout << "Eigen, n=" << n << ", average time per LU decomposition: " << (total_time / repetitions) << " seconds" << std::endl;
    }
}

int main() {
    test_eigen();
    return 0;
}
