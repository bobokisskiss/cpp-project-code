#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <Eigen>
#include <armadillo>
#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xrandom.hpp>

std::vector<int> dimensions = {5, 10, 50, 100, 500, 1000, 5000, 10000};
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
    std::cout << "Testing Eigen..." << std::endl;
    for (int n : dimensions) {
        Eigen::MatrixXd eigen_matrix = read_matrix<Eigen::MatrixXd>("data/eigen_matrix_" + std::to_string(n) + ".txt", n, n);
        
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < repetitions; ++i) {
            Eigen::MatrixXd transposed = eigen_matrix.transpose();
        }
        auto end = std::chrono::high_resolution_clock::now();
        
        std::chrono::duration<double> elapsed = end - start;
        std::cout << "Eigen, n=" << n << ", average time per transpose: " << (elapsed.count() / repetitions) << " seconds" << std::endl;
    }
}

void test_armadillo() {
    std::cout << "Testing Armadillo..." << std::endl;
    for (int n : dimensions) {
        arma::mat arma_matrix;
        arma_matrix.load("data/arma_matrix_" + std::to_string(n) + ".txt");

        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < repetitions; ++i) {
            arma::mean(arma_matrix); // Simply call arma::mean to test time
        }
        auto end = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double> elapsed = end - start;
        std::cout << "Armadillo, n=" << n << ", average time per mean: " << (elapsed.count() / repetitions) << " seconds" << std::endl;
    }
}

void test_xtensor() {
    std::cout << "Testing XTensor..." << std::endl;
    for (int n : dimensions) {
        xt::xarray<double> xtensor_matrix = xt::zeros<double>({n, n});
        std::ifstream file("xtensor_data/xtensor_matrix_" + std::to_string(n) + ".txt");
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                file >> xtensor_matrix(i, j);
            }
        }

        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < repetitions; ++i) {
            auto transposed = xt::transpose(xtensor_matrix);
        }
        auto end = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double> elapsed = end - start;
        std::cout << "XTensor, n=" << n << ", average time per transpose: " << (elapsed.count() / repetitions) << " seconds" << std::endl;
    }
}

int main() {
    test_eigen();
    test_armadillo();
    test_xtensor();
    return 0;
}
