#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <Eigen>
#include <armadillo>
#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xview.hpp>
#include <xtensor/xrandom.hpp>
#include <xtensor-blas/xlinalg.hpp>

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
        if (n > 1000) break; // Stop testing for Eigen when n > 1000
        Eigen::MatrixXd eigen_matrix = read_matrix<Eigen::MatrixXd>("data/eigen_matrix_" + std::to_string(n) + ".txt", n, n);
        
        double total_time = 0.0;
        for (int i = 0; i < repetitions; ++i) {
            auto start = std::chrono::high_resolution_clock::now();
            Eigen::MatrixXd result = eigen_matrix * eigen_matrix.transpose();
            auto end = std::chrono::high_resolution_clock::now();

            std::chrono::duration<double> elapsed = end - start;
            total_time += elapsed.count();
        }
        
        std::cout << "Eigen, n=" << n << ", average time per A*A^T: " << (total_time / repetitions) << " seconds" << std::endl;
    }
}

void test_armadillo() {
    std::cout << "Testing Armadillo..." << std::endl;
    for (int n : dimensions) {
        arma::mat arma_matrix;
        arma_matrix.load("data/arma_matrix_" + std::to_string(n) + ".txt");

        double total_time = 0.0;
        for (int i = 0; i < repetitions; ++i) {
            auto start = std::chrono::high_resolution_clock::now();
            arma::mat result = arma_matrix * arma_matrix.t();
            auto end = std::chrono::high_resolution_clock::now();

            std::chrono::duration<double> elapsed = end - start;
            total_time += elapsed.count();
        }

        std::cout << "Armadillo, n=" << n << ", average time per A*A^T: " << (total_time / repetitions) << " seconds" << std::endl;
    }
}

void test_xtensor() {
    std::cout << "Testing XTensor..." << std::endl;
    for (int n : dimensions) {
        xt::xarray<double> xtensor_matrix = xt::zeros<double>({n, n});
        std::ifstream file("data/xtensor_matrix_" + std::to_string(n) + ".txt");
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                file >> xtensor_matrix(i, j);
            }
        }

        double total_time = 0.0;
        for (int i = 0; i < repetitions; ++i) {
            auto start = std::chrono::high_resolution_clock::now();
            auto result = xt::linalg::dot(xtensor_matrix, xt::transpose(xtensor_matrix));
            auto end = std::chrono::high_resolution_clock::now();

            std::chrono::duration<double> elapsed = end - start;
            total_time += elapsed.count();
        }

        std::cout << "XTensor, n=" << n << ", average time per A*A^T: " << (total_time / repetitions) << " seconds" << std::endl;
    }
}

int main() {
    test_eigen();
    test_armadillo();
    test_xtensor();
    return 0;
}
