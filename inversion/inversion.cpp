#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <Eigen>
#include <armadillo>
#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xview.hpp>
#include <xtensor-blas/xlinalg.hpp>

std::vector<int> dimensions = {5, 10, 50, 100, 500, 1000, 5000, 10000};
std::vector<int> eigen_dimensions = {5, 10, 50, 100, 500};
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

xt::xarray<double> read_xtensor_matrix(const std::string& filename, int rows, int cols) {
    std::ifstream file(filename);
    xt::xarray<double> matrix = xt::zeros<double>({rows, cols});
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            file >> matrix(i, j);
        }
    }
    return matrix;
}

void test_eigen() {
    std::cout << "Testing Eigen..." << std::endl;
    for (int n : eigen_dimensions) {
        Eigen::MatrixXd eigen_matrix = read_matrix<Eigen::MatrixXd>("data/eigen_matrix_" + std::to_string(n) + ".txt", n, n);
        
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < repetitions; ++i) {
            Eigen::MatrixXd inverted = eigen_matrix.inverse();
        }
        auto end = std::chrono::high_resolution_clock::now();
        
        std::chrono::duration<double> elapsed = end - start;
        std::cout << "Eigen, n=" << n << ", average time per inversion: " << (elapsed.count() / repetitions) << " seconds" << std::endl;
    }
}

void test_armadillo() {
    std::cout << "Testing Armadillo..." << std::endl;
    for (int n : dimensions) {
        arma::mat arma_matrix;
        arma_matrix.load("data/arma_matrix_" + std::to_string(n) + ".txt");

        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < repetitions; ++i) {
            arma::mat inverted = arma::inv(arma_matrix);
        }
        auto end = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double> elapsed = end - start;
        std::cout << "Armadillo, n=" << n << ", average time per inversion: " << (elapsed.count() / repetitions) << " seconds" << std::endl;
    }
}

void test_xtensor() {
    std::cout << "Testing XTensor..." << std::endl;
    for (int n : dimensions) {
        xt::xarray<double> xtensor_matrix = read_xtensor_matrix("data/arma_matrix_" + std::to_string(n) + ".txt", n, n);

        double total_time = 0;
        int successful_repetitions = 0;

        for (int i = 0; i < repetitions; ++i) {
            try {
                auto start = std::chrono::high_resolution_clock::now();
                auto inverted = xt::linalg::inv(xtensor_matrix);
                auto end = std::chrono::high_resolution_clock::now();

                std::chrono::duration<double> elapsed = end - start;
                total_time += elapsed.count();
                successful_repetitions++;
            } catch (const std::runtime_error& e) {
                // Skip this repetition due to singular matrix
                continue;
            }
        }

        if (successful_repetitions > 0) {
            std::cout << "XTensor, n=" << n << ", average time per inversion: " << (total_time / successful_repetitions) << " seconds" << std::endl;
        } else {
            std::cout << "XTensor, n=" << n << ", all matrices were singular and not invertible." << std::endl;
        }
    }
}

int main() {
    test_eigen();
    test_armadillo();
    test_xtensor();
    return 0;
}
