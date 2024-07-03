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

// Function to read matrix from file using Eigen
Eigen::MatrixXd read_matrix_eigen(const std::string& filename, int rows, int cols) {
    std::ifstream file(filename);
    Eigen::MatrixXd matrix(rows, cols);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            file >> matrix(i, j);
        }
    }
    return matrix;
}

// Function to read matrix from file using Armadillo
arma::mat read_matrix_armadillo(const std::string& filename) {
    arma::mat matrix;
    matrix.load(filename);
    return matrix;
}

// Function to read matrix from file using XTensor
xt::xarray<double> read_matrix_xtensor(const std::string& filename, int rows, int cols) {
    std::ifstream file(filename);
    xt::xarray<double> matrix = xt::zeros<double>({rows, cols});
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            file >> matrix(i, j);
        }
    }
    return matrix;
}

void test_eigen_min_max() {
    std::cout << "Testing Eigen min/max..." << std::endl;
    for (int n : dimensions) {
        if (n > 1000) break; // Limit Eigen tests to n <= 1000
        Eigen::MatrixXd eigen_matrix = read_matrix_eigen("data/eigen_matrix_" + std::to_string(n) + ".txt", n, n);
        
        double total_min_time = 0.0;
        double total_max_time = 0.0;
        
        for (int i = 0; i < repetitions; ++i) {
            auto start_min = std::chrono::high_resolution_clock::now();
            double min_val = eigen_matrix.minCoeff();
            auto end_min = std::chrono::high_resolution_clock::now();

            auto start_max = std::chrono::high_resolution_clock::now();
            double max_val = eigen_matrix.maxCoeff();
            auto end_max = std::chrono::high_resolution_clock::now();

            std::chrono::duration<double> elapsed_min = end_min - start_min;
            std::chrono::duration<double> elapsed_max = end_max - start_max;

            total_min_time += elapsed_min.count();
            total_max_time += elapsed_max.count();
        }
        
        std::cout << "Eigen, n=" << n << ", average time per min: " << (total_min_time / repetitions) << " seconds" << std::endl;
        std::cout << "Eigen, n=" << n << ", average time per max: " << (total_max_time / repetitions) << " seconds" << std::endl;
    }
}

void test_armadillo_min_max() {
    std::cout << "Testing Armadillo min/max..." << std::endl;
    for (int n : dimensions) {
        arma::mat arma_matrix = read_matrix_armadillo("data/arma_matrix_" + std::to_string(n) + ".txt");

        double total_min_time = 0.0;
        double total_max_time = 0.0;

        for (int i = 0; i < repetitions; ++i) {
            auto start_min = std::chrono::high_resolution_clock::now();
            double min_val = arma::min(arma_matrix);
            auto end_min = std::chrono::high_resolution_clock::now();

            auto start_max = std::chrono::high_resolution_clock::now();
            double max_val = arma::max(arma_matrix);
            auto end_max = std::chrono::high_resolution_clock::now();

            std::chrono::duration<double> elapsed_min = end_min - start_min;
            std::chrono::duration<double> elapsed_max = end_max - start_max;

            total_min_time += elapsed_min.count();
            total_max_time += elapsed_max.count();
        }

        std::cout << "Armadillo, n=" << n << ", average time per min: " << (total_min_time / repetitions) << " seconds" << std::endl;
        std::cout << "Armadillo, n=" << n << ", average time per max: " << (total_max_time / repetitions) << " seconds" << std::endl;
    }
}

void test_xtensor_min_max() {
    std::cout << "Testing XTensor min/max..." << std::endl;
    for (int n : dimensions) {
        xt::xarray<double> xtensor_matrix = read_matrix_xtensor("data/xtensor_matrix_" + std::to_string(n) + ".txt", n, n);

        double total_min_time = 0.0;
        double total_max_time = 0.0;

        for (int i = 0; i < repetitions; ++i) {
            auto start_min = std::chrono::high_resolution_clock::now();
            double min_val = xt::amin(xtensor_matrix)();
            auto end_min = std::chrono::high_resolution_clock::now();

            auto start_max = std::chrono::high_resolution_clock::now();
            double max_val = xt::amax(xtensor_matrix)();
            auto end_max = std::chrono::high_resolution_clock::now();

            std::chrono::duration<double> elapsed_min = end_min - start_min;
            std::chrono::duration<double> elapsed_max = end_max - start_max;

            total_min_time += elapsed_min.count();
            total_max_time += elapsed_max.count();
        }

        std::cout << "XTensor, n=" << n << ", average time per min: " << (total_min_time / repetitions) << " seconds" << std::endl;
        std::cout << "XTensor, n=" << n << ", average time per max: " << (total_max_time / repetitions) << " seconds" << std::endl;
    }
}

int main() {
    test_eigen_min_max();
    test_armadillo_min_max();
    test_xtensor_min_max();
    return 0;
}
