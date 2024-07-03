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

// Specialized template for xt::xarray
template<>
xt::xarray<double> read_matrix<xt::xarray<double>>(const std::string& filename, int rows, int cols) {
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
    for (int n : dimensions) {
        std::string filename = "data/eigen_matrix_" + std::to_string(n) + ".txt";
        Eigen::MatrixXd eigen_matrix = read_matrix<Eigen::MatrixXd>(filename, n, n);
        
        double total_time = 0.0;
        for (int i = 0; i < repetitions; ++i) {
            auto start = std::chrono::high_resolution_clock::now();
            double sum = eigen_matrix.sum();
            auto end = std::chrono::high_resolution_clock::now();

            std::chrono::duration<double> elapsed = end - start;
            total_time += elapsed.count();
        }
        
        std::cout << "Eigen, n=" << n << ", average time per sum: " << (total_time / repetitions) << " seconds" << std::endl;
    }
}

void test_armadillo() {
    std::cout << "Testing Armadillo..." << std::endl;
    for (int n : dimensions) {
        std::string filename = "data/arma_matrix_" + std::to_string(n) + ".txt";
        arma::mat arma_matrix;
        arma_matrix.load(filename);

        double total_time = 0.0;
        for (int i = 0; i < repetitions; ++i) {
            auto start = std::chrono::high_resolution_clock::now();
            double sum = arma::accu(arma_matrix);
            auto end = std::chrono::high_resolution_clock::now();

            std::chrono::duration<double> elapsed = end - start;
            total_time += elapsed.count();
        }

        std::cout << "Armadillo, n=" << n << ", average time per sum: " << (total_time / repetitions) << " seconds" << std::endl;
    }
}

void test_xtensor() {
    std::cout << "Testing XTensor..." << std::endl;
    for (int n : dimensions) {
        std::string filename = "data/xtensor_matrix_" + std::to_string(n) + ".txt";
        xt::xarray<double> xtensor_matrix = read_matrix<xt::xarray<double>>(filename, n, n);

        double total_time = 0.0;
        for (int i = 0; i < repetitions; ++i) {
            auto start = std::chrono::high_resolution_clock::now();
            double sum = xt::sum(xtensor_matrix)();
            auto end = std::chrono::high_resolution_clock::now();

            std::chrono::duration<double> elapsed = end - start;
            total_time += elapsed.count();
        }

        std::cout << "XTensor, n=" << n << ", average time per sum: " << (total_time / repetitions) << " seconds" << std::endl;
    }
}

int main() {
    test_eigen();
    test_armadillo();
    test_xtensor();
    return 0;
}
