#include <iostream>
#include <vector>
#include <chrono>
#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xrandom.hpp>
#include <xtensor-blas/xlinalg.hpp>

std::vector<int> dimensions = {5, 10, 50, 100, 500, 1000, 5000};
const int repetitions = 100;

xt::xarray<double> generate_non_singular_matrix(int n) {
    xt::xarray<double> matrix;
    bool invertible = false;

    while (!invertible) {
        matrix = xt::random::rand<double>({n, n});
        
        // Ensure diagonal elements are non-zero to avoid singular matrix
        for (int i = 0; i < n; ++i) {
            matrix(i, i) += 1.0;
        }
        
        try {
            auto inverted = xt::linalg::inv(matrix);
            invertible = true;
        } catch (const std::runtime_error& e) {
            invertible = false;
        }
    }
    return matrix;
}

void monte_carlo_test(int n) {
    xt::xarray<double> matrix = generate_non_singular_matrix(n);

    double total_time = 0;
    int successful_repetitions = 0;

    for (int i = 0; i < repetitions; ++i) {
        try {
            auto start = std::chrono::high_resolution_clock::now();
            auto inverse_matrix = xt::linalg::inv(matrix);
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

int main() {
    for (int n : dimensions) {
        monte_carlo_test(n);
    }
    return 0;
}
