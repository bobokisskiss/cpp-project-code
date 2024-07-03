#include <iostream>
#include <fstream>
#include <chrono>
#include <armadillo>

// Function to read matrix from file
arma::mat read_matrix(const std::string& filename) {
    arma::mat matrix;
    matrix.load(filename);
    return matrix;
}

int main() {
    std::vector<int> dimensions = {5, 10, 50, 100, 500, 1000, 5000, 10000};
    const int repetitions = 100;

    for (int n : dimensions) {
        arma::mat arma_matrix = read_matrix("data/arma_matrix_" + std::to_string(n) + ".txt");

        double total_time = 0.0;

        for (int i = 0; i < repetitions; ++i) {
            auto start = std::chrono::high_resolution_clock::now();
            double arma_mean = arma::accu(arma_matrix) / arma_matrix.n_elem; // Compute mean using sum and count
            auto end = std::chrono::high_resolution_clock::now();

            std::chrono::duration<double> elapsed = end - start;
            total_time += elapsed.count();
        }

        double average_time = total_time / repetitions;

        std::cout << "Armadillo, n=" << n << ", average time per mean: " << average_time << " seconds" << std::endl;
    }

    return 0;
}
