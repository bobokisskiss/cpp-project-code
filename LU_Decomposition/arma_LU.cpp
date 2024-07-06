#include <iostream>
#include <armadillo>
#include <chrono>

std::vector<int> dimensions = {5, 10, 50, 100, 500, 1000, 5000, 10000};
const int repetitions = 100;

arma::mat read_matrix(const std::string& filename) {
    arma::mat matrix;
    if (!matrix.load(filename)) {
        std::cerr << "Failed to load matrix from file: " << filename << std::endl;
    }
    return matrix;
}

void test_armadillo_lu() {
    for (int n : dimensions) {
        arma::mat arma_matrix = read_matrix("data/arma_matrix_" + std::to_string(n) + ".txt");

        if (arma_matrix.is_empty()) {
            std::cerr << "Matrix " << n << " is empty, skipping LU decomposition." << std::endl;
            continue;
        }

        double total_time = 0.0;
        for (int i = 0; i < repetitions; ++i) {
            arma::mat L, U;

            auto start = std::chrono::high_resolution_clock::now();
            arma::lu(L, U, arma_matrix);
            auto end = std::chrono::high_resolution_clock::now();

            std::chrono::duration<double> elapsed = end - start;
            total_time += elapsed.count();
        }

        std::cout << "Armadillo, n=" << n << ", average time per LU decomposition: " << (total_time / repetitions) << " seconds" << std::endl;
    }
}

int main() {
    test_armadillo_lu();
    return 0;
}
