#include <iostream>
#include <armadillo>
#include <chrono>

std::vector<int> dimensions = {5, 10, 50, 100, 500, 1000, 5000};

void generate_matrices_and_qr() {
    for (int n : dimensions) {
        // 生成随机矩阵
        arma::mat A = arma::randu<arma::mat>(n, n);

        // 进行QR分解
        arma::mat Q, R;
        auto start = std::chrono::high_resolution_clock::now();
        arma::qr(Q, R, A);
        auto end = std::chrono::high_resolution_clock::now();

        // 计算QR分解时间
        std::chrono::duration<double> elapsed = end - start;

        // 将矩阵及其分解结果存储到文件
        std::string matrix_filename = "data/arma_matrix_" + std::to_string(n) + ".txt";
        std::string Q_filename = "data/arma_Q_" + std::to_string(n) + ".txt";
        std::string R_filename = "data/arma_R_" + std::to_string(n) + ".txt";

        A.save(matrix_filename, arma::raw_ascii);
        Q.save(Q_filename, arma::raw_ascii);
        R.save(R_filename, arma::raw_ascii);

        // 输出QR分解时间
        std::cout << "Armadillo, n=" << n << ", QR decomposition time: " << elapsed.count() << " seconds" << std::endl;
    }
}

int main() {
    generate_matrices_and_qr();
    std::cout << "Matrices and their QR decompositions saved successfully." << std::endl;
    return 0;
}
