#include <iostream>
#include <fstream>
#include <Eigen/Dense>
#include <chrono>

std::vector<int> dimensions = {5, 10, 50, 100, 500, 1000,5000};

void generate_matrices_and_qr() {
    for (int n : dimensions) {
        // 生成随机矩阵
        Eigen::MatrixXd A = Eigen::MatrixXd::Random(n, n);

        // 计时开始
        auto start = std::chrono::high_resolution_clock::now();

        // 进行QR分解
        Eigen::HouseholderQR<Eigen::MatrixXd> qr(A);
        Eigen::MatrixXd Q = qr.householderQ();
        Eigen::MatrixXd R = qr.matrixQR().triangularView<Eigen::Upper>();

        // 计时结束
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = end - start;
        double qr_time = duration.count();

        // 将矩阵及其分解结果存储到文件
        std::string matrix_filename = "data/eigen_matrix_" + std::to_string(n) + ".txt";
        std::string Q_filename = "data/eigen_Q_" + std::to_string(n) + ".txt";
        std::string R_filename = "data/eigen_R_" + std::to_string(n) + ".txt";

        // 保存矩阵A
        std::ofstream matrix_file(matrix_filename);
        if (matrix_file.is_open()) {
            matrix_file << A << std::endl;
            matrix_file.close();
        } else {
            std::cerr << "Failed to save matrix to file: " << matrix_filename << std::endl;
        }

        // 保存矩阵Q
        std::ofstream Q_file(Q_filename);
        if (Q_file.is_open()) {
            Q_file << Q << std::endl;
            Q_file.close();
        } else {
            std::cerr << "Failed to save matrix Q to file: " << Q_filename << std::endl;
        }

        // 保存矩阵R
        std::ofstream R_file(R_filename);
        if (R_file.is_open()) {
            R_file << R << std::endl;
            R_file.close();
        } else {
            std::cerr << "Failed to save matrix R to file: " << R_filename << std::endl;
        }

        // 输出QR分解时间
        std::cout << "Eigen, n=" << n << ", QR decomposition time: " << qr_time << " seconds" << std::endl;
    }
}

int main() {
    generate_matrices_and_qr();
    std::cout << "Matrices and their QR decompositions saved successfully." << std::endl;
    return 0;
}
