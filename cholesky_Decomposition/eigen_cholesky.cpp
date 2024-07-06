#include <iostream>
#include <Eigen/Dense>
#include <chrono>

void cholesky_with_eigen() {
    std::vector<int> dimensions = {5, 10, 50, 100, 500, 1000, 5000};

    for (int n : dimensions) {
        // 生成随机正定矩阵
        Eigen::MatrixXd A = Eigen::MatrixXd::Random(n, n);
        Eigen::MatrixXd A_transp = A.transpose();
        A = A * A_transp + Eigen::MatrixXd::Identity(n, n) * 0.001;  // 使得 A 成为正定矩阵

        // 进行 Cholesky 分解
        Eigen::MatrixXd L;
        auto start = std::chrono::high_resolution_clock::now();
        L = A.llt().matrixL();
        auto end = std::chrono::high_resolution_clock::now();

        // 计算 Cholesky 分解时间
        std::chrono::duration<double> elapsed = end - start;

        // 输出测试结果
        std::cout << "Eigen, n=" << n << ", Cholesky decomposition time: " << elapsed.count() << " seconds" << std::endl;
    }
}

int main() {
    cholesky_with_eigen();
    return 0;
}
