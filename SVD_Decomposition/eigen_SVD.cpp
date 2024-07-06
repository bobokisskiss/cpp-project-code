#include <iostream>
#include <Eigen/Dense>
#include <chrono>

void svd_test() {
    std::vector<int> dimensions = {5, 10, 50, 100, 500, 1000, 5000};

    for (int n : dimensions) {
        // 生成随机矩阵
        Eigen::MatrixXd A = Eigen::MatrixXd::Random(n, n);

        // 进行SVD分解
        Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeThinU | Eigen::ComputeThinV);
        auto start = std::chrono::high_resolution_clock::now();
        svd.compute(A, Eigen::ComputeThinU | Eigen::ComputeThinV);
        auto end = std::chrono::high_resolution_clock::now();

        // 计算SVD分解时间
        std::chrono::duration<double> elapsed = end - start;

        // 输出测试结果
        std::cout << "Eigen, n=" << n << ", SVD decomposition time: " << elapsed.count() << " seconds" << std::endl;
    }
}

int main() {
    svd_test();
    return 0;
}
