#include <iostream>
#include <Eigen>
#include <chrono>

void qr_decomposition_with_eigen() {
    std::vector<int> dimensions = {5, 10, 50, 100, 500, 1000, 5000};

    for (int n : dimensions) {
        // 加载矩阵和其QR分解结果
        Eigen::MatrixXd A(n, n);
        Eigen::MatrixXd Q(n, n);
        Eigen::MatrixXd R(n, n);

        // 从文件加载矩阵和其分解结果
        A = Eigen::MatrixXd::Random(n, n);  // 这里仅为示例，实际应从文件加载矩阵 A

        auto start = std::chrono::high_resolution_clock::now();
        // 执行 Eigen 的 QR 分解操作
        // 请根据 Eigen 的文档或示例适当调整QR分解操作
        Eigen::HouseholderQR<Eigen::MatrixXd> qr(A);
        Q = qr.householderQ();
        R = qr.matrixQR().triangularView<Eigen::Upper>();

        auto end = std::chrono::high_resolution_clock::now();

        // 计算QR分解时间
        std::chrono::duration<double> elapsed = end - start;

        // 输出测试结果
        std::cout << "Eigen, n=" << n << ", QR decomposition completed in " << elapsed.count() << " seconds." << std::endl;
    }
}

int main() {
    qr_decomposition_with_eigen();
    return 0;
}
