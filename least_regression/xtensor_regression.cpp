#include <iostream>
#include <xtensor/xarray.hpp>
#include <chrono>

void qr_decomposition_with_xtensor() {
    std::vector<int> dimensions = {5, 10, 50, 100, 500, 1000, 5000};

    for (int n : dimensions) {
        // 加载矩阵和其QR分解结果
        xt::xarray<double> A;
        xt::xarray<double> Q;
        xt::xarray<double> R;

        // 从文件加载矩阵和其分解结果
        // 请根据 xtensor 的文档或示例适当调整文件加载方法
        // 示例中的文件路径仅供参考，实际路径应根据您的设置修改
        std::string matrix_filename = "data/arma_matrix_" + std::to_string(n) + ".txt";
        std::string Q_filename = "data/arma_Q_" + std::to_string(n) + ".txt";
        std::string R_filename = "data/arma_R_" + std::to_string(n) + ".txt";

        // 执行 QR 分解并测量时间
        auto start = std::chrono::high_resolution_clock::now();
        // 在这里执行 xtensor 的 QR 分解操作
        auto end = std::chrono::high_resolution_clock::now();

        // 计算QR分解时间
        std::chrono::duration<double> elapsed = end - start;

        // 输出测试结果
        std::cout << "xtensor, n=" << n << ", QR decomposition completed in " << elapsed.count() << " seconds." << std::endl;
    }
}

int main() {
    qr_decomposition_with_xtensor();
    return 0;
}
