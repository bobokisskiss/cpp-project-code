#include <iostream>
#include <chrono>
#include <xtensor/xrandom.hpp>
#include <xtensor-blas/xlinalg.hpp>

std::vector<int> dimensions = {5, 10, 50, 100, 500, 1000, 5000, 10000};

void test_qr_decomposition_time() {
    for (int n : dimensions) {
        // 生成随机矩阵
        xt::xtensor<double, 2> A = xt::random::randn<double>({n, n});

        // 进行QR分解并测量时间
        auto start = std::chrono::steady_clock::now();
        auto QR = xt::linalg::qr(A);
        auto end = std::chrono::steady_clock::now();
        std::chrono::duration<double> elapsed_seconds = end - start;
        double qr_time = elapsed_seconds.count();

        // 输出QR分解时间
        std::cout << "xtensor, n=" << n << ", QR decomposition time: " << qr_time << " seconds" << std::endl;
    }
}

int main() {
    test_qr_decomposition_time();
    return 0;
}
