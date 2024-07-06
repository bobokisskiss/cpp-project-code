#include <iostream>
#include <chrono>
#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xrandom.hpp>
#include <xtensor-blas/xlinalg.hpp>

void cholesky_with_xtensor() {
    std::vector<int> dimensions = {5, 10, 50, 100, 500, 1000, 5000};

    for (int n : dimensions) {
        // 生成随机正定矩阵
        xt::xarray<double> A = xt::random::randn<double>({n, n});
        A = xt::linalg::dot(xt::transpose(A), A) + xt::eye<double>(n) * 0.001;

        // 进行 Cholesky 分解
        xt::xarray<double> L;
        auto start = std::chrono::high_resolution_clock::now();
        L = xt::linalg::cholesky(A);
        auto end = std::chrono::high_resolution_clock::now();

        // 计算 Cholesky 分解时间
        std::chrono::duration<double> elapsed = end - start;

        // 输出测试结果
        std::cout << "xtensor, n=" << n << ", Cholesky decomposition time: " << elapsed.count() << " seconds" << std::endl;
    }
}

int main() {
    cholesky_with_xtensor();
    return 0;
}
