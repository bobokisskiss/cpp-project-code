#include <iostream>
#include <chrono>
#include <xtensor/xarray.hpp>
#include <xtensor/xrandom.hpp>
#include <xtensor-blas/xlinalg.hpp>

void svd_test() {
    std::vector<int> dimensions = {5, 10, 50, 100, 500, 1000, 5000};

    for (int n : dimensions) {
        // 生成随机矩阵
        xt::xarray<double> A = xt::random::randn<double>({n, n});

        // 进行SVD分解
        auto start = std::chrono::high_resolution_clock::now();
        auto svd_result = xt::linalg::svd(A);
        auto end = std::chrono::high_resolution_clock::now();

        // 计算SVD分解时间
        std::chrono::duration<double> elapsed = end - start;

        // 输出测试结果
        std::cout << "xtensor, n=" << n << ", SVD decomposition time: " << elapsed.count() << " seconds" << std::endl;
    }
}

int main() {
    svd_test();
    return 0;
}
