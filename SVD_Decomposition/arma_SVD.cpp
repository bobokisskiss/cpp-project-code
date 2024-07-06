#include <iostream>
#include <armadillo>
#include <chrono>

void svd_test() {
    std::vector<int> dimensions = {5, 10, 50, 100, 500, 1000, 5000};

    for (int n : dimensions) {
        // 生成随机矩阵
        arma::mat A = arma::randu<arma::mat>(n, n);

        // 进行SVD分解
        arma::mat U, V;
        arma::vec s;
        auto start = std::chrono::high_resolution_clock::now();
        arma::svd(U, s, V, A);
        auto end = std::chrono::high_resolution_clock::now();

        // 计算SVD分解时间
        std::chrono::duration<double> elapsed = end - start;

        // 输出测试结果
        std::cout << "Armadillo, n=" << n << ", SVD decomposition time: " << elapsed.count() << " seconds" << std::endl;
    }
}

int main() {
    svd_test();
    return 0;
}
