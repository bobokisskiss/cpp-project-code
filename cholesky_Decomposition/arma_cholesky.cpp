#include <iostream>
#include <armadillo>
#include <chrono>

void cholesky_with_armadillo(int n) {
    // 生成随机正定对称矩阵
    arma::mat A = arma::randu<arma::mat>(n, n);
    A = A * A.t(); // 将随机矩阵转为正定对称矩阵
    
    // 进行 Cholesky 分解
    arma::mat L;
    auto start = std::chrono::high_resolution_clock::now();
    bool success = arma::chol(L, A);
    auto end = std::chrono::high_resolution_clock::now();
    
    if (!success) {
        std::cerr << "Cholesky decomposition failed!" << std::endl;
        return;
    }
    
    // 计算 Cholesky 分解时间
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Armadillo, n=" << n << ", Cholesky decomposition time: " << elapsed.count() << " seconds" << std::endl;
}

int main() {
    std::vector<int> dimensions = {5, 10, 50, 100, 500, 1000, 5000, 10000};
    
    for (int n : dimensions) {
        cholesky_with_armadillo(n);
    }
    
    return 0;
}
