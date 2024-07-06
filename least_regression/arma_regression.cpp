#include <iostream>
#include <armadillo>
#include <chrono>

void linear_regression_timing() {
    std::vector<int> dimensions = {5, 10, 50, 100, 500, 1000, 5000};

    for (int n : dimensions) {
        // 加载矩阵和其QR分解结果
        arma::mat A, Q, R;
        A.load("../data/arma_matrix_" + std::to_string(n) + ".txt", arma::raw_ascii);
        Q.load("../data/arma_Q_" + std::to_string(n) + ".txt", arma::raw_ascii);
        R.load("../data/arma_R_" + std::to_string(n) + ".txt", arma::raw_ascii);

        // 确认矩阵加载情况
        std::cout << "Loaded matrix A size: " << A.n_rows << " x " << A.n_cols << std::endl;
        std::cout << "Loaded Q matrix size: " << Q.n_rows << " x " << Q.n_cols << std::endl;
        std::cout << "Loaded R matrix size: " << R.n_rows << " x " << R.n_cols << std::endl;

        // 确认矩阵维度是否正确
        if (Q.n_rows != n || Q.n_cols != n) {
            std::cerr << "Error: Loaded Q matrix dimensions do not match expected dimensions." << std::endl;
            continue;  // 如果维度不匹配，跳过当前迭代
        }
        if (R.n_rows != n || R.n_cols != n) {
            std::cerr << "Error: Loaded R matrix dimensions do not match expected dimensions." << std::endl;
            continue;  // 如果维度不匹配，跳过当前迭代
        }

        // 进行线性回归，例如估计参数 beta
        arma::mat y = arma::randn<arma::mat>(n, 1);  // 生成随机的y向量，这里只是示例
        arma::mat beta;

        auto start = std::chrono::high_resolution_clock::now();
        try {
            beta = arma::solve(R, arma::trans(Q) * y);  // 根据QR分解计算最小二乘解
        } catch (const std::exception &e) {
            std::cerr << "Exception occurred: " << e.what() << std::endl;
            continue;  // 如果计算出现异常，跳过当前迭代
        }
        auto end = std::chrono::high_resolution_clock::now();

        // 计算线性回归运行时间
        std::chrono::duration<double> elapsed = end - start;

        // 输出测试结果
        std::cout << "Armadillo, n=" << n << ", Linear regression completed in " << elapsed.count() << " seconds." << std::endl;
    }
}

int main() {
    linear_regression_timing();
    return 0;
}
