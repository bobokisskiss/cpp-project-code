extern "C" {
#include <lapacke.h>
}

#include <iostream>
#include <vector>
#include <complex>

int main() {
    int n = 3;
    std::vector<double> A = {4.0, 1.0, -2.0, 1.0, 4.0, 1.0, -2.0, 1.0, 3.0};
    std::vector<double> wr(n), wi(n);

    LAPACKE_dgeev(LAPACK_ROW_MAJOR, 'N', 'N', n, A.data(), n, wr.data(), wi.data(), nullptr, n, nullptr, n);

    std::cout << "Eigenvalues:\n";
    for(int i = 0; i < n; i++) {
        std::cout << wr[i] << " + " << wi[i] << "i\n";
    }

    return 0;
}
