#include <iostream>
#include <vector>
#include "alglibmisc.h"


#include "linalg.h"

using namespace alglib;
using namespace std;

void matrix_eigenvalues_test(int size) {
    // Generate a random matrix
    real_2d_array a;
    a.setlength(size, size);
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            a(i, j) = rand() % 100;
        }
    }

    // Calculate eigenvalues
    ae_vector wr;
    ae_vector wi;
    rmatrixevd(a, size, 0, &wr, &wi, NULL, NULL, NULL);

    // Print eigenvalues
    cout << "Eigenvalues (Real part): " << wr.tostring() << endl;
    cout << "Eigenvalues (Imaginary part): " << wi.tostring() << endl;
}

int main() {
    matrix_eigenvalues_test(3);   // Low dimension
    matrix_eigenvalues_test(100); // High dimension

    return 0;
}
