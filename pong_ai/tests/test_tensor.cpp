#include <iostream>
#include "utec/algebra/Tensor.h"

using namespace std;
using namespace utec::algebra;

int main() {
    Tensor<double, 2> row(1, 3);
    row = {10.0, 20.0, 30.0};
    Tensor<double, 2> col(2, 1);
    col = {1.0, 2.0};
    auto M = col + row;
    cout << "row shape: [" << row.shape()[0] << "," << row.shape()[1] << "]\n";
    cout << "col shape: [" << col.shape()[0] << "," << col.shape()[1] << "]\n";
    cout << "result shape: [" << M.shape()[0] << "," << M.shape()[1] << "]\n\n";
    cout << "Broadcast result M:\n" << M << "\n";
    cout << "M.size() == " << M.size() << ", expected 6\n\n";
    Tensor<int, 2> t2(2, 3);
    t2 = {1, 2, 3, 4, 5, 6};
    t2.reshape({3, 2});
    cout << "t2(2,1) = " << t2(2, 1) << " (esperado 6)\n\n";
    try {
        Tensor<int, 3> t3(2, 2, 2);
        t3.reshape({2, 4, 1});
        cout << "ERROR: reshape inválido no lanzó excepción\n";
    } catch (const std::invalid_argument& e) {
        cout << "Excepción capturada correctamente por reshape inválido\n\n";
    }
    Tensor<double, 2> a(2, 2), b(2, 2);
    a.fill(0.0);
    b.fill(2.0);
    a(0, 1) = 5.5;
    auto sum = a + b;
    auto diff = sum - b;
    cout << "sum(0,1) = " << sum(0,1) << " (esperado 7.5)\n";
    cout << "diff(0,1) = " << diff(0,1) << " (esperado 5.5)\n\n";
    Tensor<float, 1> v(3);
    v.fill(2.0f);
    auto scaled = v * 4.0f;
    cout << "scaled(2) = " << scaled(2) << " (esperado 8.0)\n";
    Tensor<int, 3> cube(2, 2, 2);
    cube.fill(1);
    auto cube2 = cube * cube;
    cout << "cube2(1,1,1) = " << cube2(1,1,1) << " (esperado 1)\n\n";
    Tensor<int, 2> m(2,1);
    m(0,0) = 3; m(1,0) = 4;
    Tensor<int, 2> n(2,3);
    n.fill(5);
    auto p = m * n;
    cout << "p(0,2) = " << p(0,2) << " (esperado 15)\n";
    cout << "p(1,1) = " << p(1,1) << " (esperado 20)\n\n";
    Tensor<int, 2> m2(2,3);
    m2 = {1, 2, 3, 4, 5, 6}; // [[1,2,3],[4,5,6]]
    auto mt = m2.transpose_2d();
    cout << "mt.shape() = [" << mt.shape()[0] << "," << mt.shape()[1] << "] (esperado [3,2])\n";
    cout << "mt(0,1) = " << mt(0,1) << " (esperado 4)\n";
    return 0;
}