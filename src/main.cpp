#include "autograd/value.hpp"
#include <iomanip>
#include <iostream>

int main() {
    std::cout << std::fixed << std::setprecision(6);

    std::cout << "=== Scalar Autograd Demo by Nart Madi ===\n\n";

    // Build a computation graph:
    //
    // c = a * b
    // d = c + a
    //
    //        d
    //       / \
    //      c   a
    //     / \
    //    a   b

    Value a(2.0);
    Value b(3.0);

    Value c = a * b;
    Value d = c + a;

    std::cout << "Forward pass\n";
    std::cout << "------------\n";
    std::cout << "a = " << a.data() << '\n';
    std::cout << "b = " << b.data() << '\n';
    std::cout << "c = a * b = " << c.data() << '\n';
    std::cout << "d = c + a = " << d.data() << "\n\n";

    std::cout << "Graph metadata\n";
    std::cout << "--------------\n";
    std::cout << "c operation: " << op_name(c.op()) << '\n';
    std::cout << "c parents:   " << c.parent_count() << '\n';
    std::cout << "d operation: " << op_name(d.op()) << '\n';
    std::cout << "d parents:   " << d.parent_count() << "\n\n";

    d.backward();

    std::cout << "Backward pass\n";
    std::cout << "-------------\n";
    std::cout << "d.grad = " << d.grad() << '\n';
    std::cout << "c.grad = " << c.grad() << '\n';
    std::cout << "a.grad = " << a.grad() << '\n';
    std::cout << "b.grad = " << b.grad() << "\n\n";

    d.zero_grad();

    std::cout << "After zero_grad()\n";
    std::cout << "-----------------\n";
    std::cout << "d.grad = " << d.grad() << '\n';
    std::cout << "c.grad = " << c.grad() << '\n';
    std::cout << "a.grad = " << a.grad() << '\n';
    std::cout << "b.grad = " << b.grad() << "\n\n";

    std::cout << "Additional operations\n";
    std::cout << "---------------------\n";

    Value x(2.0);

    Value squared = x.power(2.0);
    Value negated = -x;
    Value activated = negated.relu();
    Value normalized = x.tanh();

    std::cout << "x          = " << x.data() << '\n';
    std::cout << "x^2        = " << squared.data() << '\n';
    std::cout << "-x         = " << negated.data() << '\n';
    std::cout << "ReLU(-x)   = " << activated.data() << '\n';
    std::cout << "tanh(x)    = " << normalized.data() << '\n';

    return 0;
}
