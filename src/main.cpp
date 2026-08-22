#include "autograd/value.hpp"
#include <iostream>

int main() {
    Value a(2.0);
    Value b(3.0);

    Value c = a * b;
    Value d = c + a;

    std::cout << c.data() << '\n';
    std::cout << d.data() << '\n';
}

