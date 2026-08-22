#include "autograd/value.hpp"
#include <iostream>

int main() {
    Value a(2.0);
    Value b(3.0);

    Value c = a * b;
    Value d = c + a;

    std::cout << "c =\t\t\t\t" << c.data() << '\n';
    std::cout << "c's parent count =\t\t" << c.parent_count() << '\n';
    std::cout << "Operation that produced c:\t" << op_name(c.op()) << "\n\n";

    std::cout << "d =\t\t\t\t" << d.data() << '\n';
    std::cout << "d's parent count =\t\t" << d.parent_count() << '\n';
    std::cout << "Operation that produced d:\t" << op_name(d.op()) << '\n';
}

