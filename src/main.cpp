#include "autograd/value.hpp"
#include <iostream>

int	main() {
	Value a(2.0);
	Value b(3.0);
	Value a2(5.0);
	Value b2(2.0);
	Value a3(6.0);
	Value b3(3.0);
	Value x(3.0);
	Value n(4.0);

	Value x1(1.0);
	Value y1 = x1.tanh();
	y1.backward();
	std::cout << "x1.grad = \t\t\t" << x1.grad() << "\n\n";

	Value m = -n;
	Value m2 = m.relu();
	Value n2 = n.relu();
	std::cout << "m2.data = \t\t\t" << m2.data() << '\n';
	std::cout << "m2's parent count =\t\t" << m2.parent_count() << '\n';
	std::cout << "n2.data = \t\t\t" << n2.data() << '\n';
	std::cout << "n2's parent count =\t\t" << n2.parent_count() << '\n';

	Value l = -m;
	std::cout << "m.data = \t\t\t" << m.data() << '\n';
	std::cout << "m's parent count =\t\t" << m.parent_count() << '\n';
	std::cout << "l.data = \t\t\t" << l.data() << '\n';
	std::cout << "l's parent count =\t\t" << l.parent_count() << "\n\n";

	Value y = x.power(2.0);
	std::cout << "y.data = \t\t\t" << y.data() << '\n';
	std::cout << "y's parent count =\t\t" << y.parent_count() << "\n\n";

	Value c = a * b;
	Value d = c + a;
	Value c2 = a2 - b2;
	Value c3 = a3 / b3;

	std::cout << "c =\t\t\t\t" << c.data() << '\n';
	std::cout << "c's parent count =\t\t" << c.parent_count() << '\n';
	std::cout << "Operation that produced c:\t" << op_name(c.op()) << "\n\n";

	std::cout << "d =\t\t\t\t" << d.data() << '\n';
	std::cout << "d's parent count =\t\t" << d.parent_count() << '\n';
	std::cout << "Operation that produced d:\t" << op_name(d.op()) << "\n\n";

	d.backward();
	std::cout << "d.grad = " << d.grad() << '\n';
	std::cout << "c.grad = " << c.grad() << '\n';
	std::cout << "a.grad = " << a.grad() << '\n';
	std::cout << "b.grad = " << b.grad() << "\n\n";
	d.zero_grad();
	std::cout << "d.zero_grad() called" << '\n';
	std::cout << "d.grad = " << d.grad() << '\n';
	std::cout << "c.grad = " << c.grad() << '\n';
	std::cout << "a.grad = " << a.grad() << '\n';
	std::cout << "b.grad = " << b.grad() << "\n\n";

	c2.backward();
	std::cout << "c2.data = " << c2.data() << '\n';
	std::cout << "c2.grad = " << c2.grad() << '\n';
	std::cout << "a2.grad = " << a2.grad() << '\n';
	std::cout << "b2.grad = " << b2.grad() << '\n';
	c2.zero_grad();
	std::cout << "c2.zero_grad() called" << '\n';
	std::cout << "c2.grad = " << c2.grad() << '\n';
	std::cout << "a2.grad = " << a2.grad() << '\n';
	std::cout << "b2.grad = " << b2.grad() << "\n\n";

	c3.backward();
	std::cout << "c3.data = " << c3.data() << '\n';
	std::cout << "c3.grad = " << c3.grad() << '\n';
	std::cout << "a3.grad = " << a3.grad() << '\n';
	std::cout << "b3.grad = " << b3.grad() << '\n';
	c3.zero_grad();
	std::cout << "c3.zero_grad() called" << '\n';
	std::cout << "c3.grad = " << c3.grad() << '\n';
	std::cout << "a3.grad = " << a3.grad() << '\n';
	std::cout << "b3.grad = " << b3.grad() << '\n';
}
