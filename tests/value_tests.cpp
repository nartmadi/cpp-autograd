#include "autograd/value.hpp"
#include <cassert>
#include <cmath>
#include <stdexcept>

bool	close(double a, double b) {
	return std::abs(a - b) < 1e-6;
}

int	main() {
	// Addition
	{
		Value a(2.0);
		Value b(3.0);
		Value c = a + b;
		assert(close(c.data(), 5.0));
		c.backward();
		assert(close(a.grad(), 1.0));
		assert(close(b.grad(), 1.0));
	}
	// Multiplication
	{
		Value a(2.0);
		Value b(3.0);
		Value c = a * b;
		assert(close(c.data(), 6.0));
		c.backward();
		assert(close(a.grad(), 3.0));
		assert(close(b.grad(), 2.0));
	}
	// Subtraction
	{
		Value a(5.0);
		Value b(2.0);
		Value c = a - b;
		assert(close(c.data(), 3.0));
		c.backward();
		assert(close(a.grad(), 1.0));
		assert(close(b.grad(), -1.0));
	}
	// Division
	{
		Value a(6.0);
		Value b(3.0);
		Value c = a / b;
		assert(close(c.data(), 2.0));
		c.backward();
		assert(close(a.grad(), 1.0 / 3.0));
		assert(close(b.grad(), -2.0 / 3.0));
	}
	// Division by zero
	{
		bool threw = false;
		try {
			Value a(1.0);
			Value b(0.0);
			Value c = a / b;
		}
		catch (const std::domain_error&) {
			threw = true;
		}
		assert(threw);
	}
	// Power
	{
		Value x(3.0);
		Value y = x.power(2.0);
		assert(close(y.data(), 9.0));
		y.backward();
		assert(close(x.grad(), 6.0));
	}
	// Unary negation
	{
		Value x(3.0);
		Value y = -x;
		assert(close(y.data(), -3.0));
		y.backward();
		assert(close(x.grad(), -1.0));
	}
	// ReLU
	{
		Value positive(4.0);
		Value negative(-4.0);
		Value a = positive.relu();
		Value b = negative.relu();
		assert(close(a.data(), 4.0));
		assert(close(b.data(), 0.0));
		a.backward();
		b.backward();
		assert(close(positive.grad(), 1.0));
		assert(close(negative.grad(), 0.0));
	}
	// Tanh
	{
		Value x(1.0);
		Value y = x.tanh();
		assert(close(y.data(), std::tanh(1.0)));
		y.backward();
		assert(close(x.grad(), 1.0 - y.data() * y.data()));
	}

	return 0;
}
