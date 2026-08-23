#include "autograd/value.hpp"
#include <cassert>
#include <cmath>

bool close(double a, double b) { return std::abs(a - b) < 1e-6; }

int main() {
    // Chained expression
    {
        Value a(2.0);
        Value b(3.0);
        Value c = a * b;
        Value d = c + a;
        d.backward();
        assert(close(d.grad(), 1.0));
        assert(close(c.grad(), 1.0));
        assert(close(a.grad(), 4.0));
        assert(close(b.grad(), 2.0));
    }
    // Shared node
    {
        Value x(2.0);
        Value y = x * x + x;
        y.backward();
        assert(close(x.grad(), 5.0));
    }
    // Gradient reset
    {
        Value a(2.0);
        Value b(3.0);
        Value c = a * b;
        Value d = c + a;
        d.backward();
        d.zero_grad();
        assert(close(d.grad(), 0.0));
        assert(close(c.grad(), 0.0));
        assert(close(a.grad(), 0.0));
        assert(close(b.grad(), 0.0));
    }
    // Backward after gradient reset
    {
        Value a(2.0);
        Value b(3.0);
        Value y = a * b + a;
        y.backward();
        y.zero_grad();
        y.backward();
        assert(close(y.grad(), 1.0));
        assert(close(a.grad(), 4.0));
        assert(close(b.grad(), 2.0));
    }
    // Deeper computation graph
    {
        Value a(2.0);
        Value b(3.0);
        Value c(4.0);
        Value x = a * b;
        Value y = x + c;
        Value z = y.power(2.0);
        z.backward();
        assert(close(z.grad(), 1.0));
        assert(close(y.grad(), 20.0));
        assert(close(x.grad(), 20.0));
        assert(close(a.grad(), 60.0));
        assert(close(b.grad(), 40.0));
        assert(close(c.grad(), 20.0));
    }

    return 0;
}
