# Scalar Autograd

A scalar reverse-mode automatic differentiation engine written from scratch in C++.

I built this project to understand how automatic differentiation works under the hood in libraries like PyTorch.

Each `Value` represents a scalar in a computation graph and stores both its forward value and gradient.

Calling `backward()` traverses the graph in reverse topological order and computes gradients using the chain rule.

## Features

- Scalar computation graphs
- Reverse-mode automatic differentiation
- Gradient accumulation
- Addition, subtraction, multiplication, and division
- Power and unary negation
- ReLU and tanh
- Gradient reset with `zero_grad()`
- Automated tests with CTest

## Build and run
```bash
cmake -S . -B build
cmake --build build
./build/demo
```

## Build and run tests
```bash
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

## Requirements

* CMake 3.20 or later
* C++23-compatible compiler

## License

MIT
