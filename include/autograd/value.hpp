#pragma once

#include <vector>
#include <memory>

enum class Op {
	None,
	Add,
	Multiply,
	Subtract,
	Divide,
	Power,
	Negate
};

const char* op_name(Op op);

class Value {
private:
	struct Node {
		double								data;           // Scalar value produced during forward pass.
		double								grad = 0.0;     // Gradient of the final output with respect to this node.
		Op									op = Op::None;  // Operation that produced this node.
		std::vector<std::shared_ptr<Node>>	parents;        // Direct input nodes used to produce this node.
		double								exponent;		// Exponent used by the power operation that produced this node.
	};

	std::shared_ptr<Node>	node;                             // Shared handle to this Value's computation-graph node.

public:
	Value(double data);

	double		data() const;
	double		grad() const;
	Op			op() const;
	std::size_t	parent_count() const;

	friend Value	operator+(const Value& lhs, const Value& rhs);
	friend Value	operator*(const Value& lhs, const Value& rhs);
	friend Value	operator-(const Value& lhs, const Value& rhs);
	friend Value	operator/(const Value& lhs, const Value& rhs);
	Value			operator-() const;
	Value			power(double exponent) const;

	void	backward();
};

