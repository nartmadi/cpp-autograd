#include "autograd/value.hpp"
#include <set>
#include <functional>

const char* op_name(Op op) {
	switch(op) {
		case Op::None:
			return "None";
		case Op::Add:
			return "Add";
		case Op::Multiply:
			return "Multiply";
		case Op::Subtract:
			return "Subtract";
	}
	return "Unknown";
}

Value::Value(double data) : node(std::make_shared<Node>()) {
	node->data = data;
}

double  Value::data() const {
	return node->data;
}

double  Value::grad() const {
	return node->grad;
}

Op      Value::op() const {
	return node->op;
}

std::size_t Value::parent_count() const {
	return node->parents.size();
}

Value	operator+(const Value& lhs, const Value& rhs) {
	Value result(lhs.data() + rhs.data());

	result.node->op = Op::Add;
	result.node->parents = {lhs.node, rhs.node};

	return result;
}

Value	operator*(const Value& lhs, const Value& rhs) {
	Value result(lhs.data() * rhs.data());

	result.node->op = Op::Multiply;
	result.node->parents = {lhs.node, rhs.node};

	return result;
}

Value	operator-(const Value& lhs, const Value& rhs) {
	Value result(lhs.data() - rhs.data());

	result.node->op = Op::Subtract;
	result.node->parents = {lhs.node, rhs.node};

	return result;
}

void	Value::backward() {
	std::vector<std::shared_ptr<Node>>	topo;
	std::set<Node*>	visited;

	std::function<void(const std::shared_ptr<Node>&)> dfs;
	dfs = [&](const std::shared_ptr<Node>& n) {
		if (visited.contains(n.get())) {
			return;
		}
		visited.insert(n.get());
		for (const std::shared_ptr<Node>& parent : n->parents) {
			dfs(parent);
		}
		topo.push_back(n);
	};

	dfs(node);
	node->grad = 1.0;

	for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
		std::shared_ptr<Node>	n = *it;

		if (n->op == Op::Add) {
			n->parents[0]->grad += n->grad;
			n->parents[1]->grad += n->grad;
		}
		else if (n->op == Op::Multiply) {
			n->parents[0]->grad += n->grad * n->parents[1]->data;
			n->parents[1]->grad += n->grad * n->parents[0]->data;
		}
		else if (n->op == Op::Subtract) {
			n->parents[0]->grad += n->grad;
			n->parents[1]->grad -= n->grad;
		}
	}
}
