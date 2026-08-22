#include "autograd/value.hpp"

Value::Value(double data) : node(std::make_shared<Node>()) {
    node->data = data;
}

double  Value::data() const {
    return node->data;
}

double  Value::grad() const {
    return node->grad;
}

Value   operator+(const Value& lhs, const Value& rhs) {
    Value result(lhs.data() + rhs.data());

    result.node->op = Op::Add;
    result.node->parents = {lhs.node, rhs.node};
    
    return result;
}


Value   operator*(const Value& lhs, const Value& rhs) {
    Value result(lhs.data() * rhs.data());

    result.node->op = Op::Multiply;
    result.node->parents = {lhs.node, rhs.node};
    
    return result;
}

