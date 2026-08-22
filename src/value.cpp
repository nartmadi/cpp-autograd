#include "autograd/value.hpp"

const char* op_name(Op op) {
    switch(op) {
        case Op::None:
            return "None";
        case Op::Add:
            return "Add";
        case Op::Multiply:
            return "Multiply";
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

