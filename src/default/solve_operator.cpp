#include "solve_operator.h"

namespace onnx {

#define X(name) operator_t* resolver_default_op_ ## name(int opset);
#include "ops.h"
#undef X

namespace {

std::map<std::string_view, operator_t*(*)(int opset)> op_map {
#define X(name) { #name, resolver_default_op_ ## name },
#include "ops.h"
#undef X
};

} // namespace {

operator_t* solve_operator(std::string_view op_type, int opset)
{
	auto it = op_map.find(op_type);
	if (it != op_map.end()) {
		return it->second(opset);
	}
	return nullptr;
}

} // namespace onnx

