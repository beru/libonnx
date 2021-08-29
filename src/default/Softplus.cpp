#include <onnx.h>
#include "util.h"

namespace {

bool Softplus_init(onnx_node_t* n)
{
	return is_inout_size(n, 1, 1);
}

template <typename T>
void Softplus_generic(onnx_node_t* n)
{
	foreach_tensor<T>(n, [](auto x){ return log(exp(x) + 1); });
}

GEN_HOLEDR_TYPE(holder, Softplus_generic)

} // namespace

void resolver_default_op_Softplus(onnx_node_t* n)
{
	if (n->opset >= 1) {
		n->ope = onnx_ope_type_select<holder,
			float16_t, float, double
		>(n->inputs[0]->type);
	}
	if (n->ope) {
		n->init = Softplus_init;
	}
}
