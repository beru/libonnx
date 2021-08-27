#include <onnx.h>
#include "util.h"

namespace {

bool Acosh_init(onnx_node_t* n)
{
	return is_inout_size(n, 1, 1);
}

int Acosh_reshape(onnx_node_t* n)
{
	onnx_tensor_t* x = n->inputs[0];
	onnx_tensor_t* y = n->outputs[0];

	return y->reshape_identity(x);
}

template <typename T>
void Acosh_generic(onnx_node_t* n)
{
	foreach_tensor<T>(n, [](auto x){return acosh(x);});
}

GEN_HOLEDR_TYPE(holder, Acosh_generic)

} // namespace {

void resolver_default_op_Acosh(onnx_node_t* n)
{
	if (n->opset >= 9) {
		n->ope = onnx_ope_type_select<holder,
			float16_t, float, double
		>(n->inputs[0]->type);
	}
	if (n->ope) {
		n->init = Acosh_init;
		n->reshape = Acosh_reshape;
	}
}
