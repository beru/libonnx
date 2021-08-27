#include <onnx.h>
#include "util.h"

namespace {

bool Acos_init(onnx_node_t* n)
{
	return is_inout_size(n, 1, 1);
}

int Acos_reshape(onnx_node_t* n)
{
	onnx_tensor_t* x = n->inputs[0];
	onnx_tensor_t* y = n->outputs[0];

	return y->reshape_identity(x);
}

template <typename T>
void Acos_generic(onnx_node_t* n)
{
	foreach_tensor<T>(n, [](auto x){return acos(x);});
}

GEN_HOLEDR_TYPE(holder, Acos_generic)

} // namespace {

void resolver_default_op_Acos(onnx_node_t* n)
{
	if (n->opset >= 7) {
		n->ope = onnx_ope_type_select<holder,
			float16_t, float, double
		>(n->inputs[0]->type);
	}
	if (n->ope) {
		n->init = Acos_init;
		n->reshape = Acos_reshape;
	}
}

