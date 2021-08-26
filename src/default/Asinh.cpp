#include <onnx.h>
#include "util.h"

namespace {

bool Asinh_init(onnx_node_t* n)
{
	return is_inout_size(n, 1, 1);
}

int Asinh_exit(onnx_node_t* n)
{
	return 1;
}

int Asinh_reshape(onnx_node_t* n)
{
	onnx_tensor_t* x = n->inputs[0];
	onnx_tensor_t* y = n->outputs[0];

	return y->reshape_identity(x);
}

template <typename T>
void Asinh_generic(onnx_node_t* n)
{
	foreach_tensor<T>(n, [](auto x){return asinh(x);});
}

GEN_HOLEDR_TYPE(holder, Asinh_generic)

} // namespace

void resolver_default_op_Asinh(onnx_node_t* n)
{
	if (n->opset >= 9) {
		n->ope = onnx_ope_type_select<holder,
			float16_t, float, double
		>(n->inputs[0]->type);
	}
	if (n->ope) {
		n->init = Asinh_init;
		n->exit = Asinh_exit;
		n->reshape = Asinh_reshape;
	}
}
