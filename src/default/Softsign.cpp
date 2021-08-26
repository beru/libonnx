#include <onnx.h>
#include "util.h"

namespace {

bool Softsign_init(onnx_node_t* n)
{
	return is_inout_size(n, 1, 1);
}

int Softsign_exit(onnx_node_t* n)
{
	return 1;
}

int Softsign_reshape(onnx_node_t* n)
{
	onnx_tensor_t* x = n->inputs[0];
	onnx_tensor_t* y = n->outputs[0];

	return y->reshape_identity(x);
}

template <typename T>
void Softsign_generic(onnx_node_t* n)
{
	foreach_tensor<T>(n, [](auto x){ return x / (1 + fabs(x)); });
}

GEN_HOLEDR_TYPE(holder, Softsign_generic)

} // namespace

void resolver_default_op_Softsign(onnx_node_t* n)
{
	if (n->opset >= 1) {
		n->ope = onnx_ope_type_select<holder,
			float16_t, float, double
		>(n->inputs[0]->type);
	}
	if (n->ope) {
		n->init = Softsign_init;
		n->exit = Softsign_exit;
		n->reshape = Softsign_reshape;
	}
}
