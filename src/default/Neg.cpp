#include <onnx.h>
#include "util.h"

namespace {

bool Neg_init(onnx_node_t* n)
{
	return is_inout_size(n, 1, 1);
}

int Neg_reshape(onnx_node_t* n)
{
	onnx_tensor_t* x = n->inputs[0];
	onnx_tensor_t* y = n->outputs[0];

	return y->reshape_identity(x);
}

template <typename T>
void Neg_generic(onnx_node_t* n)
{
	foreach_tensor<T>(n, [](auto x){return -x;});
}

GEN_HOLEDR_TYPE(holder, Neg_generic)

} // namespace

void resolver_default_op_Neg(onnx_node_t* n)
{
	if (n->opset >= 13) {
		n->ope = onnx_ope_type_select<holder,
			int8_t, int16_t, int32_t, int64_t,
			bfloat16_t, float16_t, float, double
		>(n->inputs[0]->type);
	}else if (n->opset >= 6) {
		n->ope = onnx_ope_type_select<holder,
			int8_t, int16_t, int32_t, int64_t,
			float16_t, float, double
		>(n->inputs[0]->type);
	}else if (n->opset >= 1) {
		n->ope = onnx_ope_type_select<holder,
			float16_t, float, double
		>(n->inputs[0]->type);
	}
	if (n->ope) {
		n->init = Neg_init;
		n->reshape = Neg_reshape;
	}
}
