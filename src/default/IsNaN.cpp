#include <onnx.h>
#include "util.h"

namespace {

bool IsNaN_init(onnx_node_t* n)
{
	return is_inout_size(n, 1, 1);
}

int IsNaN_reshape(onnx_node_t* n)
{
	onnx_tensor_t* x = n->inputs[0];
	onnx_tensor_t* y = n->outputs[0];

	return y->reshape_identity(x, ONNX_TENSOR_TYPE_BOOL);
}

template <typename T>
void IsNaN_generic(onnx_node_t* n)
{
	onnx_tensor_t* x = n->inputs[0];
	onnx_tensor_t* y = n->outputs[0];
	T* px = (T*)x->data;
	uint8_t* py = (uint8_t*)y->data;

	for (size_t i = 0, l = y->ndata; i < l; i++)
		py[i] = isnan(px[i]) ? 1 : 0;
}

GEN_HOLEDR_TYPE(holder, IsNaN_generic)

} // namespace

void resolver_default_op_IsNaN(onnx_node_t* n)
{
	if (n->opset >= 13) {
		n->ope = onnx_ope_type_select<holder,
			bfloat16_t, float16_t, float, double
		>(n->inputs[0]->type);
	}else if (n->opset >= 9) {
		n->ope = onnx_ope_type_select<holder,
			float16_t, float, double
		>(n->inputs[0]->type);
	}
	if (n->ope) {
		n->init = IsNaN_init;
		n->reshape = IsNaN_reshape;
	}
}
