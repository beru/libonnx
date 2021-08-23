#include <onnx.h>
#include "util.h"

namespace {

bool Relu_init(onnx_node_t* n)
{
	return is_inout_size(n, 1, 1);
}

int Relu_exit(onnx_node_t* n)
{
	return 1;
}

int Relu_reshape(onnx_node_t* n)
{
	onnx_tensor_t* x = n->inputs[0];
	onnx_tensor_t* y = n->outputs[0];

	return y->reshape_identity(x);
}

template <typename T>
void Relu_generic(onnx_node_t* n)
{
	onnx_tensor_t* x = n->inputs[0];
	onnx_tensor_t* y = n->outputs[0];
	T* px = (T*)x->data;
	T* py = (T*)y->data;

	for (size_t i = 0, l = y->ndata; i < l; i++) {
		if (px[i] < 0) {
			py[i] = 0;
		}else {
			py[i] = px[i];
		}
	}
}

GEN_HOLEDR_TYPE(holder, Relu_generic)

} // namespace

void resolver_default_op_Relu(onnx_node_t* n)
{
	if (n->opset >= 14) {
		n->ope = onnx_ope_type_select<holder,
			int8_t, int16_t, int32_t, int64_t,
			bfloat16_t, float16_t, float, double
		>(n->inputs[0]->type);
	}else if (n->opset >= 13) {
		n->ope = onnx_ope_type_select<holder,
			bfloat16_t, float16_t, float, double
		>(n->inputs[0]->type);
	}else if (n->opset >= 6) {
		n->ope = onnx_ope_type_select<holder,
			float16_t, float, double
		>(n->inputs[0]->type);
	}else if (n->opset >= 1) {
		n->ope = onnx_ope_type_select<holder,
			float16_t, float, double
		>(n->inputs[0]->type);
	}
	if (n->ope) {
		n->init = Relu_init;
		n->exit = Relu_exit;
		n->reshape = Relu_reshape;
	}
}
