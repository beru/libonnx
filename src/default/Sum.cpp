#include <onnx.h>
#include "util.h"

namespace {

bool Sum_init(onnx_node_t* n)
{
	return (n->inputs.size() >= 1) && (n->outputs.size() == 1);
}

int Sum_reshape(onnx_node_t* n)
{
	onnx_tensor_t* y = n->outputs[0];
	int i;

	if (!y->reshape_identity(n->inputs[0]))
		return 0;
	for (i = 1; i < n->inputs.size(); i++) {
		if (!y->reshape_multi_broadcast(y, n->inputs[i], y->type))
			return 0;
	}
	return 1;
}

template <typename T>
void Sum_generic(onnx_node_t* n)
{
	onnx_tensor_t* y = n->outputs[0];
	onnx_tensor_t* x;
	T* py = (T*)y->data;

	for (size_t i = 0, l = y->ndata; i < l; i++) {
		T sum = 0;
		for (size_t j = 0; j < n->inputs.size(); j++) {
			x = n->inputs[j];
			T* px = (T*)x->broadcast_map_address(y, i);
			sum += *px;
		}
		py[i] = sum;
	}
}

GEN_HOLEDR_TYPE(holder, Sum_generic)

} // namespace

void resolver_default_op_Sum(onnx_node_t* n)
{
	if (n->opset >= 13) {
		n->ope = onnx_ope_type_select<holder,
			bfloat16_t, float16_t, float, double
		>(n->inputs[0]->type);
	}else if (n->opset >= 8) {
		n->ope = onnx_ope_type_select<holder,
			float16_t, float, double
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
		n->init = Sum_init;
		n->reshape = Sum_reshape;
	}
}
