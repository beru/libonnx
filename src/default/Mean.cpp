#include <onnx.h>
#include "util.h"

namespace {

bool Mean_init(onnx_node_t* n)
{
	return (n->inputs.size() >= 1) && (n->outputs.size() == 1);
}

int Mean_exit(onnx_node_t* n)
{
	return 1;
}

int Mean_reshape(onnx_node_t* n)
{
	onnx_tensor_t* y = n->outputs[0];

	if (!y->reshape_identity(n->inputs[0]))
		return 0;
	for (size_t i = 1; i < n->inputs.size(); i++) {
		if (!y->reshape_multi_broadcast(y, n->inputs[i], y->type))
			return 0;
	}
	return 1;
}

template <typename T>
void Mean_generic(onnx_node_t* n)
{
	onnx_tensor_t* y = n->outputs[0];
	T* py = (T*)y->data;
	for (size_t i = 0, l = y->ndata; i < l; i++) {
		T sum = 0;
		for (size_t j = 0; j < n->inputs.size(); j++) {
			onnx_tensor_t* x = n->inputs[j];
			T* px = (T*)x->broadcast_map_address(y, i);
			sum += *px;
		}
		py[i] = sum / n->inputs.size();
	}
}

GEN_HOLEDR_TYPE(holder, Mean_generic)

} // namespace

void resolver_default_op_Mean(onnx_node_t* n)
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
		n->init = Mean_init;
		n->exit = Mean_exit;
		n->reshape = Mean_reshape;
	}
}
