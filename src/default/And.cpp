#include <onnx.h>
#include "util.h"

namespace {

bool And_7_init(onnx_node_t* n)
{
	return is_inout_size(n, 2, 1);
}

int And_7_exit(onnx_node_t* n)
{
	return 1;
}

int And_7_reshape(onnx_node_t* n)
{
	onnx_tensor_t* y = n->outputs[0];
	onnx_tensor_t* a = n->inputs[0];
	onnx_tensor_t* b = n->inputs[1];

	return y->reshape_multi_broadcast(a, b, ONNX_TENSOR_TYPE_BOOL);
}

void And_7_bool(onnx_node_t* n)
{
	onnx_tensor_t* y = n->outputs[0];
	onnx_tensor_t* a = n->inputs[0];
	onnx_tensor_t* b = n->inputs[1];
	uint8_t* py = (uint8_t*)y->data;

	for (size_t i = 0, l = y->ndata; i < l; i++) {
		uint8_t* pa = (uint8_t*)a->broadcast_map_address(y, i);
		uint8_t* pb = (uint8_t*)b->broadcast_map_address(y, i);
		py[i] = (*pa && *pb) ? 1 : 0;
	}
}

} // namespace {

void resolver_default_op_And(onnx_node_t* n)
{
	if (n->opset >= 7) {
		switch (n->inputs[0]->type) {
		case ONNX_TENSOR_TYPE_BOOL:
			n->ope = And_7_bool;
			break;
		default:
			break;
		}
	}else if (n->opset >= 1)	{
	}
	if (n->ope) {
		n->init = And_7_init;
		n->exit = And_7_exit;
		n->reshape = And_7_reshape;
	}
}
