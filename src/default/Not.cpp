#include <onnx.h>
#include "util.h"

namespace {

bool Not_init(onnx_node_t* n)
{
	return is_inout_size(n, 1, 1);
}

int Not_exit(onnx_node_t* n)
{
	return 1;
}

int Not_reshape(onnx_node_t* n)
{
	onnx_tensor_t* x = n->inputs[0];
	onnx_tensor_t* y = n->outputs[0];

	return y->reshape_identity(x, ONNX_TENSOR_TYPE_BOOL);
}

void Not_bool(onnx_node_t* n)
{
	onnx_tensor_t* x = n->inputs[0];
	onnx_tensor_t* y = n->outputs[0];
	uint8_t* px = (uint8_t*)x->data;
	uint8_t* py = (uint8_t*)y->data;

	for (size_t i = 0, l = y->ndata; i < l; i++)
		py[i] = !px[i];
}

} // namespace

void resolver_default_op_Not(onnx_node_t* n)
{
	if (n->opset >= 1) {
		switch (n->inputs[0]->type)	{
		case ONNX_TENSOR_TYPE_BOOL:
			n->ope = Not_bool;
			break;
		default:
			break;
		}
	}
	if (n->ope) {
		n->init = Not_init;
		n->exit = Not_exit;
		n->reshape = Not_reshape;
	}
}
