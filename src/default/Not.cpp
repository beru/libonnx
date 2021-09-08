#include <onnx.h>
#include "util.h"

namespace onnx {

namespace {

void Not_bool(node_t* n)
{
	tensor_t* x = n->inputs[0];
	tensor_t* y = n->outputs[0];
	bool_t* px = (bool_t*)x->data;
	bool_t* py = (bool_t*)y->data;

	for (size_t i = 0, l = y->ndata; i < l; i++)
		py[i] = !px[i];
}

} // namespace

void resolver_default_op_Not(node_t* n)
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
		n->init = [](node_t* n){
			return is_inout_size(n, 1, 1);
		};
		n->reshape = [](node_t* n){
			tensor_t* x = n->inputs[0];
			tensor_t* y = n->outputs[0];
			return y->reshape_identity(x, ONNX_TENSOR_TYPE_BOOL);
		};
	}
}

} // namespace onnx
