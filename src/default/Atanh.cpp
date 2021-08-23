#include <onnx.h>
#include "util.h"

namespace {

bool Atanh_init(onnx_node_t* n)
{
	return is_inout_size(n, 1, 1);
}

int Atanh_exit(onnx_node_t* n)
{
	return 1;
}

int Atanh_reshape(onnx_node_t* n)
{
	onnx_tensor_t* x = n->inputs[0];
	onnx_tensor_t* y = n->outputs[0];

	return y->reshape_identity(x);
}

template <typename T>
void Atanh_generic(onnx_node_t* n)
{
	onnx_tensor_t* x = n->inputs[0];
	onnx_tensor_t* y = n->outputs[0];
	T* px = (T*)x->data;
	T* py = (T*)y->data;

	for (size_t i = 0, l = y->ndata; i < l; i++)
		py[i] = atanh(px[i]);
}

GEN_HOLEDR_TYPE(holder, Atanh_generic)

} // namespace

void resolver_default_op_Atanh(onnx_node_t* n)
{
	if (n->opset >= 9) {
		n->ope = onnx_ope_type_select<holder,
			float16_t, float, double
		>(n->inputs[0]->type);
	}
	if (n->ope) {
		n->init = Atanh_init;
		n->exit = Atanh_exit;
		n->reshape = Atanh_reshape;
	}
}
