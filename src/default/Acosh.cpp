#include <onnx.h>
#include "float16.h"

namespace {

int Acosh_init(onnx_node_t* n)
{
	if ((n->inputs.size() == 1) && (n->outputs.size() == 1))
		return 1;
	return 0;
}

int Acosh_exit(onnx_node_t* n)
{
	return 1;
}

int Acosh_reshape(onnx_node_t* n)
{
	onnx_tensor_t* x = n->inputs[0];
	onnx_tensor_t* y = n->outputs[0];

	return y->reshape_identity(x, x->type);
}

template <typename T>
void Acosh_generic(onnx_node_t* n)
{
	onnx_tensor_t* x = n->inputs[0];
	onnx_tensor_t* y = n->outputs[0];
	T* px = (T*)x->datas;
	T* py = (T*)y->datas;
	size_t i, l;

	for (i = 0, l = y->ndata; i < l; i++) {
		py[i] = acosh(px[i]);
	}
}

} // namespace {

void resolver_default_op_Acosh(onnx_node_t* n)
{
	if (n->opset >= 9) {
		n->ope = onnx_ope_type_selector{
			.float16_ = Acosh_generic<float16_t>,
			.float32_ = Acosh_generic<float>,
			.float64_ = Acosh_generic<double>,
		}.select(n->inputs[0]->type);
	}
	if (n->ope) {
		n->init = Acosh_init;
		n->exit = Acosh_exit;
		n->reshape = Acosh_reshape;
	}
}
