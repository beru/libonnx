#include <onnx.h>
#include "float16.h"

namespace {

int Atan_init(onnx_node_t* n)
{
	if ((n->inputs.size() == 1) && (n->outputs.size() == 1))
		return 1;
	return 0;
}

int Atan_exit(onnx_node_t* n)
{
	return 1;
}

int Atan_reshape(onnx_node_t* n)
{
	onnx_tensor_t* x = n->inputs[0];
	onnx_tensor_t* y = n->outputs[0];

	return y->reshape_identity(x, x->type);
}

template <typename T>
void Atan_generic(onnx_node_t* n)
{
	onnx_tensor_t* x = n->inputs[0];
	onnx_tensor_t* y = n->outputs[0];
	T* px = (T*)x->datas;
	T* py = (T*)y->datas;

	for (size_t i = 0, l = y->ndata; i < l; i++)
		py[i] = atan(px[i]);
}

} // namespace

void resolver_default_op_Atan(onnx_node_t* n)
{
	if (n->opset >= 7) {
		n->ope = onnx_ope_type_selector{
			.float16_ = Atan_generic<float16_t>,
			.float32_ = Atan_generic<float>,
			.float64_ = Atan_generic<double>,
		}.select(n->inputs[0]->type);
	}
	if (n->ope) {
		n->init = Atan_init;
		n->exit = Atan_exit;
		n->reshape = Atan_reshape;
	}
}
