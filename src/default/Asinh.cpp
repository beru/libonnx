#include <onnx.h>
#include "float16.h"

namespace {

int Asinh_init(onnx_node_t* n)
{
	if ((n->inputs.size() == 1) && (n->outputs.size() == 1))
		return 1;
	return 0;
}

int Asinh_exit(onnx_node_t* n)
{
	return 1;
}

int Asinh_reshape(onnx_node_t* n)
{
	onnx_tensor_t* x = n->inputs[0];
	onnx_tensor_t* y = n->outputs[0];

	return y->reshape_identity(x, x->type);
}

template <typename T>
void Asinh_generic(onnx_node_t* n)
{
	onnx_tensor_t* x = n->inputs[0];
	onnx_tensor_t* y = n->outputs[0];
	T* px = (T*)x->datas;
	T* py = (T*)y->datas;

	for (size_t i = 0, l = y->ndata; i < l; i++)
		py[i] = asinh(px[i]);
}

} // namespace

void resolver_default_op_Asinh(onnx_node_t* n)
{
	if (n->opset >= 9) {
		n->ope = onnx_ope_type_selector{
			.float16_ = Asinh_generic<float16_t>,
			.float32_ = Asinh_generic<float>,
			.float64_ = Asinh_generic<double>,
		}.select(n->inputs[0]->type);
	}
	if (n->ope) {
		n->init = Asinh_init;
		n->exit = Asinh_exit;
		n->reshape = Asinh_reshape;
	}
}
