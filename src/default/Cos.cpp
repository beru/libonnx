#include <onnx.h>
#include "float16.h"

static int Cos_init(onnx_node_t* n)
{
	if ((n->inputs.size() == 1) && (n->outputs.size() == 1))
		return 1;
	return 0;
}

static int Cos_exit(onnx_node_t* n)
{
	return 1;
}

static int Cos_reshape(onnx_node_t* n)
{
	onnx_tensor_t* x = n->inputs[0];
	onnx_tensor_t* y = n->outputs[0];

	return y->reshape_identity(x, x->type);
}

template <typename T>
static void Cos_generic(onnx_node_t* n)
{
	onnx_tensor_t* x = n->inputs[0];
	onnx_tensor_t* y = n->outputs[0];
	T* px = (T*)x->datas;
	T* py = (T*)y->datas;

	for (size_t i = 0, l = y->ndata; i < l; i++)
		py[i] = cos(px[i]);
}

void resolver_default_op_Cos(onnx_node_t* n)
{
	if (n->opset >= 7) {
		n->ope = onnx_ope_type_selector{
			.float16_ = Cos_generic<float16_t>,
			.float32_ = Cos_generic<float>,
			.float64_ = Cos_generic<double>,
		}.select(n->inputs[0]->type);
	}
	if (n->ope) {
		n->init = Cos_init;
		n->exit = Cos_exit;
		n->reshape = Cos_reshape;
	}
}
