#include <onnx.h>
#include "float16.h"

static int Softplus_init(onnx_node_t* n)
{
	if ((n->inputs.size() == 1) && (n->outputs.size() == 1))
		return 1;
	return 0;
}

static int Softplus_exit(onnx_node_t* n)
{
	return 1;
}

static int Softplus_reshape(onnx_node_t* n)
{
	onnx_tensor_t* x = n->inputs[0];
	onnx_tensor_t* y = n->outputs[0];

	return y->reshape_identity(x, x->type);
}

template <typename T>
static void Softplus_generic(onnx_node_t* n)
{
	onnx_tensor_t* x = n->inputs[0];
	onnx_tensor_t* y = n->outputs[0];
	T* px = (T*)x->datas;
	T* py = (T*)y->datas;

	for (size_t i = 0, l = y->ndata; i < l; i++)
		py[i] = log(exp(px[i]) + 1);
}

void resolver_default_op_Softplus(onnx_node_t* n)
{
	if (n->opset >= 1) {
		n->ope = onnx_ope_type_selector{
			.float16_ = Softplus_generic<float16_t>,
			.float32_ = Softplus_generic<float>,
			.float64_ = Softplus_generic<double>,
		}.select(n->inputs[0]->type);
	}
	if (n->ope) {
		n->init = Softplus_init;
		n->exit = Softplus_exit;
		n->reshape = Softplus_reshape;
	}
}
