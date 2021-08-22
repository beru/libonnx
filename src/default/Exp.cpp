#include <onnx.h>
#include "float16.h"
#include "bfloat16.h"

static int Exp_init(onnx_node_t* n)
{
	if ((n->inputs.size() == 1) && (n->outputs.size() == 1))
		return 1;
	return 0;
}

static int Exp_exit(onnx_node_t* n)
{
	return 1;
}

static int Exp_reshape(onnx_node_t* n)
{
	onnx_tensor_t* x = n->inputs[0];
	onnx_tensor_t* y = n->outputs[0];

	return y->reshape_identity(x, x->type);
}

template <typename T>
static void Exp_generic(onnx_node_t* n)
{
	onnx_tensor_t* x = n->inputs[0];
	onnx_tensor_t* y = n->outputs[0];
	T* px = (T*)x->datas;
	T* py = (T*)y->datas;

	for (size_t i = 0, l = y->ndata; i < l; i++)
		py[i] = exp(px[i]);
}

void resolver_default_op_Exp(onnx_node_t* n)
{
	if (n->opset >= 13) {
		n->ope = onnx_ope_type_selector{
			.bfloat16_ = Exp_generic<bfloat16_t>,
			.float16_ = Exp_generic<float16_t>,
			.float32_ = Exp_generic<float>,
			.float64_ = Exp_generic<double>,
		}.select(n->inputs[0]->type);
	}else if (n->opset >= 6) {
		n->ope = onnx_ope_type_selector{
			.float16_ = Exp_generic<float16_t>,
			.float32_ = Exp_generic<float>,
			.float64_ = Exp_generic<double>,
		}.select(n->inputs[0]->type);
	}else if (n->opset >= 1) {
		n->ope = onnx_ope_type_selector{
			.float16_ = Exp_generic<float16_t>,
			.float32_ = Exp_generic<float>,
			.float64_ = Exp_generic<double>,
		}.select(n->inputs[0]->type);
	}
	if (n->ope) {
		n->init = Exp_init;
		n->exit = Exp_exit;
		n->reshape = Exp_reshape;
	}
}
