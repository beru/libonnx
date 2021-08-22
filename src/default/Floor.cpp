#include <onnx.h>
#include "float16.h"
#include "bfloat16.h"

static int Floor_init(onnx_node_t* n)
{
	if ((n->inputs.size() == 1) && (n->outputs.size() == 1))
		return 1;
	return 0;
}

static int Floor_exit(onnx_node_t* n)
{
	return 1;
}

static int Floor_reshape(onnx_node_t* n)
{
	onnx_tensor_t* x = n->inputs[0];
	onnx_tensor_t* y = n->outputs[0];

	return y->reshape_identity(x, x->type);
}

template <typename T>
static void Floor_generic(onnx_node_t* n)
{
	onnx_tensor_t* x = n->inputs[0];
	onnx_tensor_t* y = n->outputs[0];
	T* px = (T*)x->datas;
	T* py = (T*)y->datas;

	for (size_t i = 0, l = y->ndata; i < l; i++)
		py[i] = floor(px[i]);
}

void resolver_default_op_Floor(onnx_node_t* n)
{
	if (n->opset >= 13) {
		n->ope = onnx_ope_type_selector{
			.bfloat16_ = Floor_generic<bfloat16_t>,
			.float16_ = Floor_generic<float16_t>,
			.float32_ = Floor_generic<float>,
			.float64_ = Floor_generic<double>,
		}.select(n->inputs[0]->type);
	}else if (n->opset >= 6) {
		n->ope = onnx_ope_type_selector{
			.float16_ = Floor_generic<float16_t>,
			.float32_ = Floor_generic<float>,
			.float64_ = Floor_generic<double>,
		}.select(n->inputs[0]->type);
	}else if (n->opset >= 1) {
		n->ope = onnx_ope_type_selector{
			.float16_ = Floor_generic<float16_t>,
			.float32_ = Floor_generic<float>,
			.float64_ = Floor_generic<double>,
		}.select(n->inputs[0]->type);
	}
	if (n->ope) {
		n->init = Floor_init;
		n->exit = Floor_exit;
		n->reshape = Floor_reshape;
	}
}
