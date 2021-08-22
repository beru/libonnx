#include <onnx.h>
#include "float16.h"
#include "bfloat16.h"

static int Sum_init(onnx_node_t* n)
{
	if ((n->inputs.size() >= 1) && (n->outputs.size() == 1))
		return 1;
	return 0;
}

static int Sum_exit(onnx_node_t* n)
{
	return 1;
}

static int Sum_reshape(onnx_node_t* n)
{
	onnx_tensor_t* y = n->outputs[0];
	int i;

	if (!y->reshape_identity(n->inputs[0], n->inputs[0]->type))
		return 0;
	for (i = 1; i < n->inputs.size(); i++) {
		if (!y->reshape_multi_broadcast(y, n->inputs[i], y->type))
			return 0;
	}
	return 1;
}

template <typename T>
static void Sum_generic(onnx_node_t* n)
{
	onnx_tensor_t* y = n->outputs[0];
	onnx_tensor_t* x;
	T* py = (T*)y->datas;
	T* px;
	T sum;
	int j;

	for (size_t i = 0, l = y->ndata; i < l; i++) {
		for (j = 0, sum = 0; j < n->inputs.size(); j++) {
			x = n->inputs[j];
			px = (T*)x->broadcast_map_address(y, i);
			sum += *px;
		}
		py[i] = sum;
	}
}

void resolver_default_op_Sum(onnx_node_t* n)
{
	if (n->opset >= 13) {
		n->ope = onnx_ope_type_selector{
			.bfloat16_ = Sum_generic<bfloat16_t>,
			.float16_ = Sum_generic<float16_t>,
			.float32_ = Sum_generic<float>,
			.float64_ = Sum_generic<double>,
		}.select(n->inputs[0]->type);
	}else if (n->opset >= 8) {
		n->ope = onnx_ope_type_selector{
			.float16_ = Sum_generic<float16_t>,
			.float32_ = Sum_generic<float>,
			.float64_ = Sum_generic<double>,
		}.select(n->inputs[0]->type);
	}else if (n->opset >= 6) {
		n->ope = onnx_ope_type_selector{
			.float16_ = Sum_generic<float16_t>,
			.float32_ = Sum_generic<float>,
			.float64_ = Sum_generic<double>,
		}.select(n->inputs[0]->type);
	}else if (n->opset >= 1) {
		n->ope = onnx_ope_type_selector{
			.float16_ = Sum_generic<float16_t>,
			.float32_ = Sum_generic<float>,
			.float64_ = Sum_generic<double>,
		}.select(n->inputs[0]->type);
	}
	if (n->ope) {
		n->init = Sum_init;
		n->exit = Sum_exit;
		n->reshape = Sum_reshape;
	}
}
