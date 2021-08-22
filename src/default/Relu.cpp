#include <onnx.h>
#include "float16.h"
#include "bfloat16.h"

static int Relu_init(onnx_node_t* n)
{
	if ((n->inputs.size() == 1) && (n->outputs.size() == 1))
		return 1;
	return 0;
}

static int Relu_exit(onnx_node_t* n)
{
	return 1;
}

static int Relu_reshape(onnx_node_t* n)
{
	onnx_tensor_t* x = n->inputs[0];
	onnx_tensor_t* y = n->outputs[0];

	return y->reshape_identity(x, x->type);
}

template <typename T>
static void Relu_generic(onnx_node_t* n)
{
	onnx_tensor_t* x = n->inputs[0];
	onnx_tensor_t* y = n->outputs[0];
	T* px = (T*)x->datas;
	T* py = (T*)y->datas;

	for (size_t i = 0, l = y->ndata; i < l; i++) {
		if (px[i] < 0) {
			py[i] = 0;
		}else {
			py[i] = px[i];
		}
	}
}

void resolver_default_op_Relu(onnx_node_t* n)
{
	if (n->opset >= 14) {
		n->ope = onnx_ope_type_selector{
			.int8_ = Relu_generic<int8_t>,
			.int16_ = Relu_generic<int16_t>,
			.int32_ = Relu_generic<int32_t>,
			.int64_ = Relu_generic<int64_t>,
			.bfloat16_ = Relu_generic<bfloat16_t>,
			.float16_ = Relu_generic<float16_t>,
			.float32_ = Relu_generic<float>,
			.float64_ = Relu_generic<double>,
		}.select(n->inputs[0]->type);
	}else if (n->opset >= 13) {
		n->ope = onnx_ope_type_selector{
			.bfloat16_ = Relu_generic<bfloat16_t>,
			.float16_ = Relu_generic<float16_t>,
			.float32_ = Relu_generic<float>,
			.float64_ = Relu_generic<double>,
		}.select(n->inputs[0]->type);
	}else if (n->opset >= 6) {
		n->ope = onnx_ope_type_selector{
			.float16_ = Relu_generic<float16_t>,
			.float32_ = Relu_generic<float>,
			.float64_ = Relu_generic<double>,
		}.select(n->inputs[0]->type);
	}else if (n->opset >= 1) {
		n->ope = onnx_ope_type_selector{
			.float16_ = Relu_generic<float16_t>,
			.float32_ = Relu_generic<float>,
			.float64_ = Relu_generic<double>,
		}.select(n->inputs[0]->type);
	}
	if (n->ope) {
		n->init = Relu_init;
		n->exit = Relu_exit;
		n->reshape = Relu_reshape;
	}
}
