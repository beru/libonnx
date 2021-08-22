#include <onnx.h>
#include "float16.h"
#include "bfloat16.h"

static int Neg_init(onnx_node_t* n)
{
	if ((n->inputs.size() == 1) && (n->outputs.size() == 1))
		return 1;
	return 0;
}

static int Neg_exit(onnx_node_t* n)
{
	return 1;
}

static int Neg_reshape(onnx_node_t* n)
{
	onnx_tensor_t* x = n->inputs[0];
	onnx_tensor_t* y = n->outputs[0];

	return y->reshape_identity(x, x->type);
}

template <typename T>
static void Neg_generic(onnx_node_t* n)
{
	onnx_tensor_t* x = n->inputs[0];
	onnx_tensor_t* y = n->outputs[0];
	T* px = (T*)x->datas;
	T* py = (T*)y->datas;

	for (size_t i = 0, l = y->ndata; i < l; i++)
		py[i] = -px[i];
}

void resolver_default_op_Neg(onnx_node_t* n)
{
	if (n->opset >= 13) {
		n->ope = onnx_ope_type_selector{
			.int8_ = Neg_generic<int8_t>,
			.int16_ = Neg_generic<int16_t>,
			.int32_ = Neg_generic<int32_t>,
			.int64_ = Neg_generic<int64_t>,
			.bfloat16_ = Neg_generic<bfloat16_t>,
			.float16_ = Neg_generic<float16_t>,
			.float32_ = Neg_generic<float>,
			.float64_ = Neg_generic<double>,
		}.select(n->inputs[0]->type);
	}else if (n->opset >= 6) {
		n->ope = onnx_ope_type_selector{
			.int8_ = Neg_generic<int8_t>,
			.int16_ = Neg_generic<int16_t>,
			.int32_ = Neg_generic<int32_t>,
			.int64_ = Neg_generic<int64_t>,
			.float16_ = Neg_generic<float16_t>,
			.float32_ = Neg_generic<float>,
			.float64_ = Neg_generic<double>,
		}.select(n->inputs[0]->type);
	}else if (n->opset >= 1) {
		n->ope = onnx_ope_type_selector{
			.float16_ = Neg_generic<float16_t>,
			.float32_ = Neg_generic<float>,
			.float64_ = Neg_generic<double>,
		}.select(n->inputs[0]->type);
	}
	if (n->ope) {
		n->init = Neg_init;
		n->exit = Neg_exit;
		n->reshape = Neg_reshape;
	}
}
