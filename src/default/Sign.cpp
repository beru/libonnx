#include <onnx.h>
#include "float16.h"
#include "bfloat16.h"

static int Sign_init(onnx_node_t* n)
{
	if ((n->inputs.size() == 1) && (n->outputs.size() == 1))
		return 1;
	return 0;
}

static int Sign_exit(onnx_node_t* n)
{
	return 1;
}

static int Sign_reshape(onnx_node_t* n)
{
	onnx_tensor_t* x = n->inputs[0];
	onnx_tensor_t* y = n->outputs[0];

	return y->reshape_identity(x, x->type);
}

template <typename T>
static void Sign_generic(onnx_node_t* n)
{
	onnx_tensor_t* x = n->inputs[0];
	onnx_tensor_t* y = n->outputs[0];
	T* px = (T*)x->datas;
	T* py = (T*)y->datas;

	for (size_t i = 0, l = y->ndata; i < l; i++) {
		if (px[i] > 0)
			py[i] = 1;
		else if (px[i] < 0)
			py[i] = -1;
		else
			py[i] = 0;
	}
}

void resolver_default_op_Sign(onnx_node_t* n)
{
	if (n->opset >= 13) {
		n->ope = onnx_ope_type_selector{
			.int8_ = Sign_generic<int8_t>,
			.int16_ = Sign_generic<int16_t>,
			.int32_ = Sign_generic<int32_t>,
			.int64_ = Sign_generic<int64_t>,
			.uint8_ = Sign_generic<uint8_t>,
			.uint16_ = Sign_generic<uint16_t>,
			.uint32_ = Sign_generic<uint32_t>,
			.uint64_ = Sign_generic<uint64_t>,
			.bfloat16_ = Sign_generic<bfloat16_t>,
			.float16_ = Sign_generic<float16_t>,
			.float32_ = Sign_generic<float>,
			.float64_ = Sign_generic<double>,
		}.select(n->inputs[0]->type);
	}else if (n->opset >= 9) {
		n->ope = onnx_ope_type_selector{
			.int8_ = Sign_generic<int8_t>,
			.int16_ = Sign_generic<int16_t>,
			.int32_ = Sign_generic<int32_t>,
			.int64_ = Sign_generic<int64_t>,
			.uint8_ = Sign_generic<uint8_t>,
			.uint16_ = Sign_generic<uint16_t>,
			.uint32_ = Sign_generic<uint32_t>,
			.uint64_ = Sign_generic<uint64_t>,
			.float16_ = Sign_generic<float16_t>,
			.float32_ = Sign_generic<float>,
			.float64_ = Sign_generic<double>,
		}.select(n->inputs[0]->type);
	}
	if (n->ope) {
		n->init = Sign_init;
		n->exit = Sign_exit;
		n->reshape = Sign_reshape;
	}
}
