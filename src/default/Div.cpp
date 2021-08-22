#include <onnx.h>
#include "float16.h"
#include "bfloat16.h"

static int Div_init(onnx_node_t* n)
{
	if ((n->inputs.size() == 2) && (n->outputs.size() == 1))
		return 1;
	return 0;
}

static int Div_exit(onnx_node_t* n)
{
	return 1;
}

static int Div_reshape(onnx_node_t* n)
{
	onnx_tensor_t* y = n->outputs[0];
	onnx_tensor_t* a = n->inputs[0];
	onnx_tensor_t* b = n->inputs[1];

	return y->reshape_multi_broadcast(a, b, a->type);
}

template <typename T>
static void Div_generic(onnx_node_t* n)
{
	onnx_tensor_t* y = n->outputs[0];
	onnx_tensor_t* a = n->inputs[0];
	onnx_tensor_t* b = n->inputs[1];
	T* py = (T*)y->datas;
	T* pa;
	T* pb;

	for (size_t i = 0, l = y->ndata; i < l; i++) {
		pa = (T*)a->broadcast_map_address(y, i);
		pb = (T*)b->broadcast_map_address(y, i);
		py[i] = *pa / *pb;
	}
}

void resolver_default_op_Div(onnx_node_t* n)
{
	if (n->opset >= 14) {
		n->ope = onnx_ope_type_selector{
			.int8_ = Div_generic<int8_t>,
			.int16_ = Div_generic<int16_t>,
			.int32_ = Div_generic<int32_t>,
			.int64_ = Div_generic<int64_t>,
			.uint8_ = Div_generic<uint8_t>,
			.uint16_ = Div_generic<uint16_t>,
			.uint32_ = Div_generic<uint32_t>,
			.uint64_ = Div_generic<uint64_t>,
			.bfloat16_ = Div_generic<bfloat16_t>,
			.float16_ = Div_generic<float16_t>,
			.float32_ = Div_generic<float>,
			.float64_ = Div_generic<double>,
		}.select(n->inputs[0]->type);
	}else if (n->opset >= 13) {
		n->ope = onnx_ope_type_selector{
			.int32_ = Div_generic<int32_t>,
			.int64_ = Div_generic<int64_t>,
			.uint32_ = Div_generic<uint32_t>,
			.uint64_ = Div_generic<uint64_t>,
			.bfloat16_ = Div_generic<bfloat16_t>,
			.float16_ = Div_generic<float16_t>,
			.float32_ = Div_generic<float>,
			.float64_ = Div_generic<double>,
		}.select(n->inputs[0]->type);
	}else if (n->opset >= 7) {
		n->ope = onnx_ope_type_selector{
			.int32_ = Div_generic<int32_t>,
			.int64_ = Div_generic<int64_t>,
			.uint32_ = Div_generic<uint32_t>,
			.uint64_ = Div_generic<uint64_t>,
			.float16_ = Div_generic<float16_t>,
			.float32_ = Div_generic<float>,
			.float64_ = Div_generic<double>,
		}.select(n->inputs[0]->type);
	}else if (n->opset >= 6) {
	}else if (n->opset >= 1) {
	}
	if (n->ope) {
		n->init = Div_init;
		n->exit = Div_exit;
		n->reshape = Div_reshape;
	}
}
