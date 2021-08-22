#include <onnx.h>
#include "float16.h"
#include "bfloat16.h"

static int Greater_init(onnx_node_t* n)
{
	if ((n->inputs.size() == 2) && (n->outputs.size() == 1))
		return 1;
	return 0;
}

static int Greater_exit(onnx_node_t* n)
{
	return 1;
}

static int Greater_reshape(onnx_node_t* n)
{
	onnx_tensor_t* y = n->outputs[0];
	onnx_tensor_t* a = n->inputs[0];
	onnx_tensor_t* b = n->inputs[1];

	return y->reshape_multi_broadcast(a, b, ONNX_TENSOR_TYPE_BOOL);
}

template <typename T>
static void Greater_generic(onnx_node_t* n)
{
	onnx_tensor_t* y = n->outputs[0];
	onnx_tensor_t* a = n->inputs[0];
	onnx_tensor_t* b = n->inputs[1];
	uint8_t* py = (uint8_t*)y->datas;
	T* pa;
	T* pb;

	for (size_t i = 0, l = y->ndata; i < l; i++) {
		pa = (T*)a->broadcast_map_address(y, i);
		pb = (T*)b->broadcast_map_address(y, i);
		py[i] = (*pa > *pb) ? 1 : 0;
	}
}

void resolver_default_op_Greater(onnx_node_t* n)
{
	if (n->opset >= 13) {
		n->ope = onnx_ope_type_selector{
			.int8_ = Greater_generic<int8_t>,
			.int16_ = Greater_generic<int16_t>,
			.int32_ = Greater_generic<int32_t>,
			.int64_ = Greater_generic<int64_t>,
			.uint8_ = Greater_generic<uint8_t>,
			.uint16_ = Greater_generic<uint16_t>,
			.uint32_ = Greater_generic<uint32_t>,
			.uint64_ = Greater_generic<uint64_t>,
			.bfloat16_ = Greater_generic<bfloat16_t>,
			.float16_ = Greater_generic<float16_t>,
			.float32_ = Greater_generic<float>,
			.float64_ = Greater_generic<double>,
		}.select(n->inputs[0]->type);
	}else if (n->opset >= 9) {
		n->ope = onnx_ope_type_selector{
			.int8_ = Greater_generic<int8_t>,
			.int16_ = Greater_generic<int16_t>,
			.int32_ = Greater_generic<int32_t>,
			.int64_ = Greater_generic<int64_t>,
			.uint8_ = Greater_generic<uint8_t>,
			.uint16_ = Greater_generic<uint16_t>,
			.uint32_ = Greater_generic<uint32_t>,
			.uint64_ = Greater_generic<uint64_t>,
			.float16_ = Greater_generic<float16_t>,
			.float32_ = Greater_generic<float>,
			.float64_ = Greater_generic<double>,
		}.select(n->inputs[0]->type);
	}else if (n->opset >= 7) {
		n->ope = onnx_ope_type_selector{
			.float16_ = Greater_generic<float16_t>,
			.float32_ = Greater_generic<float>,
			.float64_ = Greater_generic<double>,
		}.select(n->inputs[0]->type);
	}else if (n->opset >= 1) {
	}
	if (n->ope) {
		n->init = Greater_init;
		n->exit = Greater_exit;
		n->reshape = Greater_reshape;
	}
}
