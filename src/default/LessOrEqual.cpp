#include <onnx.h>
#include "float16.h"

static int LessOrEqual_init(onnx_node_t* n)
{
	if ((n->inputs.size() == 2) && (n->outputs.size() == 1))
		return 1;
	return 0;
}

static int LessOrEqual_exit(onnx_node_t* n)
{
	return 1;
}

static int LessOrEqual_reshape(onnx_node_t* n)
{
	onnx_tensor_t* y = n->outputs[0];
	onnx_tensor_t* a = n->inputs[0];
	onnx_tensor_t* b = n->inputs[1];

	return y->reshape_multi_broadcast(a, b, ONNX_TENSOR_TYPE_BOOL);
}

template <typename T>
static void LessOrEqual_generic(onnx_node_t* n)
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
		py[i] = (*pa <= *pb) ? 1 : 0;
	}
}

void resolver_default_op_LessOrEqual(onnx_node_t* n)
{
	if (n->opset >= 12) {
		n->ope = onnx_ope_type_selector{
			.int8_ = LessOrEqual_generic<int8_t>,
			.int16_ = LessOrEqual_generic<int16_t>,
			.int32_ = LessOrEqual_generic<int32_t>,
			.int64_ = LessOrEqual_generic<int64_t>,
			.uint8_ = LessOrEqual_generic<uint8_t>,
			.uint16_ = LessOrEqual_generic<uint16_t>,
			.uint32_ = LessOrEqual_generic<uint32_t>,
			.uint64_ = LessOrEqual_generic<uint64_t>,
			.float16_ = LessOrEqual_generic<float16_t>,
			.float32_ = LessOrEqual_generic<float>,
			.float64_ = LessOrEqual_generic<double>,
		}.select(n->inputs[0]->type);
	}
	if (n->ope) {
		n->init = LessOrEqual_init;
		n->exit = LessOrEqual_exit;
		n->reshape = LessOrEqual_reshape;
	}
}
