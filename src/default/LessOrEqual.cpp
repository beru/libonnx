#include <onnx.h>
#include "util.h"

namespace {

bool LessOrEqual_init(onnx_node_t* n)
{
	return is_inout_size(n, 2, 1);
}

int LessOrEqual_reshape(onnx_node_t* n)
{
	onnx_tensor_t* y = n->outputs[0];
	onnx_tensor_t* a = n->inputs[0];
	onnx_tensor_t* b = n->inputs[1];

	return y->reshape_multi_broadcast(a, b, ONNX_TENSOR_TYPE_BOOL);
}

template <typename T>
void LessOrEqual_generic(onnx_node_t* n)
{
	onnx_tensor_t* y = n->outputs[0];
	onnx_tensor_t* a = n->inputs[0];
	onnx_tensor_t* b = n->inputs[1];
	uint8_t* py = (uint8_t*)y->data;

	for (size_t i = 0, l = y->ndata; i < l; i++) {
		T* pa = (T*)a->broadcast_map_address(y, i);
		T* pb = (T*)b->broadcast_map_address(y, i);
		py[i] = (*pa <= *pb) ? 1 : 0;
	}
}

GEN_HOLEDR_TYPE(holder, LessOrEqual_generic)

} // namespace

void resolver_default_op_LessOrEqual(onnx_node_t* n)
{
	if (n->opset >= 12) {
		n->ope = onnx_ope_type_select<holder,
			int8_t, int16_t, int32_t, int64_t,
			uint8_t, uint16_t, uint32_t, uint64_t,
			float16_t, float, double
		>(n->inputs[0]->type);
	}
	if (n->ope) {
		n->init = LessOrEqual_init;
		n->reshape = LessOrEqual_reshape;
	}
}
