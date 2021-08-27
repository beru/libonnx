#include <onnx.h>
#include "util.h"

namespace {

bool Max_init(onnx_node_t* n)
{
	return (n->inputs.size() >= 1) && (n->outputs.size() == 1);
}

int Max_reshape(onnx_node_t* n)
{
	onnx_tensor_t* y = n->outputs[0];

	if (!y->reshape_identity(n->inputs[0]))
		return 0;
	for (int i = 1; i < n->inputs.size(); i++) {
		if (!y->reshape_multi_broadcast(y, n->inputs[i], y->type))
			return 0;
	}
	return 1;
}

template <typename T>
void Max_generic(onnx_node_t* n)
{
	onnx_tensor_t* y = n->outputs[0];
	onnx_tensor_t* x;
	T* py = (T*)y->data;

	for (size_t i = 0, l = y->ndata; i < l; i++) {
		T maxv = std::numeric_limits<T>::min();
		for (int j = 0; j < n->inputs.size(); j++) {
			x = n->inputs[j];
			T* px = (T*)x->broadcast_map_address(y, i);
			if (*px > maxv)
				maxv = *px;
		}
		py[i] = maxv;
	}
}

GEN_HOLEDR_TYPE(holder, Max_generic)

} // namespace

void resolver_default_op_Max(onnx_node_t* n)
{
	if (n->opset >= 13) {
		n->ope = onnx_ope_type_select<holder,
			int8_t, int16_t, int32_t, int64_t,
			uint8_t, uint16_t, uint32_t, uint64_t,
			bfloat16_t, float16_t, float, double
		>(n->inputs[0]->type);
	}else if (n->opset >= 12) {
		n->ope = onnx_ope_type_select<holder,
			int8_t, int16_t, int32_t, int64_t,
			uint8_t, uint16_t, uint32_t, uint64_t,
			float16_t, float, double
		>(n->inputs[0]->type);
	}else if (n->opset >= 8) {
		n->ope = onnx_ope_type_select<holder,
			float16_t, float, double
		>(n->inputs[0]->type);
	}else if (n->opset >= 6) {
		n->ope = onnx_ope_type_select<holder,
			float16_t, float, double
		>(n->inputs[0]->type);
	}else if (n->opset >= 1) {
		n->ope = onnx_ope_type_select<holder,
			float16_t, float, double
		>(n->inputs[0]->type);
	}
	if (n->ope) {
		n->init = Max_init;
		n->reshape = Max_reshape;
	}
}
