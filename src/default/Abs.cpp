#include <onnx.h>
#include "util.h"

namespace {

bool Abs_init(onnx_node_t* n)
{
	return is_inout_size(n, 1, 1);
}

template <typename T>
void Abs_generic(onnx_node_t* n)
{
	onnx_tensor_t* x = n->inputs[0];
	onnx_tensor_t* y = n->outputs[0];
	T* px = (T*)x->data;
	T* py = (T*)y->data;

	for (size_t i = 0, l = y->ndata; i < l; i++) {
		if constexpr (std::is_signed_v<T>) {
			py[i] = abs(px[i]);
		}else {
			py[i] = px[i];
		}
	}
}

GEN_HOLEDR_TYPE(holder, Abs_generic)

} // namespace {

void resolver_default_op_Abs(onnx_node_t* n)
{
	if (n->opset >= 13) {
		n->ope = onnx_ope_type_select<holder,
			uint8_t, uint16_t, uint32_t, uint64_t,
			int8_t, int16_t, int32_t, int64_t,
			float16_t, float, double, bfloat16_t
		>(n->inputs[0]->type);
	}else if (n->opset >= 6) {
		n->ope = onnx_ope_type_select<holder,
			uint8_t, uint16_t, uint32_t, uint64_t,
			int8_t, int16_t, int32_t, int64_t,
			float16_t, float, double
		>(n->inputs[0]->type);
	}else if (n->opset >= 1) {
		n->ope = onnx_ope_type_select<holder,
			float16_t, float, double
		>(n->inputs[0]->type);
	}
	if (n->ope) {
		n->init = Abs_init;
	}
}
