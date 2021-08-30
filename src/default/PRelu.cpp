#include <onnx.h>
#include "util.h"

namespace {

template <typename T>
void PRelu_generic(onnx_node_t* n)
{
	onnx_tensor_t* y = n->outputs[0];
	onnx_tensor_t* a = n->inputs[0];
	onnx_tensor_t* b = n->inputs[1];
	T* py = (T*)y->data;
	T* pa = (T*)a->data;;

	for (size_t i = 0, l = y->ndata; i < l; i++) {
		if (pa[i] < 0) {
			T* pb = (T*)b->broadcast_map_address(y, i);
			py[i] = pa[i] * (*pb);
		}
		else
			py[i] = pa[i];
	}
}

GEN_HOLEDR_TYPE(holder, PRelu_generic)

} // namespace

void resolver_default_op_PRelu(onnx_node_t* n)
{
	if (n->opset >= 9) {
		n->ope = onnx_ope_type_select<holder,
			int32_t, int64_t,
			uint32_t, uint64_t,
			float16_t, float, double
		>(n->inputs[0]->type);
	}else if (n->opset >= 7) {
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
		n->init = [](onnx_node_t* n){
			return is_inout_size(n, 2, 1);
		};
	}
}
