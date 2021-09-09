#include <onnx.h>
#include "util.h"

namespace onnx {

namespace {

template <typename T>
void PRelu_generic(node_t* n)
{
	tensor_t* y = n->outputs[0];
	const tensor_t* a = n->inputs[0];
	const tensor_t* b = n->inputs[1];
	T* py = (T*)y->data;
	const T* pa = (const T*)a->data;;

	for (size_t i = 0, l = y->ndata; i < l; i++) {
		if (pa[i] < 0) {
			const T* pb = (const T*)b->broadcast_map_address(y, i);
			py[i] = pa[i] * (*pb);
		}
		else
			py[i] = pa[i];
	}
}

GEN_HOLEDR_TYPE(holder, PRelu_generic)

} // namespace

void resolver_default_op_PRelu(node_t* n)
{
	if (n->opset >= 9) {
		n->ope = ope_type_select<holder,
			int32_t, int64_t,
			uint32_t, uint64_t,
			float16_t, float, double
		>(n->inputs[0]->type);
	}else if (n->opset >= 7) {
		n->ope = ope_type_select<holder,
			float16_t, float, double
		>(n->inputs[0]->type);
	}else if (n->opset >= 6) {
		n->ope = ope_type_select<holder,
			float16_t, float, double
		>(n->inputs[0]->type);
	}else if (n->opset >= 1) {
		n->ope = ope_type_select<holder,
			float16_t, float, double
		>(n->inputs[0]->type);
	}
	if (n->ope) {
		n->init = [](node_t* n){
			return is_inout_size(n, 2, 1);
		};
	}
}

} // namespace onnx
