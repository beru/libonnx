#include <onnx.h>
#include "util.h"

namespace onnx {

namespace {

int Min_reshape(node_t* n)
{
	tensor_t* y = n->outputs[0];

	if (!y->reshape_identity(n->inputs[0]))
		return 0;
	for (size_t i = 1; i < n->inputs.size(); i++) {
		if (!y->reshape_multi_broadcast(y, n->inputs[i], y->type))
			return 0;
	}
	return 1;
}

template <typename T>
void Min_generic(node_t* n)
{
	tensor_t* y = n->outputs[0];
	T* py = (T*)y->data;
	for (size_t i = 0, l = y->ndata; i < l; i++) {
		T minv = std::numeric_limits<T>::max();
		for (size_t j = 0; j < n->inputs.size(); j++) {
			tensor_t* x = n->inputs[j];
			T* px = (T*)x->broadcast_map_address(y, i);
			if (*px < minv)
				minv = *px;
		}
		py[i] = minv;
	}
}

GEN_HOLEDR_TYPE(holder, Min_generic)

} // namespace

void resolver_default_op_Min(node_t* n)
{
	if (n->opset >= 13) {
		n->ope = ope_type_select<holder,
			int8_t, int16_t, int32_t, int64_t,
			uint8_t, uint16_t, uint32_t, uint64_t,
			bfloat16_t, float16_t, float, double
		>(n->inputs[0]->type);
	}else if (n->opset >= 12) {
		n->ope = ope_type_select<holder,
			int8_t, int16_t, int32_t, int64_t,
			uint8_t, uint16_t, uint32_t, uint64_t,
			float16_t, float, double
		>(n->inputs[0]->type);
	}else if (n->opset >= 8) {
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
			return (n->inputs.size() >= 1) && (n->outputs.size() == 1);
		};
		n->reshape = Min_reshape;
	}
}

} // namespace onnx
