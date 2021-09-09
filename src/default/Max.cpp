#include <onnx.h>
#include "util.h"

namespace onnx {

namespace {

int Max_reshape(node_t* n)
{
	tensor_t* y = n->outputs[0];

	if (!y->reshape_identity(n->inputs[0]))
		return 0;
	for (int i = 1; i < n->inputs.size(); i++) {
		if (!y->reshape_multi_broadcast(y, n->inputs[i], y->type))
			return 0;
	}
	return 1;
}

template <typename T>
void Max_generic(node_t* n)
{
	tensor_t* y = n->outputs[0];
	const tensor_t* x;
	T* py = (T*)y->data;

	for (size_t i = 0, l = y->ndata; i < l; i++) {
		T maxv = std::numeric_limits<T>::min();
		for (int j = 0; j < n->inputs.size(); j++) {
			x = n->inputs[j];
			const T* px = (const T*)x->broadcast_map_address(y, i);
			if (*px > maxv)
				maxv = *px;
		}
		py[i] = maxv;
	}
}

GEN_HOLEDR_TYPE(holder, Max_generic)

} // namespace

void resolver_default_op_Max(node_t* n)
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
		n->reshape = Max_reshape;
	}
}

} // namespace onnx
