#include <onnx.h>
#include "util.h"

namespace onnx {

template <typename T>
struct Min_operator : public operator_t {

	bool init() override {
		return (n->inputs.size() >= 1) && (n->outputs.size() == 1);
	}

	bool reshape() override {
		tensor_t* y = n->outputs[0];
		if (!y->reshape_identity(n->inputs[0]))
			return false;
		for (size_t i = 1; i < n->inputs.size(); i++) {
			if (!y->reshape_multi_broadcast(y, n->inputs[i], y->type))
				return false;
		}
		return true;
	}

	void exec() override {
		tensor_t* y = n->outputs[0];
		T* py = (T*)y->data;
		for (size_t i = 0, l = y->ndata; i < l; i++) {
			T minv = std::numeric_limits<T>::max();
			for (size_t j = 0; j < n->inputs.size(); j++) {
				const tensor_t* x = n->inputs[j];
				const T* px = (const T*)x->broadcast_map_address(y, i);
				if (*px < minv)
					minv = *px;
			}
			py[i] = minv;
		}
	}

};

void resolver_default_op_Min(node_t* n)
{
	if (n->opset >= 13) {
		n->ope = ope_type_select<Min_operator,
			int8_t, int16_t, int32_t, int64_t,
			uint8_t, uint16_t, uint32_t, uint64_t,
			bfloat16_t, float16_t, float, double
		>(n->inputs[0]->type);
	}else if (n->opset >= 12) {
		n->ope = ope_type_select<Min_operator,
			int8_t, int16_t, int32_t, int64_t,
			uint8_t, uint16_t, uint32_t, uint64_t,
			float16_t, float, double
		>(n->inputs[0]->type);
	}else if (n->opset >= 8) {
		n->ope = ope_type_select<Min_operator,
			float16_t, float, double
		>(n->inputs[0]->type);
	}else if (n->opset >= 6) {
		n->ope = ope_type_select<Min_operator,
			float16_t, float, double
		>(n->inputs[0]->type);
	}else if (n->opset >= 1) {
		n->ope = ope_type_select<Min_operator,
			float16_t, float, double
		>(n->inputs[0]->type);
	}
}

} // namespace onnx
