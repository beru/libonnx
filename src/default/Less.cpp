#include <onnx.h>
#include "util.h"

namespace onnx {

template <typename T>
struct Less_operator : public operator_t {

	bool init() override {
		return is_inout_size(2, 1);
	}

	bool reshape() override {
		tensor_t* y = n->outputs[0];
		const tensor_t* a = n->inputs[0];
		const tensor_t* b = n->inputs[1];
		return y->reshape_multi_broadcast(a, b, ONNX_TENSOR_TYPE_BOOL);
	}

	void exec() override {
		tensor_t* y = n->outputs[0];
		const tensor_t* a = n->inputs[0];
		const tensor_t* b = n->inputs[1];
		uint8_t* py = (uint8_t*)y->data;

		for (size_t i = 0, l = y->ndata; i < l; i++) {
			const T* pa = (const T*)a->broadcast_map_address(y, i);
			const T* pb = (const T*)b->broadcast_map_address(y, i);
			py[i] = (*pa < *pb) ? 1 : 0;
		}
	}

};

void resolver_default_op_Less(node_t* n)
{
	if (n->opset >= 13) {
		n->ope = ope_type_select<Less_operator,
			int8_t, int16_t, int32_t, int64_t,
			uint8_t, uint16_t, uint32_t, uint64_t,
			bfloat16_t, float16_t, float, double
		>(n->inputs[0]->type);
	}else if (n->opset >= 9) {
		n->ope = ope_type_select<Less_operator,
			int8_t, int16_t, int32_t, int64_t,
			uint8_t, uint16_t, uint32_t, uint64_t,
			float16_t, float, double
		>(n->inputs[0]->type);
	}else if (n->opset >= 7) {
		n->ope = ope_type_select<Less_operator,
			float16_t, float, double
		>(n->inputs[0]->type);
	}else if (n->opset >= 1) {
	}
}

} // namespace onnx
