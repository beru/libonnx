#include <onnx.h>
#include "util.h"

namespace onnx {

struct Equal_operator : public operator_t {

	bool init() override {
		return is_inout_size(2, 1);
	}

	bool reshape() override {
		tensor_t* y = n->outputs[0];
		const tensor_t* a = n->inputs[0];
		const tensor_t* b = n->inputs[1];

		return y->reshape_multi_broadcast(a, b, ONNX_TENSOR_TYPE_BOOL);
	}

	template <typename T>
	void exec() {
		tensor_t* y = n->outputs[0];
		const tensor_t* a = n->inputs[0];
		const tensor_t* b = n->inputs[1];
		bool_t* py = (bool_t*)y->data;

		for (size_t i = 0, l = y->ndata; i < l; i++) {
			const T* pa = (const T*)a->broadcast_map_address(y, i);
			const T* pb = (const T*)b->broadcast_map_address(y, i);
			py[i] = (*pa == *pb);
		}
	}

	void exec() override {
		if (n->opset >= 13) {
			typed_exec<Equal_operator,
				bool_t,
				int8_t, int16_t, int32_t, int64_t,
				uint8_t, uint16_t, uint32_t, uint64_t,
				bfloat16_t, float16_t, float, double
			>(n->inputs[0]->type);
		}else if (n->opset >= 11) {
			typed_exec<Equal_operator,
				bool_t,
				int8_t, int16_t, int32_t, int64_t,
				uint8_t, uint16_t, uint32_t, uint64_t,
				float16_t, float, double
			>(n->inputs[0]->type);
		}else if (n->opset >= 7) {
			typed_exec<Equal_operator,
				bool_t,
				int32_t, int64_t
			>(n->inputs[0]->type);
		}else if (n->opset >= 1) {
		}
	}

};

void resolver_default_op_Equal(node_t* n)
{
	n->ope = std::make_shared<Equal_operator>();
}

} // namespace onnx
