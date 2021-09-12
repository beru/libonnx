#include <onnx.h>
#include "util.h"

namespace onnx {

namespace {

struct PRelu_operator : public operator_t {

	bool init() override {
		return is_inout_size(2, 1);
	}

	template <typename T>
	void exec() {
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

	void exec() override {
		if (n->opset >= 9) {
			TYPED_EXEC(n->inputs[0]->type,
				int32_t, int64_t,
				uint32_t, uint64_t,
				float16_t, float, double
			)
		}else if (n->opset >= 7) {
			TYPED_EXEC(n->inputs[0]->type,
				float16_t, float, double
			)
		}else if (n->opset >= 6) {
			TYPED_EXEC(n->inputs[0]->type,
				float16_t, float, double
			)
		}else if (n->opset >= 1) {
			TYPED_EXEC(n->inputs[0]->type,
				float16_t, float, double
			)
		}
	}

};

} // namespace {

void resolver_default_op_PRelu(node_t* n)
{
	n->ope = new PRelu_operator;
}

} // namespace onnx
