#include <onnx.h>
#include "util.h"

namespace onnx {

namespace {

struct Where_operator : public operator_t {

	bool init() override {
		return is_inout_size(3, 1);
	}

	bool reshape() override {
		tensor_t* y = n->outputs[0];
		int i;

		if (!y->reshape_identity(n->inputs[n->inputs.size() - 1]))
			return false;
		for (i = n->inputs.size() - 2; i >= 0; i--) {
			if (!y->reshape_multi_broadcast(y, n->inputs[i], y->type))
				return false;
		}
		return true;
	}

	template <typename T>
	void exec() {
		tensor_t* y = n->outputs[0];
		const tensor_t* x0 = n->inputs[0];
		const tensor_t* x1 = n->inputs[1];
		const tensor_t* x2 = n->inputs[2];
		T* py = (T*)y->data;
		T* px;

		for (size_t i = 0, l = y->ndata; i < l; i++) {
			uint8_t* c = (uint8_t*)x0->broadcast_map_address(y, i);
			if (*c)
				px = (T*)x1->broadcast_map_address(y, i);
			else
				px = (T*)x2->broadcast_map_address(y, i);
			py[i] = *px;
		}
	}

	void exec() override {
		if (n->opset >= 9) {
			if (n->inputs.size() == 3) {
				typed_exec<Where_operator,
					bool_t,
					uint8_t, uint16_t, uint32_t, uint64_t,
					int8_t, int16_t, int32_t, int64_t,
					float16_t, float, double, bfloat16_t,
					std::complex<float>, std::complex<double>,
					std::string
				>(n->inputs[0]->type);
			}
		}
	}
};

} // namespace {

void resolver_default_op_Where(node_t* n)
{
	n->ope = std::make_shared<Where_operator>();
}

} // namespace onnx
